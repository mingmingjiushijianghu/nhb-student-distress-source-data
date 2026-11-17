"""Comparison methods (Results module 1): traditional ML, deep classification, Granger causality,
hidden Markov model, PCMCI and trajectory clustering. Each is implemented as it is commonly used."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats


# ------------------------------------------------------------------ causal edge baselines (per person)
def granger_person(z, alpha=0.05):
    """Pairwise bivariate Granger F-tests (lag 1). Returns (p x p) matrix of 1-p (target x source)."""
    T, p = z.shape; S = np.zeros((p, p))
    for j in range(p):
        for k in range(p):
            if j == k:
                continue
            y = z[1:, j]; x1 = z[:-1, j]; x2 = z[:-1, k]
            ok = ~np.isnan(y) & ~np.isnan(x1) & ~np.isnan(x2)
            if ok.sum() < 20:
                continue
            X0 = np.c_[np.ones(ok.sum()), x1[ok]]; X1 = np.c_[X0, x2[ok]]
            r0 = y[ok] - X0 @ np.linalg.lstsq(X0, y[ok], rcond=None)[0]
            r1 = y[ok] - X1 @ np.linalg.lstsq(X1, y[ok], rcond=None)[0]
            df = ok.sum() - 3
            F = (r0 @ r0 - r1 @ r1) / (r1 @ r1 / df)
            S[j, k] = 1 - stats.f.sf(F, 1, df)
    return S


def lasso_var_person(z):
    T, p = z.shape; S = np.zeros((p, p))
    y = z[1:]; x = z[:-1].copy()
    rows = (~np.isnan(x)).mean(1) >= 0.5                  # same missing-data handling as CrisisDynNet
    x = np.nan_to_num(x)
    for j in range(p):
        ok = rows & ~np.isnan(y[:, j])
        if ok.sum() < 25:
            continue
        m = LassoCV(cv=3, alphas=20).fit(x[ok], y[ok, j])
        S[j] = np.abs(m.coef_)
    np.fill_diagonal(S, 0)
    return S


def pcmci_person(z, tau_max=1):
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    z = z.copy(); mask = np.isnan(z); z[mask] = 0
    df = pp.DataFrame(z, mask=mask)
    pc = PCMCI(dataframe=df, cond_ind_test=ParCorr(mask_type="y"), verbosity=0)
    r = pc.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=0.2)
    pv = r["p_matrix"][:, :, 1]            # (source, target)
    S = (1 - pv).T                          # -> (target, source)
    np.fill_diagonal(S, 0)
    return S


def xcorr_person(z):
    T, p = z.shape; S = np.zeros((p, p))
    for j in range(p):
        for k in range(p):
            if j == k:
                continue
            a = z[1:, j]; b = z[:-1, k]; ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() > 10:
                S[j, k] = abs(np.corrcoef(a[ok], b[ok])[0, 1])
    return S


# ------------------------------------------------------------------ subtype baselines (person level)
def gmm_cluster(F, K, seed=0):
    X = StandardScaler().fit_transform(F.fillna(F.median()))
    return GaussianMixture(K, n_init=10, random_state=seed).fit_predict(X)


def granger_cluster(Z, K, seed=0):
    feats = []
    for i in range(Z.shape[0]):
        feats.append(granger_person(Z[i]).ravel())
    return KMeans(K, n_init=10, random_state=seed).fit_predict(StandardScaler().fit_transform(np.array(feats)))


def lcga_cluster(Z_raw_mood, K, seed=0):
    """Trajectory clustering on monthly means of mood (latent-class-growth-style)."""
    n, T = Z_raw_mood.shape
    bins = np.array_split(np.arange(T), 6)
    M = np.c_[[np.nanmean(Z_raw_mood[:, b], 1) for b in bins]].T
    M = np.where(np.isnan(M), np.nanmean(M, 1, keepdims=True), M)
    return GaussianMixture(K, n_init=10, random_state=seed).fit_predict(M)


def hmm_features(Z, n_states=4, seed=0):
    from hmmlearn.hmm import GaussianHMM
    X = np.nan_to_num(Z); lengths = [Z.shape[1]] * Z.shape[0]
    m = GaussianHMM(n_states, covariance_type="diag", n_iter=50, random_state=seed).fit(X.reshape(-1, Z.shape[2]), lengths)
    post = np.stack([m.predict_proba(X[i]) for i in range(Z.shape[0])])          # n, T, S
    occ = post.mean(1)
    trans = np.stack([(post[i, :-1, :, None] * post[i, 1:, None, :]).sum(0).ravel() for i in range(Z.shape[0])])
    return m, post, np.c_[occ, trans / trans.sum(1, keepdims=True)]


def hmm_cluster(Z, K, seed=0):
    _, _, f = hmm_features(Z, seed=seed)
    return KMeans(K, n_init=10, random_state=seed).fit_predict(StandardScaler().fit_transform(f))


def gru_ae_cluster(Z, K, seed=0, epochs=80):
    """GRU auto-encoder embedding + GMM (cf. AttentiveGRUAE, Soley et al. 2025)."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(np.nan_to_num(Z), dtype=torch.float32, device=dev)
    Mk = torch.tensor(~np.isnan(Z), dtype=torch.float32, device=dev)
    enc = nn.GRU(Z.shape[2], 16, batch_first=True).to(dev); dec = nn.GRU(16, Z.shape[2], batch_first=True).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=5e-3)
    for _ in range(epochs):
        opt.zero_grad()
        _, h = enc(X); rep = h[-1][:, None, :].expand(-1, Z.shape[1], -1)
        out, _ = dec(rep)
        loss = (((out - X) ** 2) * Mk).sum() / Mk.sum(); loss.backward(); opt.step()
    with torch.no_grad():
        emb = enc(X)[1][-1].cpu().numpy()
    return GaussianMixture(K, n_init=10, random_state=seed).fit_predict(StandardScaler().fit_transform(emb))


# ------------------------------------------------------------------ prediction baselines (person-day)
def rolling_summary(Z, win=14):
    n, T, p = Z.shape
    out = np.full((n, T, 2 * p + 1), np.nan)
    for t in range(T):
        lo = max(0, t - win + 1); w = Z[:, lo:t + 1]
        out[:, t, :p] = np.nanmean(w, 1); out[:, t, p:2 * p] = np.nanstd(w, 1); out[:, t, -1] = t
    return out


def flat(X, y, valid, idx):
    Xi = X[idx][valid[idx]]; yi = y[idx][valid[idx]]
    return Xi, yi


def fit_predict_gbm(Xs, y, valid, tr, te):
    Xtr, ytr = flat(Xs, y, valid, tr)
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, class_weight="balanced").fit(Xtr, ytr)
    n, T, f = Xs.shape
    return m.predict_proba(Xs[te].reshape(-1, f))[:, 1].reshape(len(te), T)


def fit_predict_static(S, y, valid, tr, te):
    """Traditional screening: baseline scale totals only (constant risk per person)."""
    T = y.shape[1]
    Xtr = np.repeat(S[tr][:, None, :], T, 1)[valid[tr]]; ytr = y[tr][valid[tr]]
    sc = StandardScaler().fit(S[tr])
    m = LogisticRegression(max_iter=500, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return np.repeat(m.predict_proba(sc.transform(S[te]))[:, 1][:, None], T, 1)


def fit_predict_lstm(Z, M, y, valid, tr, te, epochs=60, seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.concatenate([np.nan_to_num(Z), M], -1)

    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.rnn = nn.LSTM(X.shape[2], 32, batch_first=True); s.out = nn.Linear(32, 1)

        def forward(s, x):
            return s.out(s.rnn(x)[0]).squeeze(-1)

    net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    tx = torch.tensor(X[tr], dtype=torch.float32, device=dev); ty = torch.tensor(y[tr], dtype=torch.float32, device=dev)
    tv = torch.tensor(valid[tr], dtype=torch.float32, device=dev)
    pos = (ty * tv).sum() / tv.sum(); pw = ((1 - pos) / pos.clamp(min=1e-3)).clamp(max=50)
    lf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    for _ in range(epochs):
        opt.zero_grad(); l = (lf(net(tx), ty) * tv).sum() / tv.sum(); l.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(X[te], dtype=torch.float32, device=dev))).cpu().numpy()


def fit_predict_hmm(Z, y, valid, tr, te, seed=0):
    m, post, _ = hmm_features(Z, seed=seed)
    T = Z.shape[1]
    Xtr = post[tr][valid[tr]]; ytr = y[tr][valid[tr]]
    lr = LogisticRegression(max_iter=500, class_weight="balanced").fit(Xtr, ytr)
    return lr.predict_proba(post[te].reshape(-1, post.shape[2]))[:, 1].reshape(len(te), T)
