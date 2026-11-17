"""CrisisDynNet: interpretable multimodal dynamic causal framework.

Modules (Methods §3 of 实验设计方案.md):
  M1  DynCausalMixture  - mixture of sparse vector-autoregressions (person-centred, exogenous
                          calendar stress), estimated by EM. Each latent course type k has its own
                          directed lag-1 network A_k, so subtyping and pathway inference are joint.
                          Sparsity by adaptive-lasso reweighting; edge uncertainty by person-block
                          bootstrap.
  M2  dynamic indicators - rolling-window autocorrelation, variance and cross-lagged coupling
                          (critical-slowing-down family) -> early-warning tipping points.
  M3  GatedFusionGRU    - multimodal temporal encoder with modality gates (EMA, wearable, lab
                          context) + class posteriors -> 14-day crisis hazard.
  M4  explanations      - gate weights, integrated-gradient attributions per modality, bootstrap
                          network edges.
Baselines live in baselines.py.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DAILY = ["low_mood", "anxious", "stress", "rmssd_night", "resting_hr", "sleep_minutes", "scl_day_mean"]
NODE = {"low_mood": "Mood", "anxious": "Anx", "stress": "Stress", "rmssd_night": "HRV", "resting_hr": "HR",
        "sleep_minutes": "Sleep", "scl_day_mean": "EDA"}


# =============================================================== data assembly
def daily_panel(data, vars_=DAILY, n_days=None):
    """Person x day x var array (person-standardised), mask, exogenous inputs, calendar."""
    cal = data["academic_calendar"].copy(); cal["date"] = pd.to_datetime(cal["date"])
    n_days = n_days or len(cal)
    E = data["daily_ema"].copy(); W = data["daily_wearable"].copy()
    for d_ in (E, W):
        d_["date"] = pd.to_datetime(d_["date"])
    M = W.merge(E, on=["pid", "date"], how="left")
    M["day"] = (M.date - cal.date.min()).dt.days
    pids = sorted(data["participants"].pid)
    pidx = {p: i for i, p in enumerate(pids)}
    X = np.full((len(pids), n_days, len(vars_)), np.nan)
    ii = M.pid.map(pidx).values; dd = M.day.values
    ok = (dd >= 0) & (dd < n_days)
    for j, v in enumerate(vars_):
        X[ii[ok], dd[ok], j] = M[v].values[ok]
    X[..., vars_.index("rmssd_night")] = np.log(X[..., vars_.index("rmssd_night")])
    X[..., vars_.index("scl_day_mean")] = np.log(X[..., vars_.index("scl_day_mean")] + 0.05)
    mu = np.nanmean(X, 1, keepdims=True); sd = np.nanstd(X, 1, keepdims=True)
    Z = (X - mu) / np.where(sd > 0, sd, 1)
    mask = ~np.isnan(Z)
    per = cal.period.values[:n_days]
    U = np.c_[per == "examination", per == "winter_break", per == "national_holiday"].astype(float)
    ev = np.zeros((len(pids), n_days))
    if "life_event" in E:
        e2 = E[E.life_event > 0].copy(); e2["day"] = (e2.date - cal.date.min()).dt.days
        e2 = e2[(e2.day >= 0) & (e2.day < n_days)]
        ev[e2.pid.map(pidx).values, e2.day.values] = e2.life_event.values
    return dict(Z=Z, mask=mask, U=U, ev=ev, pids=pids, vars=vars_, mu=mu[:, 0], sd=sd[:, 0], cal=cal)


def lagged_design(Z, mask, U, ev, i, lag=1):
    """Lag design with missing data handled without listwise deletion.

    Predictors (lagged, person-standardised) are mean-imputed (0) and rows need >= half of them
    observed; each target uses only the rows where that target is observed. Returns y (NaN kept),
    x (imputed), target mask.
    """
    y = Z[i, lag:]; x = Z[i, :-lag].copy()
    mx = mask[i, :-lag]
    rows = mx.mean(1) >= 0.5
    x[~mx] = 0.0
    u = np.c_[U[lag:], ev[i, lag:], ev[i, :-lag]]
    X = np.c_[x, u][rows]; Y = y[rows]; M = mask[i, lag:][rows]
    return Y, X, M


# =============================================================== M1 mixture of sparse VARs
class DynCausalMixture:
    def __init__(self, K=4, ridge=2.0, alasso=0.15, n_iter=40, n_init=6, feat_weight=1.0, seed=0, temper=None):
        self.K, self.ridge, self.alasso, self.n_iter, self.n_init = K, ridge, alasso, n_iter, n_init
        self.feat_weight, self.seed, self.temper = feat_weight, seed, temper

    # ----- per-person sufficient statistics (per target, observed rows only)
    def _prep(self, P, F):
        G, H, YY, NN = [], [], [], []
        for i in range(P["Z"].shape[0]):
            y, x, m = lagged_design(P["Z"], P["mask"], P["U"], P["ev"], i)
            p = y.shape[1]; q = x.shape[1]
            g = np.zeros((p, q, q)); h = np.zeros((p, q)); yy = np.zeros(p); nn = np.zeros(p)
            for j in range(p):
                o = m[:, j]
                if o.sum() == 0:
                    continue
                xo = x[o]; yo = y[o, j]
                g[j] = xo.T @ xo; h[j] = xo.T @ yo; yy[j] = yo @ yo; nn[j] = o.sum()
            G.append(g); H.append(h); YY.append(yy); NN.append(nn)
        self.G, self.H, self.YY, self.NN = map(np.array, (G, H, YY, NN))      # (n,p,q,q) (n,p,q) (n,p) (n,p)
        self.p = self.G.shape[1]; self.q = self.G.shape[2]
        self.F = StandardScaler().fit_transform(np.nan_to_num(F)) if F is not None else None
        self.n_eff = self.NN.mean(1)

    def _fit_class(self, w, idx=None):
        """Weighted ridge with adaptive-lasso reweighting; returns B (q x p) and sigma2 (p)."""
        G, H, YY, NN = (self.G, self.H, self.YY, self.NN) if idx is None else (self.G[idx], self.H[idx], self.YY[idx], self.NN[idx])
        XtX = np.einsum("n,npqr->pqr", w, G); XtY = np.einsum("n,npq->pq", w, H); n = np.einsum("n,np->p", w, NN)
        I = np.eye(self.q)
        B = np.stack([np.linalg.solve(XtX[j] + self.ridge * I, XtY[j]) for j in range(self.p)], 1)
        for _ in range(3):                                    # adaptive lasso via reweighted ridge
            pen = self.alasso * n[None, :] / (np.abs(B) + 1e-3) / 50
            B = np.stack([np.linalg.solve(XtX[j] + np.diag(pen[:, j]) + self.ridge * I, XtY[j]) for j in range(self.p)], 1)
        rss = np.einsum("n,np->p", w, YY) - 2 * np.einsum("pq,qp->p", XtY, B) + np.einsum("qp,pqr,rp->p", B, XtX, B)
        s2 = np.maximum(rss / np.maximum(n, 1), 1e-3)
        return B, s2

    def _loglik(self, B, s2):
        rss = self.YY - 2 * np.einsum("npq,qp->np", self.H, B) + np.einsum("qp,npqr,rp->np", B, self.G, B)
        return (-0.5 * (rss / s2[None, :] + self.NN * np.log(2 * np.pi * s2[None, :]))).sum(1)

    def fit(self, P, F=None):
        self._prep(P, F)
        n = len(self.G); rng = np.random.default_rng(self.seed)
        # initial partition: k-means on person-specific VAR coefficients (+ trajectory features)
        coefs = []
        for i in range(len(self.G)):
            b = np.stack([np.linalg.solve(self.G[i, j] + 5 * np.eye(self.q), self.H[i, j]) for j in range(self.p)], 1)
            coefs.append(b[:self.p].ravel())
        init_feat = StandardScaler().fit_transform(np.c_[np.array(coefs), self.F if self.F is not None else np.zeros((n, 0))])
        temper = self.temper or 1.0 / np.median(self.n_eff).clip(1)      # per-observation tempering
        best = None
        for r in range(self.n_init):
            lab = KMeans(self.K, n_init=3, random_state=self.seed + r).fit_predict(init_feat) if r % 2 == 0 \
                else rng.integers(0, self.K, n)
            R = np.eye(self.K)[lab] * 0.8 + 0.2 / self.K
            for it in range(self.n_iter):
                pi = R.mean(0)
                params = [self._fit_class(R[:, k]) for k in range(self.K)]
                LL = np.stack([self._loglik(B, s2) for B, s2 in params], 1) * temper * 40
                if self.F is not None and self.feat_weight > 0:
                    for k in range(self.K):
                        w = R[:, k] / R[:, k].sum()
                        m = (w[:, None] * self.F).sum(0); dF = self.F - m
                        S = (w[:, None, None] * dF[:, :, None] * dF[:, None, :]).sum(0) + 0.05 * np.eye(dF.shape[1])
                        Si = np.linalg.inv(S); _, ld = np.linalg.slogdet(S)
                        LL[:, k] += self.feat_weight * (-0.5 * (np.einsum("ni,ij,nj->n", dF, Si, dF) + ld))
                LL += np.log(pi + 1e-9)
                Rn = np.exp(LL - LL.max(1, keepdims=True)); Rn /= Rn.sum(1, keepdims=True)
                if np.abs(Rn - R).max() < 1e-4:
                    R = Rn; break
                R = Rn
            tot = float(np.log(np.exp(LL - LL.max(1, keepdims=True)).sum(1)).sum() + LL.max(1).sum())
            if best is None or tot > best[0]:
                best = (tot, R, params)
        self.loglik_, self.R_, self.params_ = best
        self.labels_ = self.R_.argmax(1)
        k_par = self.K * (self.q * self.p + self.p) + (self.K - 1) + (0 if self.F is None else self.K * 2 * self.F.shape[1])
        self.bic_ = -2 * self.loglik_ + k_par * np.log(n)
        return self

    def networks(self):
        """Lag-1 directed networks A_k (target x source) on the observed nodes."""
        return [B[:self.p].T for B, _ in self.params_]

    def bootstrap_edges(self, n_boot=100, seed=0):
        rng = np.random.default_rng(seed); n = len(self.G)
        out = np.zeros((n_boot, self.K, self.p, self.p))
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            for k in range(self.K):
                out[b, k] = self._fit_class(self.R_[idx, k], idx)[0][:self.p].T
        return out


def select_K(P, F, Ks=range(1, 7), seed=0, **kw):
    rows, models = [], {}
    for K in Ks:
        m = DynCausalMixture(K=K, seed=seed, **kw).fit(P, F)
        rows.append(dict(K=K, bic=m.bic_, loglik=m.loglik_, min_class=float(np.bincount(m.labels_, minlength=K).min() / len(m.labels_)),
                         entropy=float(1 - (-(m.R_ * np.log(m.R_ + 1e-12)).sum(1).mean()) / np.log(max(K, 2)))))
        models[K] = m
    return pd.DataFrame(rows), models


# =============================================================== trajectory features (person level)
def ar1(x):
    x = x[~np.isnan(x)]
    if len(x) < 8 or x.std() == 0:
        return np.nan
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def trajectory_features(P, raw=None):
    """Person-level evolution signatures (raw units where meaningful).

    level/slope of mood, HRV slope (vagal decline), response to major life events, day-to-day
    instability (MSSD), inertia (AR1), arousal-report dissociation (EDA when mood looks fine) and
    EDA-mood covariation. Uses only observed data.
    """
    Z = P["Z"]
    if raw is None:
        raw = Z * P["sd"][:, None, :] + P["mu"][:, None, :]
    vi = {v: j for j, v in enumerate(P["vars"])}
    T = raw.shape[1]; t = np.arange(T); rows = []
    for i in range(raw.shape[0]):
        m = raw[i, :, vi["low_mood"]]; hv = raw[i, :, vi["rmssd_night"]]
        ok = ~np.isnan(m); okh = ~np.isnan(hv)
        sl = np.polyfit(t[ok], m[ok], 1)[0] * 30 if ok.sum() > 20 else np.nan
        slh = np.polyfit(t[okh], hv[okh], 1)[0] * 30 if okh.sum() > 20 else np.nan
        evd = np.where(P["ev"][i] >= 2)[0]
        er = np.nanmean(m[evd[0]:evd[0] + 7]) - np.nanmean(m[max(0, evd[0] - 14):evd[0]]) if len(evd) else np.nan
        z = Z[i]
        dis = np.nanmean(z[:, vi["scl_day_mean"]] * (z[:, vi["low_mood"]] < 0.3))
        q = np.nanquantile(m, [.5, .9]) if ok.any() else [np.nan, np.nan]
        rows.append([q[0], sl, slh, er, np.nanstd(np.diff(m[ok])) if ok.sum() > 2 else np.nan, ar1(m), dis,
                     np.nanmean(z[:, vi["scl_day_mean"]] * z[:, vi["low_mood"]])])
    F = pd.DataFrame(rows, columns=["level", "slope", "hrv_slope", "event_resp", "instab", "inertia", "arousal_dissoc",
                                    "eda_mood_cov"], index=P["pids"])
    return F


# =============================================================== M2 dynamic indicators (early warning)
def rolling_indicators(z, win=21):
    """Rolling AR1, variance and cross-lagged HRV->mood coupling on person-standardised series."""
    T = len(z); out = np.full((T, 3), np.nan)
    m, h = z[:, 0], z[:, 3]
    for t in range(win, T):
        a = m[t - win:t]; b = h[t - win:t]
        ok = ~np.isnan(a)
        if ok.sum() < win * 0.6:
            continue
        out[t, 0] = ar1(a); out[t, 1] = np.nanvar(a)
        ok2 = ~np.isnan(a[1:]) & ~np.isnan(b[:-1])
        if ok2.sum() > 8:
            out[t, 2] = np.corrcoef(b[:-1][ok2], a[1:][ok2])[0, 1]
    return out


def kendall_trend(x):
    from scipy.stats import kendalltau
    ok = ~np.isnan(x)
    if ok.sum() < 6:
        return np.nan
    return kendalltau(np.arange(ok.sum()), x[ok]).statistic


# =============================================================== M3 gated multimodal fusion (torch)
def make_sequences(P, data, horizon=14, start=28):
    """Inputs (n, T, f), labels (n, T) = crisis detected in (t, t+horizon], valid mask."""
    Z = np.nan_to_num(P["Z"]); M = P["mask"].astype(float)
    groups = {"ema": [0, 1, 2], "wear": [3, 4, 5, 6]}
    n, T, _ = Z.shape
    ce = data["crisis_events"]
    det = np.full(n, np.inf)
    if len(ce):
        cal0 = P["cal"].date.min()
        for p, d in zip(ce.pid, pd.to_datetime(ce.detected_date)):
            det[P["pids"].index(p)] = min(det[P["pids"].index(p)], (d - cal0).days)
    y = np.zeros((n, T)); valid = np.zeros((n, T), bool)
    for i in range(n):
        for t in range(start, T - 1):
            if t >= det[i]:
                break
            y[i, t] = float(t < det[i] <= t + horizon); valid[i, t] = t + horizon < T or det[i] < T
    return Z, M, groups, y, valid, det


def fusion_model(n_in_groups, n_static, hidden=32):
    import torch
    import torch.nn as nn

    class GatedFusionGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.ModuleList([nn.GRU(2 * g, hidden, batch_first=True) for g in n_in_groups])
            self.gate = nn.Sequential(nn.Linear(hidden * len(n_in_groups) + n_static, len(n_in_groups)))
            self.head = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, xs, static, return_gates=False):
            hs = [enc(x)[0] for enc, x in zip(self.enc, xs)]                    # each (n, T, hidden)
            st = static[:, None, :].expand(-1, hs[0].shape[1], -1)
            g = torch.softmax(self.gate(torch.cat(hs + [st], -1)), -1)        # (n, T, G) modality gates
            h = sum(g[..., k:k + 1] * hs[k] for k in range(len(hs)))
            out = self.head(torch.cat([h, st], -1)).squeeze(-1)
            return (out, g) if return_gates else out

    return GatedFusionGRU()


def train_fusion(Xg, static, y, valid, train_idx, epochs=60, lr=3e-3, seed=0, device=None, hidden=32):
    import torch
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = fusion_model([x.shape[2] // 2 for x in Xg], static.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    tx = [torch.tensor(x[train_idx], dtype=torch.float32, device=device) for x in Xg]
    ts = torch.tensor(static[train_idx], dtype=torch.float32, device=device)
    ty = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
    tv = torch.tensor(valid[train_idx], dtype=torch.float32, device=device)
    pos = (ty * tv).sum() / tv.sum()
    pw = ((1 - pos) / pos.clamp(min=1e-3)).clamp(max=50)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = model(tx, ts)
        loss = (lossf(out, ty) * tv).sum() / tv.sum()
        loss.backward(); opt.step()
    return model


def predict_fusion(model, Xg, static, idx, gates=False):
    import torch
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        o = model([torch.tensor(x[idx], dtype=torch.float32, device=dev) for x in Xg],
                  torch.tensor(static[idx], dtype=torch.float32, device=dev), return_gates=gates)
    if gates:
        return torch.sigmoid(o[0]).cpu().numpy(), o[1].cpu().numpy()
    return torch.sigmoid(o).cpu().numpy()


def integrated_gradients(model, Xg, static, idx, steps=16):
    """Mean |IG| attribution per input group over valid days (explanation module M4)."""
    import torch
    dev = next(model.parameters()).device
    model.train()                                   # cuDNN RNN backward needs training mode (no dropout used)
    xs = [torch.tensor(x[idx], dtype=torch.float32, device=dev) for x in Xg]
    st = torch.tensor(static[idx], dtype=torch.float32, device=dev)
    tot = [torch.zeros_like(x) for x in xs]; tot_s = torch.zeros_like(st)
    for a in np.linspace(0.05, 1, steps):
        xa = [(a * x).requires_grad_(True) for x in xs]; sa = (a * st).requires_grad_(True)
        out = model(xa, sa).sum()
        grads = torch.autograd.grad(out, xa + [sa])
        for k in range(len(xs)):
            tot[k] += grads[k] / steps
        tot_s += grads[-1] / steps
    att = [(t * x).abs().mean().item() for t, x in zip(tot, xs)] + [(tot_s * st).abs().mean().item()]
    return att
