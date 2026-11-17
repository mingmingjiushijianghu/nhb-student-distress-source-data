"""Step 08 - Study 4: CrisisDynNet on the prospective cohort (data/), with baselines and ablations.

Results module 1 (method validation): subtype identification, causal-pathway recovery, 14-day crisis
prediction vs five comparison methods; ablations of modalities, temporal module, subtype posteriors
and lab context; interpretability (class networks, gates, attributions).
Results module 2 (mechanism): subtype profiles, lab pathway differences, tipping points, ecological
(exam / recruitment season) validation.
Design checks against ground truth (cache/cohort_truth*.npz/json) are written to
results/cohort_design_checks.csv and are NOT manuscript results.
"""

# --- 路径引导（让 pipeline/ 下的脚本能找到 common/ 与 config.py；可用 python pipeline/xx.py 直接运行）---
import sys as _NHB_SYS
from pathlib import Path as _NHB_PATH
_NHB_ROOT = _NHB_PATH(__file__).resolve().parents[1]
for _NHB_P in (_NHB_ROOT / "common", _NHB_ROOT):
    if str(_NHB_P) not in _NHB_SYS.path:
        _NHB_SYS.path.insert(0, str(_NHB_P))
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score, average_precision_score, adjusted_rand_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import config as C
import crisisdynnet as CD
import baselines as BL

RES = C.RES
SEED = C.SEED
NAMES = ["stable", "acute", "chronic", "latent"]
FW = 4.0          # 特征/VAR 证据权重，设计阶段固定（生成数据上做过 0.5–32 的网格；见 实验设计方案.md §7）


def _ari(a, b):
    """与真值比较的 ARI；无真值缓存时返回 NaN，其余列照常输出。"""
    return np.nan if a is None else adjusted_rand_score(a, b)


def load():
    keys = ["academic_calendar", "daily_ema", "daily_wearable", "participants", "crisis_events", "wave_questionnaires",
            "lab_epochs", "counselor_assessments", "weekly_phq4"]
    return {k: pd.read_csv(C.COHORT / f"{k}.csv") for k in keys}


def name_clusters(F, labels, K):
    """Pre-registered naming rule from evolution signatures (no ground truth)."""
    g = F.assign(c=labels).groupby("c").mean()
    names = {}
    z = (g - g.mean()) / g.std().replace(0, 1)
    order = list(g.index)
    stable = z["level"].idxmin(); names[stable] = "stable"
    rest = [c for c in order if c != stable]
    # theory-ordered (V21): the most specific physiological signature is assigned first
    if rest:
        chr_ = min(rest, key=lambda c: z.loc[c, "hrv_slope"]); names[chr_] = "chronic"; rest.remove(chr_)   # vagal decline
    if rest:
        lat = max(rest, key=lambda c: z.loc[c, "arousal_dissoc"]); names[lat] = "latent"; rest.remove(lat)  # arousal without report
    if rest:
        acu = max(rest, key=lambda c: z.loc[c, "event_resp"] + z.loc[c, "instab"]); names[acu] = "acute"; rest.remove(acu)
    for c in rest:
        names[c] = f"other{c}"
    return np.array([names[c] for c in labels])


def bootstrap_stability(P, F, K, n=20, fw=2.0):
    base = CD.DynCausalMixture(K=K, feat_weight=fw, seed=SEED).fit(P, F).labels_
    rng = np.random.default_rng(SEED); aris = []
    for b in range(n):
        idx = np.unique(rng.integers(0, len(base), len(base)))
        Pb = dict(P, Z=P["Z"][idx], mask=P["mask"][idx], ev=P["ev"][idx], pids=[P["pids"][i] for i in idx])
        lab = CD.DynCausalMixture(K=K, feat_weight=fw, seed=SEED + b + 1, n_init=3).fit(Pb, F[idx]).labels_
        aris.append(adjusted_rand_score(base[idx], lab))
    return np.array(aris)


def baseline_stability(fn, n=20):
    base = fn(None); rng = np.random.default_rng(SEED); aris = []
    for b in range(n):
        idx = np.unique(rng.integers(0, len(base), len(base)))
        aris.append(adjusted_rand_score(base[idx], fn(idx)))
    return np.array(aris)


def main():
    t0 = time.time()
    D = load()
    P = CD.daily_panel(D)
    n, T, p = P["Z"].shape
    F = CD.trajectory_features(P)
    Ff = F.fillna(F.median())
    # 真值缓存（cache/cohort_truth*.npz/json）是设计检验用的参考答案；缺失时
    # 与真值有关的列写 NaN、因果路径恢复整段跳过，其余分析照常进行。
    truth, graphs = None, None
    if C.has_truth():
        truth_npz = np.load(C.TRUTH_NPZ, allow_pickle=True)
        typ = dict(zip(truth_npz["pid"], truth_npz["type"]))
        truth = np.array([typ[q] for q in P["pids"]])
        graphs = json.loads(C.TRUTH_GRAPHS.read_text())
        print("ground truth found:", len(truth), "participants", flush=True)
    else:
        print("no truth cache - truth-based design checks are skipped", flush=True)
    checks, summary = [], {}

    # ================================================================ 1. subtype identification
    selK, models = CD.select_K(P, Ff.values, Ks=range(1, 7), seed=SEED, feat_weight=FW)
    selK.to_csv(RES / "cohort_K_selection.csv", index=False)
    K = int(selK.loc[selK.bic.idxmin(), "K"])
    summary["K_selected_bic"] = K
    K_eval = 4                                   # pre-registered hypothesis: stable + three crisis subtypes
    mdl = models[K_eval]
    lab = mdl.labels_
    sub = name_clusters(F, lab, K_eval)
    Xs = StandardScaler().fit_transform(Ff)
    rows = [dict(method="CrisisDynNet (mixture dynamic causal + signatures)", ari_truth=_ari(truth, lab),
                 silhouette=silhouette_score(Xs, lab) if len(set(lab)) > 1 else np.nan)]
    stab = bootstrap_stability(P, Ff.values, K_eval, fw=FW)
    rows[0]["boot_ari_mean"] = stab.mean(); rows[0]["boot_ari_sd"] = stab.std()
    Zr = P["Z"] * P["sd"][:, None, :] + P["mu"][:, None, :]
    conv = pd.DataFrame(np.c_[np.nanmean(Zr, 1), np.nanstd(Zr, 1)], index=P["pids"])
    base_fns = {
        "GMM on per-variable mean/SD (traditional ML)": lambda idx: BL.gmm_cluster(conv if idx is None else conv.iloc[idx], K_eval),
        "Trajectory clustering on monthly means (LCGA-type)": lambda idx: BL.lcga_cluster(Zr[:, :, 0] if idx is None else Zr[idx, :, 0], K_eval),
        "Granger-coefficient clustering": lambda idx: BL.granger_cluster(P["Z"] if idx is None else P["Z"][idx], K_eval),
        "Hidden Markov model + k-means": lambda idx: BL.hmm_cluster(P["Z"][:, :, [0, 3, 6]] if idx is None else P["Z"][idx][:, :, [0, 3, 6]], K_eval),
        "GRU auto-encoder + GMM (deep clustering)": lambda idx: BL.gru_ae_cluster(P["Z"] if idx is None else P["Z"][idx], K_eval),
    }
    for nm, fn in base_fns.items():
        lb = fn(None)
        st = baseline_stability(fn, n=10)
        rows.append(dict(method=nm, ari_truth=_ari(truth, lb), silhouette=silhouette_score(Xs, lb),
                         boot_ari_mean=st.mean(), boot_ari_sd=st.std()))
        print("subtype baseline", nm, round(rows[-1]["ari_truth"], 3) if truth is not None else "n/a", flush=True)
    # ablations of the subtype module
    lb = BL.gmm_cluster(Ff, K_eval)
    rows.append(dict(method="Ablation: evolution signatures only (no dynamic causal mixture)", ari_truth=_ari(truth, lb), silhouette=silhouette_score(Xs, lb)))
    for nm, kw, FF in [("Ablation: dynamic causal mixture only (no signatures)", dict(feat_weight=0.0), Ff.values),
                       ("Ablation: without physiology signatures", dict(feat_weight=FW), Ff.drop(columns=["hrv_slope", "arousal_dissoc", "eda_mood_cov"]).values)]:
        lb = CD.DynCausalMixture(K=K_eval, seed=SEED, **kw).fit(P, FF).labels_
        rows.append(dict(method=nm, ari_truth=_ari(truth, lb), silhouette=silhouette_score(Xs, lb)))
    subtab = pd.DataFrame(rows)
    subtab.to_csv(RES / "cohort_subtype_methods.csv", index=False)
    checks.append(dict(check="subtype ARI vs ground truth (CrisisDynNet)", value=rows[0]["ari_truth"]))
    checks.append(dict(check="subtype naming accuracy vs ground truth",
                       value=float((sub == truth).mean()) if truth is not None else np.nan))
    pd.DataFrame({"pid": P["pids"], "subtype": sub, **{f"post_{k}": mdl.R_[:, k] for k in range(K_eval)}}).to_csv(RES / "cohort_subtypes.csv", index=False)
    print("subtypes done", round(time.time() - t0), "s", flush=True)

    # ================================================================ 2. causal pathway recovery
    node_idx = {"d": 0, "h": 3, "e": 6, "s": 5}
    lat = ["d", "h", "e", "s"]; obs = [node_idx[v] for v in lat]
    nets = mdl.networks()
    # match classes to theory names via naming rule
    cls_name = {k: sub[lab == k][0] if (lab == k).any() else f"c{k}" for k in range(K_eval)}
    edge_rows = []
    if truth is not None:                        # 只有生成数据才有真值网络可对照
        def truth_edges(tp):
            A = np.abs(np.array(graphs["daily"][tp])); np.fill_diagonal(A, 0); return (A >= 0.04).astype(int)
        off = ~np.eye(4, dtype=bool)
        scores = {"CrisisDynNet": [], "Granger (pairwise F)": [], "PCMCI (ParCorr)": [], "Lasso-VAR": [], "Lagged cross-correlation": []}
        ys = []
        for i in range(n):
            tpe = truth[i]; yt = truth_edges(tpe)[off]
            ys.append(yt)
            Ai = sum(mdl.R_[i, k] * np.abs(nets[k][np.ix_(obs, obs)]) for k in range(K_eval))
            scores["CrisisDynNet"].append(Ai[off])
            z = P["Z"][i][:, obs]
            scores["Granger (pairwise F)"].append(BL.granger_person(z)[off])
            scores["Lasso-VAR"].append(BL.lasso_var_person(z)[off])
            scores["Lagged cross-correlation"].append(BL.xcorr_person(z)[off])
            try:
                scores["PCMCI (ParCorr)"].append(BL.pcmci_person(z)[off])
            except Exception:
                scores["PCMCI (ParCorr)"].append(np.zeros(off.sum()))
        ys = np.array(ys)
        for m, sc in scores.items():
            sc = np.nan_to_num(np.array(sc))
            pooled = roc_auc_score(ys.ravel(), sc.ravel())
            per = [roc_auc_score(ys[i], sc[i]) for i in range(n) if 0 < ys[i].sum() < ys.shape[1]]
            # F1 at top-k edges (k = true number per person)
            f1 = []
            for i in range(n):
                k = ys[i].sum()
                if k == 0:
                    continue
                pred = np.zeros_like(ys[i]); pred[np.argsort(-sc[i])[:k]] = 1
                tp = (pred & ys[i]).sum(); f1.append(2 * tp / (pred.sum() + k))
            edge_rows.append(dict(method=m, auroc_pooled=pooled, auroc_person_mean=np.mean(per), f1_topk=np.mean(f1)))
        edges = pd.DataFrame(edge_rows); edges.to_csv(RES / "cohort_causal_recovery.csv", index=False)
        checks.append(dict(check="causal edge AUROC (CrisisDynNet, pooled)", value=edges.auroc_pooled.iloc[0]))
    else:
        # 没有真值网络：因果路径恢复的设计检验不适用（不产出任何"准确率"）
        pd.DataFrame(columns=["method", "auroc_pooled", "auroc_person_mean", "f1_topk"]).to_csv(
            RES / "cohort_causal_recovery.csv", index=False)
        checks.append(dict(check="causal edge AUROC (CrisisDynNet, pooled)", value=np.nan))
        print("causal recovery skipped (no truth cache)", flush=True)
    # interpretable class networks with bootstrap CIs (all 7 nodes)
    boot = mdl.bootstrap_edges(n_boot=C.N_BOOT // 2, seed=SEED)
    net_rows = []
    for k in range(K_eval):
        for a in range(p):
            for b in range(p):
                if a == b:
                    continue
                v = boot[:, k, a, b]
                net_rows.append(dict(subtype=cls_name[k], target=CD.NODE[P["vars"][a]], source=CD.NODE[P["vars"][b]],
                                     coef=nets[k][a, b], lo=np.quantile(v, .025), hi=np.quantile(v, .975)))
    nets_df = pd.DataFrame(net_rows); nets_df["sig"] = (nets_df.lo > 0) | (nets_df.hi < 0)
    nets_df.to_csv(RES / "cohort_class_networks.csv", index=False)
    print("causal done", round(time.time() - t0), "s", flush=True)

    # ================================================================ 3. 14-day crisis prediction
    Zs, Mk, groups, y, valid, det = CD.make_sequences(P, D)
    roll = np.stack([CD.rolling_indicators(P["Z"][i]) for i in range(n)])
    rmask = ~np.isnan(roll); roll = np.nan_to_num(roll)
    q = D["wave_questionnaires"]; q0 = q[q.wave == "T0"].set_index("pid")
    q0 = q0.assign(phq=q0[[f"phq9_{i}" for i in range(1, 10)]].sum(1), gad=q0[[f"gad7_{i}" for i in range(1, 8)]].sum(1))
    S_scale = q0.reindex(P["pids"])[["phq", "gad", "pss14_total", "isi_total"]]
    S_scale = S_scale.fillna(S_scale.median()).values
    L = D["lab_epochs"]; L0 = L[L.wave == "T0"]
    lm = L0.groupby(["pid", "phase"])[["fm_theta_log", "hr_bpm", "scl_us"]].mean().unstack("phase")
    labf = pd.DataFrame({"hr_react": lm["hr_bpm"]["stress"] - lm["hr_bpm"]["rest"], "scl_react": np.log(lm["scl_us"]["stress"] / lm["scl_us"]["rest"]),
                         "theta_react": lm["fm_theta_log"]["stress"] - lm["fm_theta_log"]["rest"],
                         "hr_recov": lm["hr_bpm"]["recovery"] - lm["hr_bpm"]["rest"]}).reindex(P["pids"])
    labf = labf.fillna(labf.median()).values
    stand = lambda a: StandardScaler().fit_transform(a)
    static_full = np.c_[mdl.R_, stand(S_scale), stand(labf)]
    ema = np.concatenate([Zs[..., 0:3], Mk[..., 0:3]], -1)
    wear = np.concatenate([Zs[..., 3:7], Mk[..., 3:7]], -1)
    dyn = np.concatenate([roll, rmask], -1)
    def smooth(a):
        return pd.DataFrame(a.reshape(-1, a.shape[-1])).groupby(np.repeat(np.arange(a.shape[0]), a.shape[1])).transform(
            lambda s: s.rolling(14, min_periods=1).mean()).values.reshape(a.shape)
    variants = {
        "CrisisDynNet (full)": ([ema, wear, dyn], static_full),
        "- EMA": ([wear, dyn], static_full),
        "- wearable physiology": ([ema, dyn], static_full),
        "- early-warning dynamics": ([ema, wear], static_full),
        "- subtype posteriors": ([ema, wear, dyn], np.c_[stand(S_scale), stand(labf)]),
        "- lab reactivity (EEG/ECG/EDA)": ([ema, wear, dyn], np.c_[mdl.R_, stand(S_scale)]),
        "- temporal module (14-day means)": ([smooth(ema), smooth(wear), smooth(dyn)], static_full),
    }
    gkf = GroupKFold(5)
    folds = list(gkf.split(np.arange(n), groups=np.arange(n)))
    preds = {}
    for vn, (Xg, St) in variants.items():
        pr = np.zeros((n, T))
        for f, (tr, te) in enumerate(folds):
            mdl_f = CD.train_fusion(Xg, St, y, valid, tr, epochs=80, seed=SEED + f)
            pr[te] = CD.predict_fusion(mdl_f, Xg, St, te)
        preds[vn] = pr
        print("fusion", vn, round(roc_auc_score(y[valid], pr[valid]), 3), flush=True)
    roll_sum = BL.rolling_summary(P["Z"])
    base_pred = {"Screening scales (logistic)": np.zeros((n, T)), "Gradient boosting on 14-day summaries": np.zeros((n, T)),
                 "LSTM classifier": np.zeros((n, T)), "Hidden Markov model states": np.zeros((n, T))}
    for f, (tr, te) in enumerate(folds):
        base_pred["Screening scales (logistic)"][te] = BL.fit_predict_static(S_scale, y, valid, tr, te)
        base_pred["Gradient boosting on 14-day summaries"][te] = BL.fit_predict_gbm(roll_sum, y, valid, tr, te)
        base_pred["LSTM classifier"][te] = BL.fit_predict_lstm(P["Z"], P["mask"].astype(float), y, valid, tr, te, seed=SEED + f)
        base_pred["Hidden Markov model states"][te] = BL.fit_predict_hmm(P["Z"][:, :, [0, 1, 3, 6]], y, valid, tr, te, seed=SEED)
    # CrisisDynNet-GBM: same learner as the GBM baseline, plus the framework's features
    # (early-warning dynamics, subtype posteriors, lab reactivity, screening scales)
    stat_rep = np.repeat(static_full[:, None, :], T, 1)
    Xcd = np.concatenate([roll_sum, roll, stat_rep], -1)
    preds["CrisisDynNet-GBM (framework features)"] = np.zeros((n, T))
    for f, (tr, te) in enumerate(folds):
        preds["CrisisDynNet-GBM (framework features)"][te] = BL.fit_predict_gbm(Xcd, y, valid, tr, te)
    for nm_, drop in [("CrisisDynNet-GBM - subtype posteriors", slice(0, K_eval)), ("CrisisDynNet-GBM - lab reactivity", slice(K_eval + 4, K_eval + 8))]:
        st_ = static_full.copy(); st_[:, drop] = 0
        Xa = np.concatenate([roll_sum, roll, np.repeat(st_[:, None, :], T, 1)], -1)
        preds[nm_] = np.zeros((n, T))
        for f, (tr, te) in enumerate(folds):
            preds[nm_][te] = BL.fit_predict_gbm(Xa, y, valid, tr, te)
    Xa = np.concatenate([roll_sum, np.repeat(static_full[:, None, :], T, 1)], -1)
    preds["CrisisDynNet-GBM - early-warning dynamics"] = np.zeros((n, T))
    for f, (tr, te) in enumerate(folds):
        preds["CrisisDynNet-GBM - early-warning dynamics"][te] = BL.fit_predict_gbm(Xa, y, valid, tr, te)
    print("baselines done", round(time.time() - t0), "s", flush=True)
    allp = {**preds, **base_pred}
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, n, n) for _ in range(300)]
    REF = "CrisisDynNet-GBM (framework features)"
    ref = preds[REF]
    def auc_idx(pr, idx):
        v = valid[idx]; yy = y[idx][v]
        return roc_auc_score(yy, pr[idx][v]) if 0 < yy.sum() < len(yy) else np.nan
    prow = []
    for m, pr in allp.items():
        a = roc_auc_score(y[valid], pr[valid]); ap = average_precision_score(y[valid], pr[valid])
        bs = np.array([auc_idx(pr, b) for b in boots]); br = np.array([auc_idx(ref, b) for b in boots])
        d = br - bs
        prow.append(dict(method=m, auroc=a, auroc_lo=np.nanquantile(bs, .025), auroc_hi=np.nanquantile(bs, .975), auprc=ap,
                         delta_vs_full=a - roc_auc_score(y[valid], ref[valid]),
                         p_full_better=float(np.mean(d <= 0)) if m != REF else np.nan))
    pred_tab = pd.DataFrame(prow)
    # pre-registered families (实验设计方案.md §2 H3): primary = vs screening scales (single one-sided test,
    # no adjustment); secondary = other ML/DL baselines (Holm); ablations = their own Holm family
    fam = {"Screening scales (logistic)": "primary"}
    for m in ["Gradient boosting on 14-day summaries", "LSTM classifier", "Hidden Markov model states", "CrisisDynNet (full)"]:
        fam[m] = "secondary"
    pred_tab["family"] = pred_tab.method.map(fam).fillna("ablation")
    pred_tab.loc[pred_tab.method == REF, "family"] = "reference"
    pred_tab["p_adj"] = np.nan
    for f_ in ["primary", "secondary", "ablation"]:
        sub_ = pred_tab[(pred_tab.family == f_) & pred_tab.p_full_better.notna()].sort_values("p_full_better")
        m_ = len(sub_)
        for r, ix in enumerate(sub_.index):
            pred_tab.loc[ix, "p_adj"] = min(1.0, max(min(1, (m_ - j) * sub_.p_full_better.iloc[j]) for j in range(r + 1)))
    pred_tab["p_holm"] = pred_tab["p_adj"]                         # kept for downstream readers
    pred_tab.base_rate = float(y[valid].mean())
    pred_tab.to_csv(RES / "cohort_prediction.csv", index=False)
    np.save(C.CACHE / "cohort_pred_full.npy", ref)

    # lead time vs standard care (scheduled waves / self-referral)
    thr = np.quantile(ref[valid & (y == 0)], 0.90)            # 90 % specificity on non-event days
    ce = D["crisis_events"]; cal0 = pd.Timestamp(D["academic_calendar"].date.min())
    lead = []
    for q_, d_, on_, route in zip(ce.pid, pd.to_datetime(ce.detected_date), pd.to_datetime(ce.onset_date_est), ce.detection_route):
        i = P["pids"].index(q_); td = (d_ - cal0).days; to = (on_ - cal0).days
        w = np.where(ref[i, 28:td] >= thr)[0] + 28
        first = int(w[0]) if len(w) else None
        lead.append(dict(pid=q_, subtype=sub[i], route=route, care_delay_days=td - to,
                         alert_day=first, lead_vs_care=(td - first) if first is not None else 0,
                         alert_before_care=int(first is not None)))
    lead = pd.DataFrame(lead); lead.to_csv(RES / "cohort_lead_time.csv", index=False)
    summary["alert_threshold_specificity"] = 0.90
    summary["alert_sensitivity_before_care"] = float(lead.alert_before_care.mean()) if len(lead) else np.nan
    summary["median_lead_days"] = float(lead.loc[lead.alert_before_care == 1, "lead_vs_care"].median()) if len(lead) else np.nan
    print("prediction done", round(time.time() - t0), "s", flush=True)

    # interpretability: modality gates and IG attributions by subtype (full model trained on all)
    Xg, St = variants["CrisisDynNet (full)"]
    full = CD.train_fusion(Xg, St, y, valid, np.arange(n), epochs=80, seed=SEED)
    _, gates = CD.predict_fusion(full, Xg, St, np.arange(n), gates=True)
    gate_rows = []
    for nm_ in sorted(set(sub)):
        ii = np.where(sub == nm_)[0]
        g = gates[ii][valid[ii]].mean(0)
        att = CD.integrated_gradients(full, Xg, St, ii)
        gate_rows.append(dict(subtype=nm_, n=len(ii), gate_ema=g[0], gate_wear=g[1], gate_dyn=g[2],
                              ig_ema=att[0], ig_wear=att[1], ig_dyn=att[2], ig_static=att[3]))
    pd.DataFrame(gate_rows).to_csv(RES / "cohort_explanations.csv", index=False)

    # ================================================================ 4. mechanism: profiles, lab pathways, tipping points
    part = D["participants"].set_index("pid").reindex(P["pids"])
    prof = F.assign(subtype=sub, female=(part.sex == "F").values.astype(int), final_year=part.final_year.values,
                    crisis=[int(q_ in set(ce.pid)) for q_ in P["pids"]])
    prof.groupby("subtype").agg(["mean", "std"]).round(3).to_csv(RES / "cohort_subtype_profiles.csv")
    prof_tests = []
    for c in ["level", "slope", "hrv_slope", "event_resp", "instab", "inertia", "arousal_dissoc"]:
        groups_ = [prof.loc[prof.subtype == s_, c].dropna() for s_ in sorted(set(sub))]
        H, pval = stats.kruskal(*groups_)
        eps2 = (H - len(groups_) + 1) / (len(prof) - len(groups_))
        prof_tests.append(dict(feature=c, H=H, p=pval, epsilon2=eps2))
    pd.DataFrame(prof_tests).to_csv(RES / "cohort_subtype_tests.csv", index=False)

    # lab VAR edges per session, compared across subtypes
    LV = ["fm_theta_log", "post_alpha_log", "faa", "hr_bpm", "scl_us"]
    lab_rows = []
    for (q_, w_), g in L.groupby(["pid", "wave"]):
        z = g.sort_values("epoch")[LV].copy(); z["scl_us"] = np.log(z["scl_us"])
        z = (z - z.mean()) / z.std()
        zz = z.values; ok = ~np.isnan(zz[1:]).any(1) & ~np.isnan(zz[:-1]).any(1)
        if ok.sum() < 40:
            continue
        X = np.c_[zz[:-1][ok], (g.sort_values("epoch").phase.values[1:][ok] == "stress").astype(float)]
        B = np.linalg.lstsq(np.c_[np.ones(ok.sum()), X], zz[1:][ok], rcond=None)[0][1:6].T
        lab_rows.append(dict(pid=q_, wave=w_, theta_to_hr=B[3, 0], hr_to_alpha=B[1, 3], scl_to_hr=B[3, 4], hr_to_scl=B[4, 3],
                             theta_to_scl=B[4, 0]))
    labE = pd.DataFrame(lab_rows).merge(pd.DataFrame({"pid": P["pids"], "subtype": sub}), on="pid")
    qw = D["wave_questionnaires"].assign(phq=lambda d: d[[f"phq9_{i}" for i in range(1, 10)]].sum(axis=1))
    labE = labE.merge(qw[["pid", "wave", "phq"]], on=["pid", "wave"], how="left")
    labE["phq"] = labE.phq.fillna(labE.phq.median())                 # control current severity (V13)
    cats = ["stable"] + sorted(set(labE.subtype) - {"stable"})
    labE["subtype"] = pd.Categorical(labE.subtype, categories=cats)       # reference level = stable
    labE.to_csv(RES / "cohort_lab_edges.csv", index=False)
    lt = []
    for e_ in ["theta_to_hr", "hr_to_alpha", "scl_to_hr", "theta_to_scl"]:
        m1 = smf.mixedlm(f"{e_} ~ subtype + phq", labE, groups=labE.pid).fit(reml=False)
        m0 = smf.mixedlm(f"{e_} ~ phq", labE, groups=labE.pid).fit(reml=False)
        lr = 2 * (m1.llf - m0.llf); df = len(m1.fe_params) - len(m0.fe_params)
        coefs = {k.split("[T.")[-1].rstrip("]"): (v, m1.pvalues[k]) for k, v in m1.fe_params.items() if "[T." in k}
        lt.append(dict(edge=e_, LR=lr, df=df, p=stats.chi2.sf(lr, df), **{f"b_{k}": v[0] for k, v in coefs.items()},
                       **{f"p_{k}": v[1] for k, v in coefs.items()}))
    pd.DataFrame(lt).to_csv(RES / "cohort_lab_pathways.csv", index=False)

    # tipping points: early-warning trends before estimated onset vs matched windows in non-crisis persons
    tp_rows = []
    crisis_set = set(ce.pid)
    non = [i for i, q_ in enumerate(P["pids"]) if q_ not in crisis_set]
    rng2 = np.random.default_rng(SEED)
    for q_, on_ in zip(ce.pid, pd.to_datetime(ce.onset_date_est)):
        i = P["pids"].index(q_); to = (on_ - cal0).days
        if to < 35:
            continue
        r = CD.rolling_indicators(P["Z"][i])[to - 28:to]
        j = rng2.choice(non); rc = CD.rolling_indicators(P["Z"][j])[to - 28:to]
        tp_rows.append(dict(pid=q_, subtype=sub[i], tau_ar1=CD.kendall_trend(r[:, 0]), tau_var=CD.kendall_trend(r[:, 1]),
                            tau_coupling=CD.kendall_trend(r[:, 2]), ctrl_tau_ar1=CD.kendall_trend(rc[:, 0]),
                            ctrl_tau_var=CD.kendall_trend(rc[:, 1]), ctrl_tau_coupling=CD.kendall_trend(rc[:, 2])))
    tpdf = pd.DataFrame(tp_rows); tpdf.to_csv(RES / "cohort_tipping_points.csv", index=False)
    tp_test = []
    for ind in ["ar1", "var", "coupling"]:
        a = tpdf[f"tau_{ind}"]; b = tpdf[f"ctrl_tau_{ind}"]; ok = a.notna() & b.notna()
        if ok.sum() > 5:
            w = stats.wilcoxon(a[ok], b[ok])
            tp_test.append(dict(indicator=ind, n=int(ok.sum()), median_crisis=a[ok].median(), median_control=b[ok].median(), p=w.pvalue))
    pd.DataFrame(tp_test).to_csv(RES / "cohort_tipping_tests.csv", index=False)

    # ecological validation: risk during examination and recruitment windows
    per = P["cal"].period.values
    eco = []
    for nm_, msk in [("examination", per == "examination"), ("teaching", per == "teaching"), ("winter_break", per == "winter_break")]:
        vv = valid[:, msk]
        if vv.sum() == 0:
            continue
        yy = y[:, msk][vv]; pp = ref[:, msk][vv]
        eco.append(dict(period=nm_, n_days=int(vv.sum()), event_rate=float(yy.mean()),
                        auroc=roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan, mean_risk=float(pp.mean())))
    fy = part.final_year.values.astype(bool); ja, jb = C.COHORT_DESIGN["job_window"]
    for nm_, rows_ in [("recruitment season, final-year", fy), ("recruitment season, other years", ~fy)]:
        vv = valid[rows_][:, ja:jb]; yy = y[rows_][:, ja:jb][vv]; pp = ref[rows_][:, ja:jb][vv]
        eco.append(dict(period=nm_, n_days=int(vv.sum()), event_rate=float(yy.mean()) if len(yy) else np.nan,
                        auroc=roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan, mean_risk=float(pp.mean()) if len(pp) else np.nan))
    pd.DataFrame(eco).to_csv(RES / "cohort_ecological.csv", index=False)

    if C.has_truth():
        pd.DataFrame(checks).to_csv(RES / "cohort_design_checks.csv", index=False)
    else:
        # 真值缓存缺失：这些检验不适用，明确写出来，避免被误当成"未通过"或"通过"
        pd.DataFrame([dict(check="与真值有关的设计检验不适用（真值缓存缺失）", value=np.nan,
                           note="ARI/命名准确率/因果路径恢复需要真值缓存；此处不得引用任何设计检验数值")]).to_csv(
            RES / "cohort_design_checks.csv", index=False)
    summary.update(dict(n=n, n_crisis=int(len(ce)), base_rate=float(y[valid].mean()), runtime_s=round(time.time() - t0)))
    (RES / "cohort_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(pred_tab.round(3).to_string()); print(subtab.round(3).to_string())
    if truth is not None:
        print(edges.round(3).to_string())


if __name__ == "__main__":
    main()
