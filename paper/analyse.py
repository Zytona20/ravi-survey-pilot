"""Every number of the pilot study (Section 11 of the survey) from the measurements in results/.

Usage: python paper/analyse.py <folder that contains results/> <output.json>
"""
import itertools, json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, binomtest

R = sys.argv[1]  # extracted results folder
b = pd.read_csv(f"{R}/results/bench.csv"); b = b[b.k == 10]
g = pd.read_csv(f"{R}/results/geometry.csv")
q = pd.read_csv(f"{R}/results/quality.csv")
cards = pd.read_csv(f"{R}/results/survey_geometry_cards.csv")
p2c = pd.read_csv(f"{R}/results/survey_p2_curves.csv")
p2 = pd.read_csv(f"{R}/results/survey_p2.csv")
SHORT = {"resnet50.a1_in1k": "ResNet-50", "convnext_base.fb_in22k_ft_in1k": "ConvNeXt-B",
         "vit_base_patch16_224.augreg_in21k_ft_in1k": "ViT-B/16", "vit_base_patch16_clip_224.openai": "CLIP ViT-B/16",
         "vit_base_patch14_dinov2.lvd142m": "DINOv2 ViT-B/14", "vit_base_patch16_224.mae": "MAE ViT-B/16"}
ORDER = list(SHORT)
out = {}

def interp_curve(r, n, t):
    """distance evaluations to reach recall t, log-linear between bracketing settings; (value, is_upper_bound)"""
    o = np.argsort(n); r, n = np.asarray(r)[o], np.asarray(n)[o]
    if r.max() < t: return np.nan, False
    if r[0] >= t: return float(n[0]), True
    j = int(np.argmax(r >= t)); f = (t - r[j-1]) / (r[j] - r[j-1])
    return float(np.exp(np.log(n[j-1]) + f * (np.log(n[j]) - np.log(n[j-1])))), False

def cost(t, kind, basis="recall"):
    rows = []
    for (sid, seed), s in b[b["index"] == kind].groupby(["space_id", "seed"]):
        v, ub = interp_curve(s[basis].values, s["ndis"].values, t)
        rows.append({"space_id": sid, "seed": seed, "ndis": v, "ub": ub})
    d = pd.DataFrame(rows)
    return d.groupby("space_id").agg(ndis=("ndis", "mean"), sd=("ndis", "std"), ub=("ub", "max")).reset_index().merge(g, on="space_id")

def perm(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); rx, ry = rankdata(x), rankdata(y)
    r = np.corrcoef(rx, ry)[0, 1]
    null = np.array([np.corrcoef(rx, np.array(p))[0, 1] for p in itertools.permutations(ry)])
    return float(r), float(np.mean(np.abs(null) >= abs(r) - 1e-12))

def boot(d):
    rng = np.random.default_rng(0); ms = d.model.unique(); vals = []
    for _ in range(5000):
        dd = pd.concat([d[d.model == m] for m in rng.choice(ms, len(ms))])
        if dd.qlid_median.nunique() > 3: vals.append(spearmanr(dd.qlid_median, dd.ndis).correlation)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

# ---------- costs ----------
C = {(k, t): cost(t, k) for k in ("hnsw", "ivf") for t in (0.9, 0.95)}
out["native_cost"] = {}
for (k, t), d in C.items():
    nat = d[d.reduction == "native"].set_index("model")
    p256 = d[(d.reduction == "pca") & (d.dim == 256)].set_index("model")
    out["native_cost"][f"{k}@{t}"] = {SHORT[m]: [round(nat.loc[m, "ndis"]), round(p256.loc[m, "ndis"])] for m in ORDER}
    out["native_cost"][f"{k}@{t}_range"] = [round(nat.ndis.max() / nat.ndis.min(), 2), round(p256.ndis.max() / p256.ndis.min(), 2)]
    out["native_cost"][f"{k}@{t}_max_seed_sd"] = [round(float(nat.sd.max()), 1), round(float(p256.sd.max()), 1)]
    out["native_cost"][f"{k}@{t}_upper_bounds"] = int(nat.ub.sum() + p256.ub.sum())

# ---------- P1 ----------
P1 = {}
for (k, t), d in C.items():
    for space, dd in (("pca256", d[(d.reduction == "pca") & (d.dim == 256)]), ("native", d[d.reduction == "native"])):
        r, p = perm(dd.qlid_median, dd.ndis); P1[f"{k}@{t}:{space}:across6"] = [round(r, 3), round(p, 3)]
    d256 = d[d.dim == 256]
    P1[f"{k}@{t}:pooled24"] = [round(spearmanr(d256.qlid_median, d256.ndis).correlation, 3)] + [round(v, 2) for v in boot(d256)]
    P1[f"{k}@{t}:within"] = {SHORT[m]: round(spearmanr(x.qlid_median, x.ndis).correlation, 2) for m, x in d256.groupby("model")}
    P1[f"{k}@{t}:pooled_upper_bounds"] = int(d256.ub.sum())
out["P1"] = P1

# ---------- P2 ----------
P2 = {}
for t in (0.9, 0.95, 0.99):
    v = {}
    for qt, x in p2c.groupby("query_type"):
        v[qt] = interp_curve(x["recall@10"].values, x["ndis"].values, t)[0]
    P2[f"ndis@{t}"] = {k: round(val) for k, val in v.items()}
    P2[f"ratio@{t}"] = round(v["text"] / v["image"], 2)
P2["at_ef10"] = {qt: [round(float(x[x.efSearch == 10]["recall@10"].iloc[0]), 3), round(float(x[x.efSearch == 10]["ndis"].iloc[0]))] for qt, x in p2c.groupby("query_type")}
P2["max_recall"] = {qt: round(float(x["recall@10"].max()), 4) for qt, x in p2c.groupby("query_type")}
P2["table"] = p2[["query_type", "n_queries", "qshift_nn", "qlid_median", "rc_q10", "label_p@10_exact", "ef_at_target"]].round(3).to_dict("records")
out["P2"] = P2

# ---------- P3 ----------
P3 = {"ceiling": {}}
for m in ORDER:
    P3["ceiling"][SHORT[m]] = {int(dd): round(float(q[(q.model == m) & (q.reduction == "pca") & (q.dim == dd)]["reduction_recall@10"].iloc[0]), 3) for dd in (64, 128, 256)}
e2e = {t: cost(t, "hnsw", "e2e_recall") for t in (0.8, 0.9)}
P3["e2e_feasible"] = {f"{t}": {SHORT[m]: [int(x) for x in sorted(d[(d.model == m) & d.ndis.notna()].dim)] for m in ORDER} for t, d in e2e.items()}
d8 = e2e[0.8]
for m in ("vit_base_patch14_dinov2.lvd142m", "convnext_base.fb_in22k_ft_in1k"):
    x = d8[d8.model == m].set_index("reduction")
    w_nat = x.loc["native", "ndis"] * x.loc["native", "dim"]
    p = x[(x.index == "pca")]
    p = p[p.dim == 256]
    w_256 = float(p.ndis.iloc[0] * 256)
    P3[f"work@0.8:{SHORT[m]}"] = {"native_ndis": round(x.loc["native", "ndis"]), "pca256_ndis": round(float(p.ndis.iloc[0])),
                                  "native_over_256": round(w_nat / w_256, 2)}
nm = cards[~cards.model.str.contains("mae")]
P3["rho_lid_ceiling_without_mae"] = round(spearmanr(nm.qlid_median, nm.rr10_pca256).correlation, 3)
P3["rho_lid_ceiling_all"] = round(spearmanr(cards.qlid_median, cards.rr10_pca256).correlation, 3)
out["P3"] = P3

# ---------- P4 / P5 ----------
P45 = {}
d = C[("hnsw", 0.9)]
d256 = d[d.dim == 256].set_index(["model", "reduction"])
qq = q.set_index("space_id")
agree = 0; rows = []
for m in ORDER:
    base = d256.loc[(m, "head_rank")]
    for red in ("head_hub", "head_koleo"):
        y = d256.loc[(m, red)]
        dq, dn = y.qlid_median - base.qlid_median, y.ndis - base.ndis
        agree += int(np.sign(dq) == np.sign(dn))
        lp = qq.loc[y.space_id, "label_p@10"] - qq.loc[base.space_id, "label_p@10"]
        rr = qq.loc[y.space_id, "reduction_recall@10"] - qq.loc[base.space_id, "reduction_recall@10"]
        rows.append({"backbone": SHORT[m], "head": red, "d_hub_skew": round(y.hub_skew - base.hub_skew, 2),
                     "d_qlid": round(dq, 2), "d_rc": round(y.rc_q10 - base.rc_q10, 3),
                     "d_ndis_pct": round(100 * dn / base.ndis, 1), "d_label_p": round(lp, 3), "d_ceiling": round(rr, 3)})
P45["rows"] = rows
P45["sign_agreement"] = [agree, 12, round(binomtest(agree, 12, 0.5).pvalue, 4)]
df = pd.DataFrame(rows)
for red in ("head_hub", "head_koleo"):
    x = df[df["head"] == red]
    P45[red] = {c: [int((x[c] > 0).sum()), int((x[c] < 0).sum()), round(float(x[c].mean()), 3), round(float(x[c].min()), 3), round(float(x[c].max()), 3)]
                for c in ("d_hub_skew", "d_qlid", "d_rc", "d_ndis_pct", "d_label_p", "d_ceiling")}
# PCA vs head_rank neighbour preservation at 256
P45["pca_vs_rank_ceiling"] = {SHORT[m]: [round(float(qq.loc[d256.loc[(m, 'pca')].space_id, 'reduction_recall@10']), 3),
                                          round(float(qq.loc[d256.loc[(m, 'head_rank')].space_id, 'reduction_recall@10']), 3)] for m in ORDER}
out["P45"] = P45

# ---------- figure data ----------
fig = {"p1": [], "p2": {}, "p3": {}}
for (m, red), y in d256.iterrows():
    fig["p1"].append({"backbone": SHORT[m], "reduction": red, "qlid": round(float(y.qlid_median), 2), "ndis": round(float(y.ndis), 1)})
for qt, x in p2c.groupby("query_type"):
    fig["p2"][qt] = [[round(float(a), 1), round(float(r), 4)] for a, r in zip(x["ndis"], x["recall@10"])]
for m in ORDER:
    fig["p3"][SHORT[m]] = [[dd, P3["ceiling"][SHORT[m]][dd]] for dd in (64, 128, 256)]
out["fig"] = fig

# ---------- card ----------
card = {}
for m in ORDER:
    r = cards[cards.model == m].iloc[0]
    nat = C[("hnsw", 0.9)]; ivf = C[("ivf", 0.9)]
    card[SHORT[m]] = {
        "dim": int(r.native_dim), "twonn": round(r.twonn_id, 1), "qlid": round(r.qlid_median, 1), "rc": round(r.rc_q10, 2),
        "hub": round(r.hub_skew, 2), "antihub": round(r.hub_antihub, 3), "cos": round(r.aniso_meancos, 2), "top1": round(r.aniso_top1, 2),
        "kmerr": round(r.clus_qerr, 2), "shift": round(r.qshift_nn, 2),
        "pca256": round(r.rr10_pca256, 2), "pca128": round(r.rr10_pca128, 2), "pca64": round(r.rr10_pca64, 2),
        "sq8": round(r.rr10_sq8, 2), "sq4": round(r.rr10_sq4, 2), "bin1": round(r.rr10_bin1, 2),
        "hnsw": round(float(nat[(nat.model == m) & (nat.reduction == "native")].ndis.iloc[0])),
        "ivf": round(float(ivf[(ivf.model == m) & (ivf.reduction == "native")].ndis.iloc[0])),
        "labelp": round(r.label_p10, 2)}
out["card"] = card
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "fig"}, indent=1))
