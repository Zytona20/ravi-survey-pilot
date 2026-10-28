"""The proposed method: a geometry-aware compact projection head.

A single linear map  z = normalise(W (x - mu))  from the native space R^D to R^d (d <= 256),
trained on SAVED vectors only (no images, no backbone forward pass), initialised from PCA.

Loss = neighbour ranking preservation  +  one optional geometry regulariser:

  rank    listwise KL between teacher (native) and student (compact) similarity distributions
          inside the batch, plus a multi-positive InfoNCE on true teacher nearest neighbours
  hub     soft in-degree balancing: variance of the soft k-occurrence count  (hubness)
  koleo   Kozachenko-Leonenko entropy term (uniform spreading; Sablayrolles et al., ICLR 2019)
  lid     mean in-batch MLE local intrinsic dimensionality
  iso     distance of the output covariance from a scaled identity (anisotropy)

Because every regulariser has its own weight, the head is also an INTERVENTION tool:
sweeping one weight changes one geometry property while backbone, data and dimension stay
fixed, which turns the correlational study into a controlled experiment.
"""
from __future__ import annotations

import math

import numpy as np

from .knn import exact_knn
from .metrics import recall_at_k
from .utils import log, set_seed


def pca_fit(x: np.ndarray, d: int):
    """Top-d principal directions. Returns (W [d, D], mu [D], eigenvalues [D] descending)."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / max(len(x) - 1, 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return v[:, order[:d]].T.astype(np.float32), mu.astype(np.float32), w[order]


def hubness_skew(idx: np.ndarray, n: int) -> float:
    occ = np.bincount(idx.ravel(), minlength=n).astype(np.float64)
    s = occ.std()
    return float(((occ - occ.mean()) ** 3).mean() / s**3) if s > 0 else 0.0


def train_head(x: np.ndarray, d_out: int, hp: dict, device: str, seed: int = 0):
    """x: [n, D] L2-normalised native vectors. Returns (W, mu, history)."""
    import torch
    import torch.nn.functional as F

    set_seed(seed)
    n, D = x.shape
    B, g = int(hp["batch"]), max(int(hp["group"]), 2)
    k_pos = min(int(hp["k_pos"]), n - 1)
    n_val = min(int(hp["n_val"]), n // 5)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_ids, tr_ids = perm[:n_val], perm[n_val:]
    xtr, xval = x[tr_ids], x[val_ids]
    ntr = len(xtr)

    # teacher neighbours inside the training sample (native space)
    nn_idx, _ = exact_knn(xtr, xtr, k_pos, device=device, exclude_self=True)
    val_gt, _ = exact_knn(xtr, xval, 10, device=device)

    if hp.get("init", "pca") == "pca":
        W0, mu, _ = pca_fit(xtr, d_out)
    else:
        W0 = np.linalg.qr(rng.standard_normal((D, d_out)))[0].T.astype(np.float32)
        mu = xtr.mean(0).astype(np.float32)

    X = torch.from_numpy(np.ascontiguousarray(xtr)).to(device)
    NN = torch.from_numpy(nn_idx).to(device)
    MU = torch.from_numpy(mu).to(device)
    W = torch.nn.Parameter(torch.from_numpy(W0.copy()).to(device))
    opt = torch.optim.AdamW([W], lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    steps = int(hp["steps"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(hp["lr"]), total_steps=max(steps, 2), pct_start=0.05)
    n_anchor = max(B // g, 2)
    gen = torch.Generator(device=device).manual_seed(seed)
    grp = torch.arange(n_anchor, device=device).repeat_interleave(g)
    m = n_anchor * g
    eye = torch.eye(m, dtype=torch.bool, device=device)
    pos_mask = (grp[:, None] == grp[None, :]) & ~eye
    tau_t, tau_s, tau_h = float(hp["tau_t"]), float(hp["tau_s"]), float(hp["tau_h"])
    lam = {k: float(hp[f"lam_{k}"]) for k in ("hub", "koleo", "lid", "iso")}
    k_lid = min(int(hp["k_lid"]), m - 2)

    def evaluate():
        with torch.no_grad():
            Wn = W.detach().cpu().numpy()
        zb = _apply(xtr, Wn, mu)
        zq = _apply(xval, Wn, mu)
        found, _ = exact_knn(zb, zq, 10, device=device)
        sample = zb[: min(len(zb), 20000)]
        occ_idx, _ = exact_knn(sample, sample, 10, device=device, exclude_self=True)
        return recall_at_k(found, val_gt, 10), hubness_skew(occ_idx, len(sample))

    history = []
    r0, h0 = evaluate()
    history.append({"step": 0, "val_recall10": r0, "hub_skew": h0})
    for step in range(1, steps + 1):
        a = torch.randint(ntr, (n_anchor,), device=device, generator=gen)
        nb = NN[a][:, torch.randint(k_pos, (g - 1,), device=device, generator=gen)]
        ids = torch.cat([a[:, None], nb], 1).reshape(-1)  # [anchor, its neighbours] per group
        xb = X[ids]
        z = F.normalize((xb - MU) @ W.T, dim=1)
        st = (xb @ xb.T).masked_fill(eye, -1e4)
        ss = (z @ z.T).masked_fill(eye, -1e4)
        logp_s = F.log_softmax(ss / tau_s, dim=1)
        p_t = F.softmax(st / tau_t, dim=1)
        l_kl = (p_t * (torch.log(p_t.clamp_min(1e-12)) - logp_s)).sum(1).mean()
        l_nce = -(logp_s * pos_mask).sum(1).div(pos_mask.sum(1).clamp_min(1)).mean()
        loss = float(hp["w_kl"]) * l_kl + float(hp["w_nce"]) * l_nce
        parts = {"kl": l_kl.item(), "nce": l_nce.item()}
        if lam["hub"]:
            indeg = F.softmax(ss / tau_h, dim=1).sum(0)  # soft k-occurrence, mean 1
            l_hub = ((indeg - 1.0) ** 2).mean()
            loss = loss + lam["hub"] * l_hub
            parts["hub"] = l_hub.item()
        if lam["koleo"]:
            dmin = torch.sqrt((2.0 - 2.0 * ss.max(1).values).clamp_min(1e-12))
            l_ko = -torch.log(dmin + 1e-8).mean()
            loss = loss + lam["koleo"] * l_ko
            parts["koleo"] = l_ko.item()
        if lam["lid"]:
            top = torch.topk(ss, k_lid, dim=1).values
            r = torch.sqrt((2.0 - 2.0 * top).clamp_min(1e-12))  # ascending distances
            lid = -1.0 / (torch.log(r / r[:, -1:]).mean(1) - 1e-6)
            l_lid = torch.log(lid.clamp_min(1e-6)).mean()
            loss = loss + lam["lid"] * l_lid
            parts["lid"] = l_lid.item()
        if lam["iso"]:
            zc = z - z.mean(0, keepdim=True)
            cov = zc.T @ zc / (m - 1)
            cov = cov / torch.diagonal(cov).sum().clamp_min(1e-12)  # eigenvalues now sum to 1
            l_iso = torch.log(d_out * (cov**2).sum())  # = log(d / participation ratio), 0 when isotropic
            loss = loss + lam["iso"] * l_iso
            parts["iso"] = l_iso.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % int(hp["eval_every"]) == 0 or step == steps:
            r, h = evaluate()
            history.append({"step": step, "loss": loss.item(), **parts, "val_recall10": r, "hub_skew": h})
            log.info(f"    step {step}/{steps} loss {loss.item():.4f} val_recall@10 {r:.4f} hub_skew {h:.3f}")
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"head training diverged at step {step}: {parts}")
    return W.detach().cpu().numpy().astype(np.float32), mu, history


def _apply(x: np.ndarray, W: np.ndarray, mu: np.ndarray) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float32) - mu) @ W.T
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return z.astype(np.float32)
