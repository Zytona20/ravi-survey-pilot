"""Generate the LaTeX (pgfplots) code of Figure 3 (six geometric properties).

Panels (b) and (c) are computed: (b) exactly, from the chi distribution of the
distance between two Gaussian points; (c) by simulation with a fixed seed.
The other panels are schematic. Output: fig3_block.tex (the \\FigGeometry command).
"""
import numpy as np
from scipy.special import gammaln

rng = np.random.default_rng(7)


def coords(xy, fmt="{:.3f}"):
    return " ".join("(" + fmt.format(x) + "," + fmt.format(y) + ")" for x, y in xy)


# (a) a 1-D curve in 2-D: points along a spiral
th = np.linspace(0.7 * np.pi, 4.1 * np.pi, 75)
a_pts = np.c_[th * np.cos(th), th * np.sin(th)] / (4.1 * np.pi)

# (b) distance between two iid N(0, I_d) points is sqrt(2) * chi_d; divide by the mean
def chi_density_over_mean(d, xs):
    # r = sqrt(2) chi_d; mean m = sqrt(2) * sqrt(2) * Gamma((d+1)/2) / Gamma(d/2)
    m_chi = np.sqrt(2) * np.exp(gammaln((d + 1) / 2) - gammaln(d / 2))
    c = xs * m_chi  # value of chi_d at x * mean
    logf = (d - 1) * np.log(np.maximum(c, 1e-300)) - c ** 2 / 2 - (d / 2 - 1) * np.log(2) - gammaln(d / 2)
    f = np.exp(logf) * m_chi
    return f / f.max()

xs_b = np.linspace(0.0, 2.6, 261)
b2 = chi_density_over_mean(2, xs_b)
xs_n = np.linspace(0.75, 1.25, 101)
b200 = chi_density_over_mean(200, xs_n)

# (c) k-occurrence O_10 of n Gaussian points, sorted in decreasing order
def k_occurrence(n, d, k=10):
    x = rng.standard_normal((n, d))
    sq = (x ** 2).sum(1)
    dist = sq[:, None] + sq[None, :] - 2 * x @ x.T
    np.fill_diagonal(dist, np.inf)
    nn = np.argpartition(dist, k, axis=1)[:, :k]
    return np.sort(np.bincount(nn.ravel(), minlength=n))[::-1]

n_c = 2000
o3 = k_occurrence(n_c, 3)
o100 = k_occurrence(n_c, 100)
ranks = np.linspace(0, 1, n_c)
keep = np.unique(np.r_[np.arange(0, 60), np.linspace(0, n_c - 1, 160).astype(int), n_c - 1])
c3 = np.c_[ranks[keep], o3[keep]]
c100 = np.c_[ranks[keep], np.minimum(o100[keep], 62)]
stats_c = dict(max3=int(o3.max()), max100=int(o100.max()), zero3=float((o3 == 0).mean()), zero100=float((o100 == 0).mean()))

# (d) directions: isotropic vs. a cone (unit vectors, drawn as rays)
ang_iso = np.linspace(0, 2 * np.pi, 25)[:-1] + 0.05 * rng.standard_normal(24)
ang_cone = np.deg2rad(62 + 13 * rng.standard_normal(24))

# (e) clusters: balanced and separated vs. imbalanced and overlapping
def blob(cx, cy, s, n):
    return np.c_[cx + s * rng.standard_normal(n), cy + s * rng.standard_normal(n)]

e_left = [blob(-0.64, 0.50, 0.075, 30), blob(-0.14, 0.50, 0.075, 30), blob(-0.39, 0.02, 0.075, 30)]
e_right = [blob(0.42, 0.25, 0.22, 60), blob(0.55, 0.45, 0.09, 14), blob(0.47, 0.03, 0.09, 14)]

# (f) query-base shift
f_db = np.c_[0.0 + 0.30 * rng.standard_normal(90), 0.0 + 0.20 * rng.standard_normal(90)]
f_in = np.c_[0.0 + 0.25 * rng.standard_normal(10), 0.0 + 0.16 * rng.standard_normal(10)]
f_ood = np.c_[0.95 + 0.09 * rng.standard_normal(10), 0.55 + 0.07 * rng.standard_normal(10)]

col3 = ["encA", "encB", "encC"]

# ---------------------------------------------------------------- LaTeX
W, H, GAP = "18.3mm", "15mm", "4.4mm"
common = (f"width={W}, height={H}, scale only axis, font=\\scriptsize, "
          "title style={font=\\scriptsize\\bfseries, yshift=-1mm}")
hidden = "hide axis, clip=false"

def axis_at(prev):
    return f"at={{($({prev}.east)+({GAP},0)$)}}, anchor=west" if prev else ""

L = []
L.append("\\newcommand{\\FigGeometry}{%")
L.append("\\begin{figure*}[t]")
L.append("\\centering")
L.append("\\begin{tikzpicture}")
# (a)
L.append(f"\\begin{{axis}}[name=ga, {common}, {hidden}, title={{(a) Intrinsic dim.}}, xmin=-1.29, xmax=1.29, ymin=-1.05, ymax=1.05]")
L.append(f"\\addplot[only marks, mark=*, mark size=0.55pt, color=encA] coordinates {{{coords(a_pts)}}};")
L.append("\\end{axis}")
# (b)
L.append(f"\\begin{{axis}}[name=gb, {axis_at('ga')}, {common}, title={{(b) Concentration}}, xmin=0, xmax=2.6, ymin=0, ymax=1.12, "
         "xtick={0,1,2}, ytick=\\empty, axis lines=left, axis line style={-, black!50}, tick style={black!50}, "
         "clip=false, every axis plot/.append style={line width=0.8pt}]")
L.append(f"\\addplot[color=encA, fill=encA, fill opacity=0.12] coordinates {{{coords(np.c_[xs_b, b2])}}} \\closedcycle;")
L.append(f"\\addplot[color=encB, fill=encB, fill opacity=0.12] coordinates {{{coords(np.c_[xs_n, b200])}}} \\closedcycle;")
L.append("\\node[font=\\tiny, text=encA, anchor=west] at (axis cs:1.62,0.50) {$d=2$};")
L.append("\\node[font=\\tiny, text=encB, anchor=west] at (axis cs:1.16,0.95) {$d=200$};")
L.append("\\end{axis}")
# (c)
L.append(f"\\begin{{axis}}[name=gc, {axis_at('gb')}, {common}, title={{(c) Hubness}}, xmin=0, xmax=1, ymin=-13, ymax=62, "
         "xtick=\\empty, ytick={10}, axis y line=left, axis x line=none, axis line style={-, black!50}, tick style={black!50}, "
         "clip=false, every axis plot/.append style={line width=0.8pt}]")
L.append("\\draw[densely dashed, black!40] (axis cs:0,10) -- (axis cs:1,10);")
L.append(f"\\addplot[color=encA] coordinates {{{coords(c3)}}};")
L.append(f"\\addplot[color=encB] coordinates {{{coords(c100)}}};")
L.append("\\node[font=\\tiny, text=encB, anchor=west] at (axis cs:0.06,52) {hubs};")
L.append("\\node[font=\\tiny, text=encB, anchor=north east] at (axis cs:1,-1.5) {antihubs};")
L.append("\\node[font=\\tiny, text=encA, anchor=south west] at (axis cs:0.30,12) {$d=3$};")
L.append("\\node[font=\\tiny, text=encB, anchor=south west] at (axis cs:0.30,24) {$d=100$};")
L.append("\\end{axis}")
# (d)
L.append(f"\\begin{{axis}}[name=gd, {axis_at('gc')}, {common}, {hidden}, title={{(d) Anisotropy}}, xmin=-1.1, xmax=3.3, ymin=-1.80, ymax=1.80]")
L.append("\\draw[black!25] (axis cs:0,0) circle[radius=1];")
L.append("\\draw[black!25] (axis cs:2.2,0) circle[radius=1];")
for a in ang_iso:
    L.append(f"\\draw[encA, line width=0.45pt] (axis cs:0,0) -- (axis cs:{np.cos(a):.3f},{np.sin(a):.3f});")
for a in ang_cone:
    L.append(f"\\draw[encB, line width=0.45pt] (axis cs:2.2,0) -- (axis cs:{2.2 + np.cos(a):.3f},{np.sin(a):.3f});")
L.append("\\end{axis}")
# (e)
L.append(f"\\begin{{axis}}[name=ge, {axis_at('gd')}, {common}, {hidden}, title={{(e) Clusters}}, xmin=-0.85, xmax=0.95, ymin=-0.35, ymax=0.8]")
for pts, c in zip(e_left, col3):
    L.append(f"\\addplot[only marks, mark=*, mark size=0.45pt, color={c}] coordinates {{{coords(pts)}}};")
for pts, c in zip(e_right, col3):
    L.append(f"\\addplot[only marks, mark=*, mark size=0.45pt, color={c}] coordinates {{{coords(pts)}}};")
L.append("\\end{axis}")
# (f)
L.append(f"\\begin{{axis}}[name=gf, {axis_at('ge')}, {common}, {hidden}, title={{(f) Query shift}}, xmin=-0.85, xmax=1.2, ymin=-0.6, ymax=0.8]")
L.append(f"\\addplot[only marks, mark=*, mark size=0.45pt, color=black!35] coordinates {{{coords(f_db)}}};")
L.append(f"\\addplot[only marks, mark=triangle*, mark size=1.0pt, color=encA] coordinates {{{coords(f_in)}}};")
L.append(f"\\addplot[only marks, mark=square*, mark size=0.8pt, color=encB] coordinates {{{coords(f_ood)}}};")
L.append("\\end{axis}")
subs = {"ga": "2-D data on a 1-D curve", "gb": "distance / mean", "gc": "points sorted by $O_{10}$",
        "gd": "isotropic \\qquad cone", "ge": "balanced \\quad overlapping", "gf": "ID vs.\\ OOD queries"}
for k, v in subs.items():
    L.append(f"\\node[font=\\tiny, text=black!60, anchor=north] at ($({k}.south)+(0,-3.6mm)$) {{{v}}};")
L.append("\\end{tikzpicture}")
L.append("\\Description{Six small panels in one row. (a) Points along a spiral: two ambient dimensions, one intrinsic dimension. "
         "(b) Distribution of pairwise distances divided by their mean: broad in 2 dimensions, a narrow peak at 1 in 200 dimensions. "
         "(c) k-occurrence counts sorted in decreasing order: nearly flat around 10 in 3 dimensions; in 100 dimensions a few points have very high counts and many have zero. "
         "(d) Unit vectors spread in all directions versus confined to a narrow cone. "
         "(e) Three balanced, separated clusters versus one large cluster overlapping two small ones. "
         "(f) Gray database points with in-distribution queries among them and out-of-distribution queries in a separate region.}")
L.append("\\caption{Six geometric properties of embedding spaces (Section~\\ref{sec:geometry}). "
         "(a) Two ambient dimensions but one intrinsic dimension. "
         "(b) Pairwise distances of Gaussian data divided by their mean: in 200 dimensions nearly all distances are equal. "
         "(c) The $k$-occurrence $O_{10}$ of 2,000 Gaussian points, sorted: in 100 dimensions a few hubs occur in many neighbor lists and many antihubs in none (dashed line: the mean). "
         "(d) Directions spread over the sphere or confined to a cone. "
         "(e) Balanced, separated clusters or imbalanced, overlapping ones. "
         "(f) Database points with in-distribution (ID) and out-of-distribution (OOD) queries. "
         "Panels (b) and (c) are computed, the others are schematic.}")
L.append("\\label{fig:geometry}")
L.append("\\end{figure*}")
L.append("}")
open("fig3_block.tex", "w").write("\n".join(L) + "\n")
print(stats_c)
