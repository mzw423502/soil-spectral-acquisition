"""R5 publication-style figures for selective NIR–MIR acquisition.

Only the presentation changes relative to R2. Allocation, ties, sample sets,
gain definitions, bootstrap seeds, models and all numeric output tables retain
the R2 calculation logic. Run with `python soil_r3_figures.py` using the bundled
source snapshot, or pass --source and --out to use a different location.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
if not SOURCE.exists():
    SOURCE = HERE.parent / "Soil_SOC_R2" / "reproduction"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=SOURCE)
parser.add_argument("--out", type=Path, default=HERE)
args = parser.parse_args()
base = args.source / "stage2_batch2"
out = args.out
figs = out / "figures"
tables = out / "tables"
figs.mkdir(parents=True, exist_ok=True)
tables.mkdir(exist_ok=True)

# STYLE 1: one consistent, low-saturation palette; color denotes a quantity
# or comparison, never a decorative background. The data remain unchanged.
names = {"neospectra": "NeoSpectra", "icraf": "ICRAF–ISRIC", "lucas": "LUCAS–Woodwell"}
keys = list(names)
navy = "#466686"
teal = "#397D76"
orange = "#B28656"
red = "#AE5556"
grey = "#888888"
dark = "#222222"
pal = [navy, teal, orange]

# STYLE 2: use genuine Arial or Helvetica (rather than a silent default-font
# substitution). Explicitly set every typographic level and export fonts.
available = {f.name for f in font_manager.fontManager.ttflist}
font = next((f for f in ["Arial", "Helvetica"] if f in available), None)
if font is None:
    raise RuntimeError("Arial or Helvetica is required. Install one before rendering.")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [font],
    "font.size": 9, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "legend.frameon": False,
    "axes.titlesize": 10, "axes.edgecolor": dark, "axes.linewidth": .75,
    "axes.labelcolor": dark, "text.color": dark,
    "xtick.color": dark, "ytick.color": dark,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": .65, "ytick.major.width": .65,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "mathtext.fontset": "custom",
    "mathtext.rm": font, "mathtext.it": font + ":italic", "mathtext.bf": font + ":bold",
    "axes.unicode_minus": True,
    "legend.handlelength": 2.3, "legend.handletextpad": .55,
})


def panel(ax, letter, *, x=-.10, y=1.045):
    """STYLE 3: panel letter only; no summary title or data-set heading."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom", clip_on=False)


def grid(ax):
    """STYLE 4: remove all decorative grids, including major/minor grids.

    Zero/budget/identity lines below denote scientific reference values,
    not grid lines. No vertical or horizontal chart grid is drawn.
    """
    ax.grid(False, which="both", axis="both")
    ax.set_axisbelow(True)


def legend(ax, **kw):
    """STYLE 5: borderless, no shading, uniform 8-point typography."""
    return ax.legend(frameon=False, fontsize=8, borderaxespad=.2, **kw)


def refline(ax, value=0, axis="y", ls="--"):
    """Semantic reference line, intentionally distinct from a grid."""
    func = ax.axhline if axis == "y" else ax.axvline
    return func(value, color="#777777", lw=.8, ls=ls, zorder=1)


def save(fig, name):
    # Audit the requested styling on every plotted axes before exporting.
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        grid(ax)
        assert not ax.get_title(), "Use panel letters, not titles."
        for which in (ax.xaxis, ax.yaxis):
            assert not any(line.get_visible() for line in which.get_gridlines())
        leg = ax.get_legend()
        if leg is not None:
            assert not leg.get_frame_on()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(figs / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The following data and statistical functions are unchanged from R2.
# ---------------------------------------------------------------------------
d = pd.read_csv(base / "predictions/all_corrected_predictions.csv",
                dtype={"sample_id": str, "group": str})
m = pd.read_csv(base / "predictions/m5_fixed.csv", dtype={"sample_id": str})
d = d.merge(m[["dataset", "sample_id", "outer_fold", "pred_M5_fixed"]],
            on=["dataset", "sample_id", "outer_fold"], validate="one_to_one")
assert len(d) == 6244 and d.groupby("dataset").size().to_dict() == {
    "neospectra": 1974, "icraf": 3698, "lucas": 572}
d["pred_M5"] = d.pop("pred_M5_fixed")
for j in range(1, 6):
    d[f"e{j}"] = abs(d.y - d[f"pred_M{j}"])
d["GM"] = d.e1 - d.e2
d["GF"] = d.e1 - d.e4


def subset(ds):
    return d[d.dataset == ds].copy()


def select(g, p, score="pred_M1"):
    # Identical deterministic tie-break and floor rounding to R2.
    return g.sort_values([score, "sample_id"], ascending=[False, True],
                         kind="mergesort").index[:int(np.floor(p * len(g) + 1e-10))]


def alloc(g, p, post="e4", score="pred_M1"):
    err = g.e1.copy()
    rand = g.e1.copy()
    chosen = pd.Series(False, index=g.index)
    for _, f in g.groupby("outer_fold", sort=False):
        idx = select(f, p, score)
        chosen.loc[idx] = True
        err.loc[idx] = f.loc[idx, post]
        q = len(idx) / len(f)
        rand.loc[f.index] = (1-q) * f.e1 + q * f[post]
    return err, rand, chosen


def boot(g, delta, seed=20261125, B=3000):
    z = pd.DataFrame({"group": g.group, "delta": delta}).groupby("group").delta.agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(z), size=(B, len(z)))
    v = z["sum"].to_numpy()[idx].sum(1) / z["size"].to_numpy()[idx].sum(1)
    return np.quantile(v, [.025, .975])


def fmt_soc(ax, suffix=""):
    tick = np.array([0, .5, 2, 5, 10, 25, 50])
    ax.set_xticks(np.log1p(tick))
    ax.set_xticklabels([f"{x:g}" for x in tick])
    ax.set_xlabel("SOC (%; log(1 + SOC) spacing)" + suffix)


# ---------------------------------------------------------------------------
# Figure 1. Cohort structure and explicit acquisition routes.
# Data-set identities are axis-label qualifiers, not titles. The circuit is
# reconstructed with rectangular white boxes and black solid/dashed arrows.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10.5, 6.1), layout="constrained")
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.0])
for i, k in enumerate(keys):
    ax = fig.add_subplot(gs[0, i]); g = subset(k)
    ax.hist(np.log1p(g.y), bins=25, color=pal[i], alpha=.85,
            edgecolor="white", lw=.3)
    panel(ax, "abc"[i]); fmt_soc(ax, "\n" + names[k])
    ax.set_ylabel("Soil samples" if i == 0 else "")
    unit = ["pedons", "profiles", "1° spatial cells"][i]
    ax.text(.97, .94, f"n = {len(g):,}\n{g.group.nunique():,} {unit}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))

ax = fig.add_subplot(gs[1, :]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
panel(ax, "d", x=0, y=.96)


def box(x, y, w, h, label):
    # Sharp-edged, unfilled-looking scholarly block; no colored fills or shadows.
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=dark, lw=.85))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=9, linespacing=1.25)


def arrow(x, y, X, Y, *, control=False):
    # Solid = data flow. Dashed = locked calibration/decision control.
    ax.add_patch(FancyArrowPatch((x,y), (X,Y), arrowstyle="-|>",
                               mutation_scale=9, lw=.85, color=dark,
                               linestyle=(0, (3, 3)) if control else "-",
                               shrinkA=0, shrinkB=0))


box(.025, .35, .18, .20, "All samples\nNIR spectrum\nCost = r")
box(.255, .35, .19, .20, "Initial SOC prediction\nRank / threshold")
box(.515, .58, .19, .18, "Selected subset\nAcquire MIR")
box(.515, .13, .19, .18, "Remaining samples\nNo additional scan")
box(.775, .58, .20, .18, "MIR replacement or\nNIR–MIR fusion")
box(.775, .13, .20, .18, "Retain NIR prediction")
box(.115, .77, .365, .17, "Calibration: predictors, fusion weight,\nthreshold / budget fixed before validation")
arrow(.205,.45,.255,.45)
arrow(.445,.45,.515,.67); arrow(.445,.45,.515,.22)
arrow(.705,.67,.775,.67); arrow(.705,.22,.775,.22)
arrow(.35,.77,.35,.55,control=True)
# Minimal line key and cost definition; no heading or decorative background.
ax.plot([.025,.065],[.035,.035],color=dark,lw=.85); ax.text(.074,.035,"Data flow",va="center",fontsize=8)
ax.plot([.20,.24],[.035,.035],color=dark,lw=.85,ls=(0,(3,3))); ax.text(.25,.035,"Calibration control",va="center",fontsize=8)
ax.text(.975,.035,"Total marginal cost = r + q",ha="right",va="center",fontsize=9)
save(fig, "Figure_01")


# ---------------------------------------------------------------------------
# Figure 2. Individual gains and empirical association. Lower axes retain the
# actual quintile medians; a single unframed legend describes both gain types.
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.2), layout="constrained")
gain_rows = []
for i, k in enumerate(keys):
    g = subset(k); ax = axs[0,i]; x = np.log1p(g.y)
    ax.scatter(x,g.GF,s=8,c=np.where(g.GF<0,red,teal),alpha=.24,
               linewidths=0,rasterized=True)
    b = pd.qcut(g.y,10,duplicates="drop")
    z = g.groupby(b,observed=True).agg(x=("y","median"),v=("GF","median"))
    ax.plot(np.log1p(z.x),z.v,color=dark,lw=1.25,marker="o",ms=2.7)
    refline(ax); fmt_soc(ax, "\n"+names[k]); ax.set_yscale("symlog",linthresh=.3)
    ax.set_ylabel("Fusion gain (SOC pp; symlog)" if i==0 else "")
    # Move numeric annotation into the empty lower-right region, away from points.
    panel(ax,"abc"[i]); ax.text(.97,.06,f"Negative gain: {(g.GF<0).mean():.1%}",
                                 transform=ax.transAxes,ha="right",va="bottom",fontsize=8.5)
    ax=axs[1,i]
    # Quantile assignment is for analysis, not decision fitting.
    z=g.assign(q=pd.qcut(g.pred_M1.rank(method="first"),5,labels=False)+1).groupby("q").agg(
        pred=("pred_M1","median"),GM=("GM","mean"),GF=("GF","mean"),n=("y","size"))
    z["dataset"]=k; gain_rows.append(z.reset_index()); p=np.arange(5)
    ax.bar(p-.17,z.GM,.34,color=navy,label="MIR replacement",linewidth=0)
    ax.bar(p+.17,z.GF,.34,color=teal,label="Optimized fusion",linewidth=0)
    refline(ax,ls="-"); ax.set_xticks(p,[f"Q{j}\n{v:.2f}%" for j,v in zip(z.index,z.pred)])
    ax.set_xlabel("NIR-predicted SOC quintile (median)")
    ax.set_ylabel("Mean gain (SOC pp)" if i==0 else ""); panel(ax,"def"[i])
    if i==0: legend(ax,loc="upper left")
pd.concat(gain_rows).to_csv(tables/"gain_by_predicted_SOC_quintile.csv",index=False)
save(fig,"Figure_02")


# ---------------------------------------------------------------------------
# Figure 3. Exact-budget curves and fixed 20% contrasts.
# All comparisons and bootstrap calls remain the R2 calls. Position effect
# text in a dedicated aligned column rather than attaching it to CI tips.
# ---------------------------------------------------------------------------
curves=[]; contrasts=[]; sens=[]
fig=plt.figure(figsize=(10.5,6.0),layout="constrained")
gs=fig.add_gridspec(2,3,height_ratios=[1.15,.78])
for i,k in enumerate(keys):
    g=subset(k); ax=fig.add_subplot(gs[0,i]); pgrid=np.linspace(0,1,21); rr=[]
    for p in pgrid:
        pr,ra,sel=alloc(g,p); oracle,_,_=alloc(g,p,score="GF")
        rr.append({"dataset":k,"target_p":p,"actual_p":sel.mean(),"priority":pr.mean(),
                   "random":ra.mean(),"oracle":oracle.mean()})
    z=pd.DataFrame(rr); curves.append(z)
    ax.plot(z.target_p*100,z.random,c=grey,ls="--",lw=1.35,label="Exact-budget random")
    ax.plot(z.target_p*100,z.priority,c=red,lw=1.75,label="SOC priority")
    # The retrospective optimum is the minimum attainable MAE at this budget
    # for the fixed model; retain the historical internal CSV column name.
    ax.plot(z.target_p*100,z.oracle,c=navy,ls=":",lw=1.45,label="Retrospective optimum")
    refline(ax,20,"x",ls=":"); ax.set_xlabel("Target MIR fraction (%)\n"+names[k])
    ax.set_ylabel("MAE (SOC pp)" if i==0 else ""); panel(ax,"abc"[i]); ax.set_xticks([0,20,40,60,80,100])
    if i==0: legend(ax,loc="upper right")
    pr,ra,sel=alloc(g,.2); lo,hi=boot(g,pr-ra)
    contrasts.append({"dataset":k,"n":len(g),"k":int(sel.sum()),"random":ra.mean(),
                      "priority":pr.mean(),"diff":(pr-ra).mean(),"lo":lo,"hi":hi,
                      "relative_pct":100*(ra.mean()-pr.mean())/ra.mean()})
    for loss in ["absolute","log1p"]:
        h=g.copy()
        if loss=="log1p":
            for j in [1,4]: h[f"e{j}"]=abs(np.log1p(h.y)-np.log1p(h[f"pred_M{j}"]))
        a,b,_=alloc(h,.2)
        for trunc in [False,True]:
            keep=h.y<=h.y.quantile(.95) if trunc else pd.Series(True,index=h.index)
            diff=a-b; ll,hh=boot(h[keep],diff[keep])
            sens.append({"dataset":k,"loss":loss,"exclude_top_SOC_5pct":trunc,"n":int(keep.sum()),
                         "priority":a[keep].mean(),"random":b[keep].mean(),
                         "relative_pct":100*(b[keep].mean()-a[keep].mean())/b[keep].mean(),
                         "diff_lo":ll,"diff_hi":hh})
ax=fig.add_subplot(gs[1,:]); c=pd.DataFrame(contrasts); yy=np.arange(3)
refline(ax,0,"x"); ax.errorbar(c["diff"],yy,xerr=[c["diff"]-c.lo,c.hi-c["diff"]],
                               fmt="o",ms=4.5,color=teal,capsize=2.5,lw=1.1)
ax.set_yticks(yy,[names[k] for k in keys]); ax.set_ylim(2.6,-.6)
ax.set_xlabel("Priority − random MAE (SOC pp; 95% cluster bootstrap interval)")
panel(ax,"d",x=-.025)
for i,row in c.iterrows():
    ax.text(.97,i,f"{row.relative_pct:.1f}% lower",transform=ax.get_yaxis_transform(),
            ha="right",va="center",fontsize=8.5)
ax.set_xlim(min(c.lo)-.04,.13)
pd.concat(curves).to_csv(tables/"budget_curves.csv",index=False)
c.to_csv(tables/"main_20pct.csv",index=False)
pd.DataFrame(sens).to_csv(tables/"loss_sensitivity.csv",index=False)
save(fig,"Figure_03")


# ---------------------------------------------------------------------------
# Figure 4. Cost frontiers and independent decisions. Keep full-pair model
# reference markers, use lower-weight lines and shape-coded reference points.
# ---------------------------------------------------------------------------
nc=pd.read_csv(base/"results/nested_cost_policy_results.csv")
nc=nc[np.isclose(nc.r,.1)&np.isclose(nc.cap,.7)]
costs=[]; fig=plt.figure(figsize=(10.5,6.0),layout="constrained")
gs=fig.add_gridspec(2,3,height_ratios=[1.2,.78])
for i,k in enumerate(keys):
    g=subset(k); ax=fig.add_subplot(gs[0,i]); z=curves[i]
    ax.plot(.1+z.actual_p,z.random,c=grey,ls="--",lw=1.35,label="Random → fusion")
    ax.plot(.1+z.actual_p,z.priority,c=red,lw=1.75,label="SOC priority → fusion")
    ax.scatter([.1,1,1.1,1.1],[g.e1.mean(),g.e2.mean(),g.e4.mean(),g.e5.mean()],
               c=[grey,navy,teal,orange],s=35,marker="D",zorder=4)
    # Stop the semantic budget reference below the legend-only region so it
    # never crosses unframed legend text. Its x value is unchanged.
    ax.axvline(.7,ymin=0,ymax=.75,color="#777777",lw=.8,ls=":",zorder=1)
    ep=g.e1.copy(); nsel=0
    for _,r in nc[nc.dataset==k].iterrows():
        f=g[g.outer_fold==r.outer_fold]; p=float(r.chosen.split("p=")[1]); post="e4" if "M4" in r.chosen else "e2"
        a,_,ch=alloc(f,p,post); ep.loc[f.index]=a; nsel+=int(ch.sum())
    cost=.1+nsel/len(g); delta=ep-g.e4; lo,hi=boot(g,delta)
    costs.append({"dataset":k,"realized_cost":cost,"MAE":ep.mean(),"full_M4":g.e4.mean(),
                  "full_M5":g.e5.mean(),"delta":delta.mean(),"lo":lo,"hi":hi})
    ax.scatter(cost,ep.mean(),marker="*",s=115,c=dark,edgecolor="white",lw=.45,
               zorder=6,label="Training-selected policy")
    ax.set_xlabel("Normalized marginal cost\n"+names[k]); ax.set_ylabel("MAE (SOC pp)" if i==0 else "")
    panel(ax,"abc"[i]); ax.set_xlim(.025,1.185); ax.yaxis.set_major_locator(MaxNLocator(6))
    if i==0: legend(ax,loc="upper right")
    if i==1:
        legend(ax,handles=[Line2D([],[],marker="D",ls="none",color=col,label=s,ms=4.5)
                           for col,s in zip([grey,navy,teal,orange],
                                            ["Full NIR","Full MIR","Full M4","Full M5"])],loc="upper right")
ax=fig.add_subplot(gs[1,:]); costdf=pd.DataFrame(costs); yy=np.arange(3)
refline(ax,0,"x"); ax.errorbar(costdf.delta,yy,xerr=[costdf.delta-costdf.lo,costdf.hi-costdf.delta],
                               fmt="o",color=teal,ms=4.5,capsize=2.5,lw=1.1)
ax.set_yticks(yy,[names[k] for k in keys]); ax.set_ylim(2.6,-.6)
ax.set_xlabel("Selected policy − full M4 MAE (SOC pp; 95% cluster bootstrap interval)")
panel(ax,"d",x=-.025)
for i,row in costdf.iterrows():
    ax.text(.97,i,f"{100*(1-row.realized_cost/1.1):.1f}% lower scenario cost",
            transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=8.5)
ax.set_xlim(min(costdf.lo)-.02,max(costdf.hi)+.13)
costdf.to_csv(tables/"locked_cost_policy.csv",index=False)
save(fig,"Figure_04")


# ---------------------------------------------------------------------------
# Figure 5. Thresholds and geographic shifts. Numeric axes, no commentary
# titles; the fixed 2% threshold is described by an axis label and caption.
# ---------------------------------------------------------------------------
t=pd.read_csv(base/"results/threshold_results.csv")
with open(base/"results/lucas_country_holdout_results.json",encoding="utf-8") as f:
    loco=json.load(f)["countries"]
l=pd.DataFrame(loco)
fig,axs=plt.subplots(1,3,figsize=(10.5,3.5),layout="constrained")
ax=axs[0]; fixed=t[(t.kind=="fixed")&np.isclose(t.threshold,2)]
rate=fixed.groupby("dataset").n_trigger.sum()/fixed.groupby("dataset").n_test.sum()
ax.bar(np.arange(3),[rate[k]*100 for k in keys],width=.6,color=pal,linewidth=0)
ax.set_xticks(range(3),["NeoSpectra","ICRAF","LUCAS"]); ax.set_ylabel("Actual MIR acquisition (%)")
ax.set_xlabel("Fixed threshold: 2% SOC"); panel(ax,"a")
for i,k in enumerate(keys): ax.text(i,rate[k]*100+1.25,f"{rate[k]*100:.1f}%",ha="center",fontsize=8.5)
ax.set_ylim(0,70)
ax=axs[1];qt=t[t.kind=="quantile"]
for k,col,marker in zip(keys,pal,["o","s","^"]):
    z=qt[qt.dataset==k].groupby("target_rate")[["n_trigger","n_test"]].sum()
    ax.plot(z.index*100,z.n_trigger/z.n_test*100,marker=marker,ls="-",color=col,
            label=names[k],ms=4,lw=1.2)
ax.plot([0,60],[0,60],"--",c=grey,lw=.8)
# The tested targets are 10–50%; zoom to their observed domain without
# discarding observations. Equal limits preserve the identity-line meaning.
ax.set(xlim=(0,60),ylim=(0,60))
ax.set_xlabel("Target MIR fraction (%)");ax.set_ylabel("Actual MIR acquisition (%)")
panel(ax,"b");legend(ax,loc="upper left")
ax=axs[2];ax.barh(np.arange(len(l)),l.p_thresh*100,color=orange,height=.58,linewidth=0)
refline(ax,20,"x"); ax.set_yticks(range(len(l)),l.country.str.replace("Czech Republic","Czechia"))
ax.invert_yaxis();ax.set_xlabel("Actual MIR acquisition (%)\nLUCAS country holdout");panel(ax,"c")
save(fig,"Figure_05")


# ---------------------------------------------------------------------------
# Figure 6. Geographic transfer, empirical gain cells, adjusted associations.
# Diverging cells retain original numbers/masking; coefficient zero is a
# reference line. The heatmap colorbar uses the same typographic hierarchy.
# ---------------------------------------------------------------------------
fig=plt.figure(figsize=(10.5,6.8),layout="constrained");gs=fig.add_gridspec(2,2,height_ratios=[1,1.03])
ax=fig.add_subplot(gs[0,:]);x=np.arange(len(l));ww=.19
for j,(col,label,cx) in enumerate([("mae_rand20_analytic","Random 20%",grey),
        ("mae_sel20","Priority 20%",red),("mae_adaptive","Training-selected budget",navy),
        ("mae_M4","Full fusion",teal)]):
    ax.bar(x+(j-1.5)*ww,l[col],ww,label=label,color=cx,linewidth=0)
ax.set_xticks(x,[f'{v.replace("Czech Republic","Czechia")}\n(n={n})' for v,n in zip(l.country,l.n)])
ax.set_ylabel("MAE (SOC pp)");panel(ax,"a",x=-.035)
ax.set_ylim(0,1.17*max(l.mae_rand20_analytic));legend(ax,ncol=4,loc="upper left")

ax=fig.add_subplot(gs[1,0]);g=subset("icraf")
g["SOC_bin"]=pd.cut(g.y,[-np.inf,.5,1,2,5,np.inf],labels=["≤0.5","0.5–1","1–2","2–5",">5"])
g["depth_bin"]=pd.cut(g.depth,[-np.inf,20,50,100,np.inf],labels=["0–20",">20–50",">50–100",">100"])
z=g.groupby(["depth_bin","SOC_bin"],observed=False).GF.agg(["mean","size"]).reset_index()
means=z.pivot(index="depth_bin",columns="SOC_bin",values="mean")
n=z.pivot(index="depth_bin",columns="SOC_bin",values="size")
means=means.mask(n<20); lim=max(.25,np.nanmax(abs(means.values)))
# Muted, balanced diverging palette. Masked cells stay visibly neutral.
cmap=LinearSegmentedColormap.from_list("gain",[red,"#F7F7F7",teal]);cmap.set_bad("#EFEFEF")
im=ax.imshow(means,cmap=cmap,norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect="auto",interpolation="nearest")
ax.set_xticks(range(5),means.columns); ax.set_yticks(range(4),means.index)
ax.set_xlabel("Reference SOC (%; retrospective)"); ax.set_ylabel("Depth (cm)"); panel(ax,"b")
for i in range(4):
    for j in range(5):
        v=means.iloc[i,j];nn=n.iloc[i,j]
        ax.text(j,i,("—" if pd.isna(v) else f"{v:+.2f}")+f"\n(n={nn})",
                ha="center",va="center",fontsize=8,
                color="white" if pd.notna(v) and abs(v)>.65*lim else dark)
cb=fig.colorbar(im,ax=ax,shrink=.78,pad=.025,label="Mean gain (SOC pp)")
cb.outline.set_linewidth(.6);cb.ax.tick_params(labelsize=8.5)
z.to_csv(tables/"icraf_SOC_depth_empirical_cells.csv",index=False)

ax=fig.add_subplot(gs[1,1]);
with open(base/"results/icraf_gain_regression_results.json",encoding="utf-8") as fh: f=json.load(fh)
terms=["log_depth","sand","clay","pH"];labs=["log(1 + depth)","Sand (10 pp)","Clay (10 pp)","pH"]
ys=np.arange(4);refline(ax,0,"x")
for j,(key,col,marker) in enumerate([("G_M",navy,"o"),("G_F",teal,"s")]):
    z=f[key]["L3_full"]
    ts=[x if x in z["params"] else x.lower() for x in terms];scale=np.array([1,10,10,1])
    coef=np.array([z["params"][s] for s in ts])*scale;se=np.array([z["bse"][s] for s in ts])*scale
    ax.errorbar(coef,ys+(j-.5)*.22,xerr=1.96*se,fmt=marker,ms=4.2,color=col,
                capsize=2.2,lw=1.05,label="Replacement" if j==0 else "Fusion")
ax.set_yticks(ys,labs);ax.set_ylim(3.7,-.7)
ax.set_xlabel("Adjusted association (SOC pp, 95% CI)");panel(ax,"c",x=-.035)
legend(ax,loc="best");save(fig,"Figure_06")


# Model performance output retained exactly, even though no model was retrained.
metrics=[]
for k in keys:
    g=subset(k)
    for j in range(1,6):
        res=g[f"pred_M{j}"]-g.y
        metrics.append({"dataset":k,"model":f"M{j}","MAE":abs(res).mean(),
                        "RMSE":np.sqrt((res**2).mean()),
                        "R2":1-(res**2).sum()/((g.y-g.y.mean())**2).sum(),"bias":res.mean()})
pd.DataFrame(metrics).to_csv(tables/"model_metrics.csv",index=False)
print(f"Six figures and tables generated with {font}: {out.resolve()}")
