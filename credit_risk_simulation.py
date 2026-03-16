"""
Credit Risk Score Pipeline — v2
================================
Changes from v1:
  - Overall CO rate adjusted to ~25%
  - Pair selection: JMI + k-fold CV (CMI shown as complementarity diagnostic)
  - Tier assignment: 1.5x weighted-average CO rate rule
  - Outputs: Lorenz curves, JMI/CMI pair table, coloured score grid with annotations
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 150, "font.size": 9})

np.random.seed(42)

N       = 50_000
N_BINS  = 5
N_FOLDS = 5
MULT    = 1.5
N_TIERS = 5

TIER_ORDER = ["Low Risk", "Medium-Low", "Medium", "Medium-High", "High Risk"]
TIER_PALETTE = {
    "Low Risk":    "#2ecc71",
    "Medium-Low":  "#a8e063",
    "Medium":      "#f9c74f",
    "Medium-High": "#f76c1b",
    "High Risk":   "#e03131",
}
MODELS = ["Model_A", "Model_B", "Model_C", "Model_D", "Model_E"]


# ──────────────────────────────────────────────────────────────────────────────
# 1. SIMULATE DATASET  —  target overall CO rate ≈ 25 %
# ──────────────────────────────────────────────────────────────────────────────
def simulate_dataset(n: int) -> pd.DataFrame:
    latent = np.random.normal(0, 1, n)

    # intercept = -1.5  →  sigmoid(-1.5) ≈ 0.182 at latent=0, but because
    # E[sigmoid(1.5Z - 1.5)] with Z~N(0,1) integrates to ~25 % via probit approx
    prob = 1 / (1 + np.exp(-(1.5 * latent - 1.5)))
    co   = np.random.binomial(1, prob, n)

    def score(signal, noise):
        raw  = -signal * latent + np.random.normal(0, noise, n)
        norm = (raw - raw.min()) / (raw.max() - raw.min())
        return np.round(300 + norm * 550).astype(int)

    df = pd.DataFrame({
        "account_id": np.arange(1, n + 1),
        "charge_off": co,
        "Model_A": score(2.5, 0.6),   # strong
        "Model_B": score(2.2, 0.8),   # strong
        "Model_C": score(1.6, 1.2),   # moderate
        "Model_D": score(1.1, 1.6),   # weak
        "Model_E": score(0.4, 2.2),   # near-noise
    })
    for col in MODELS:
        df.loc[np.random.rand(n) < 0.05, col] = np.nan
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. LORENZ CURVES
#    X-axis : cumulative % of accounts, sorted riskiest first (score ascending)
#    Y-axis : cumulative % of charge-offs captured
#    Gini   : 2·AUC - 1  (area between Lorenz curve and the diagonal)
# ──────────────────────────────────────────────────────────────────────────────
def plot_lorenz_curves(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    colors      = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61", "#9e4fb5"]

    for i, model in enumerate(MODELS):
        sub = df[["charge_off", model]].dropna()
        sub = sub.sort_values(model, ascending=True).reset_index(drop=True)

        cum_accts = np.arange(1, len(sub) + 1) / len(sub)
        cum_co    = sub["charge_off"].cumsum() / sub["charge_off"].sum()

        # Gini = 2 * ROC_AUC - 1.
        # The CAP (Lorenz) curve AUC ≠ ROC AUC at non-50% bad rates, so we
        # compute ROC AUC directly: high score = low risk → negate for roc_auc_score.
        gini = round(2 * roc_auc_score(sub["charge_off"], -sub[model]) - 1, 3)
        ax.plot(cum_accts, cum_co,
                linestyle=line_styles[i], color=colors[i], linewidth=1.8,
                label=f"{model}  (Gini = {gini:.3f})")

    # Random-model diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random model")

    ax.set_xlabel("Cumulative % of Accounts  (sorted riskiest → safest)")
    ax.set_ylabel("Cumulative % of Charge-Offs Captured")
    ax.set_title("Lorenz Curves — Model Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("lorenz_curves.png", bbox_inches="tight")
    plt.close()
    print("    Saved: lorenz_curves.png")


# ──────────────────────────────────────────────────────────────────────────────
# 3. LORENZ CURVES — per-bin, conditioned on the best individual model
#    Layout: (N-1) rows  ×  N_BINS cols
#      rows  = each non-best model
#      cols  = score quintile of the best model (Bin 1 = lowest/riskiest score)
#    Within each cell: Lorenz curve of the other model restricted to that bin.
# ──────────────────────────────────────────────────────────────────────────────
def _compute_ginis(df: pd.DataFrame) -> dict:
    return {
        m: round(2 * roc_auc_score(
            df[["charge_off", m]].dropna()["charge_off"],
            -df[["charge_off", m]].dropna()[m],
        ) - 1, 3)
        for m in MODELS
    }


def plot_lorenz_by_best_model_bins(df: pd.DataFrame) -> None:
    """5 charts (one per best-model quintile bin), each overlaying all other models."""
    ginis = _compute_ginis(df)
    best_model   = max(ginis, key=lambda m: ginis[m])
    other_models = [m for m in MODELS if m != best_model]

    print(f"    Best individual model : {best_model}  (Gini = {ginis[best_model]:.3f})")

    # Bin by best model's quintiles (bin 0 = lowest score = riskiest)
    base = df[["charge_off", best_model] + other_models].dropna(subset=[best_model]).copy()
    base["_bin"], cuts = pd.qcut(
        base[best_model], q=N_BINS, labels=False, retbins=True, duplicates="drop"
    )

    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    colors      = ["#2c7bb6", "#d7191c", "#1a9641", "#9e4fb5"]

    fig, axes = plt.subplots(1, N_BINS, figsize=(N_BINS * 3.8, 4.2), squeeze=False)

    for bin_i in range(N_BINS):
        ax   = axes[0][bin_i]
        lo_s = f"{int(cuts[bin_i])}"
        hi_s = f"{int(cuts[bin_i + 1])}"
        bin_mask = base["_bin"] == bin_i

        # Random-model diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, alpha=0.5, label="Random")

        for m_i, other in enumerate(other_models):
            sub = base.loc[bin_mask, ["charge_off", other]].dropna()
            enough = (
                len(sub) >= 5
                and sub["charge_off"].sum() > 0
                and sub["charge_off"].sum() < len(sub)
            )
            if not enough:
                continue

            sub_s     = sub.sort_values(other, ascending=True).reset_index(drop=True)
            cum_accts = np.arange(1, len(sub_s) + 1) / len(sub_s)
            cum_co    = sub_s["charge_off"].cumsum() / sub_s["charge_off"].sum()
            gini      = round(2 * roc_auc_score(sub_s["charge_off"], -sub_s[other]) - 1, 3)

            ax.plot(cum_accts, cum_co,
                    color=colors[m_i % len(colors)],
                    linestyle=line_styles[m_i % len(line_styles)],
                    linewidth=1.6,
                    label=f"{other}  (Gini={gini:.3f})")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_title(
            f"Bin {bin_i + 1}  [{lo_s}–{hi_s}]\n({best_model} quintile)",
            fontsize=8.5, fontweight="bold",
        )
        ax.set_xlabel("Cum. % Accounts", fontsize=8)
        if bin_i == 0:
            ax.set_ylabel("Cum. % Charge-Offs Captured", fontsize=8)
        ax.legend(fontsize=6.5, loc="lower right")

    fig.suptitle(
        f"Lorenz Curves by {best_model} Score Bin  (best individual model, Gini = {ginis[best_model]:.3f})\n"
        f"Each chart: all other models' Lorenz curves within that {best_model} quintile",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("lorenz_by_best_model_bins.png", bbox_inches="tight")
    plt.close()
    print("    Saved: lorenz_by_best_model_bins.png")


# ──────────────────────────────────────────────────────────────────────────────
# 4. PAIR METRIC FUNCTIONS
#    Two complementary metrics computed in a single CV loop:
#
#    JMI  (selection criterion — total combined power)
#      Joint MI = H(Y) − H(Y|A,B)
#      Measures total combined information the pair carries about charge-off.
#      Selecting the highest-JMI pair maximises the pair's predictive ceiling.
#
#    CMI  (diagnostic — complementarity)
#      Symmetric conditional MI = [I(Y;B|A) + I(Y;A|B)] / 2
#      = [H(Y|A) + H(Y|B) − 2·H(Y|A,B)] / 2
#      Measures how much NEW information each model adds beyond the other.
#      Low CMI with high JMI signals strong but redundant models (governance flag).
# ──────────────────────────────────────────────────────────────────────────────
def _h(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)

def _make_cuts(scores, n_bins):
    _, cuts = pd.qcut(scores, q=n_bins, retbins=True, duplicates="drop")
    cuts[0], cuts[-1] = -np.inf, np.inf
    return cuts

def _apply_cuts(scores, cuts):
    return np.array(pd.cut(scores, bins=cuts, labels=False), dtype=float)

def _h_y_given_a(y, a_bins):
    n, h = len(y), 0.0
    for av in np.unique(a_bins[~np.isnan(a_bins)]):
        m = a_bins == av
        h += m.sum() / n * _h(y[m].mean())
    return h

def _h_y_given_ab(y, a_bins, b_bins):
    n = len(y)
    dt = pd.DataFrame({"y": y, "a": a_bins, "b": b_bins}).dropna()
    h  = 0.0
    for _, grp in dt.groupby(["a", "b"]):
        h += len(grp) / n * _h(grp["y"].mean())
    return h

def compute_pair_metrics_cv(df: pd.DataFrame, m1: str, m2: str) -> dict:
    """Compute symmetric CMI and JMI for a model pair via k-fold CV."""
    sub = df[[m1, m2, "charge_off"]].dropna().reset_index(drop=True)
    y, sa, sb = sub["charge_off"].values, sub[m1].values, sub[m2].values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    cmi_vals, jmi_vals = [], []
    for tr, te in kf.split(y):
        ca, cb  = _make_cuts(sa[tr], N_BINS), _make_cuts(sb[tr], N_BINS)
        ab, bb  = _apply_cuts(sa[te], ca),    _apply_cuts(sb[te], cb)
        h_y     = _h(y[te].mean())
        h_y_a   = _h_y_given_a(y[te], ab)
        h_y_b   = _h_y_given_a(y[te], bb)
        h_y_ab  = _h_y_given_ab(y[te], ab, bb)
        cmi_vals.append((max(0.0, h_y_a - h_y_ab) + max(0.0, h_y_b - h_y_ab)) / 2)
        jmi_vals.append(max(0.0, h_y - h_y_ab))
    return {
        "Pair":     f"{m1} + {m2}",
        "N_valid":  len(sub),
        "CMI_mean": round(np.mean(cmi_vals), 6),
        "CMI_std":  round(np.std(cmi_vals),  6),
        "JMI_mean": round(np.mean(jmi_vals), 6),
        "JMI_std":  round(np.std(jmi_vals),  6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. PAIR TABLE  —  both metrics shown; rows sorted by JMI (selection criterion)
#    Each pair carries both a JMI rank and a CMI rank so the contrast is visible
# ──────────────────────────────────────────────────────────────────────────────
def plot_pair_table(pair_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    display = pair_df[["JMI_Rank", "CMI_Rank", "Pair", "N_valid",
                        "JMI_mean", "JMI_std",
                        "CMI_mean", "CMI_std"]].copy()
    display.columns = ["JMI Rank", "CMI Rank", "Model Pair", "N Valid",
                       "JMI Mean (bits)", "JMI Std",
                       "CMI Mean (bits)", "CMI Std"]

    tbl = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    n_cols = len(display.columns)
    # Base header colour
    for j in range(n_cols):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # JMI columns (0,4,5) → dark green; CMI columns (1,6,7) → dark blue
    for j in (0, 4, 5):
        tbl[0, j].set_facecolor("#145a32")
    for j in (1, 6, 7):
        tbl[0, j].set_facecolor("#1a5276")

    # Highlight best pair (rank 1 = highest JMI = first row)
    for j in range(n_cols):
        tbl[1, j].set_facecolor("#d5f5e3")
        tbl[1, j].set_text_props(fontweight="bold")

    for i in range(2, len(display) + 1):
        clr = "#f8f9fa" if i % 2 == 0 else "white"
        for j in range(n_cols):
            tbl[i, j].set_facecolor(clr)

    ax.set_title(
        "Pair Ranking — sorted by JMI  (CMI rank shown for comparison)",
        fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("pair_table.png", bbox_inches="tight")
    plt.close()
    print("    Saved: pair_table.png")


# ──────────────────────────────────────────────────────────────────────────────
# 6. SCORE GRID  —  5 × 5 quintile bins; returns grid + cut points
# ──────────────────────────────────────────────────────────────────────────────
def build_score_grid(df: pd.DataFrame, m1: str, m2: str, n_buckets: int = 5):
    sub    = df[[m1, m2, "charge_off"]].dropna().copy()
    labels = list(range(n_buckets, 0, -1))   # label 1=safest (highest score), label 5=riskiest (lowest score)

    sub[f"{m1}_tier"], cuts_m1 = pd.qcut(sub[m1], q=n_buckets, labels=labels,
                                          retbins=True, duplicates="drop")
    sub[f"{m2}_tier"], cuts_m2 = pd.qcut(sub[m2], q=n_buckets, labels=labels,
                                          retbins=True, duplicates="drop")

    grid = (
        sub.groupby([f"{m1}_tier", f"{m2}_tier"], observed=True)
        .agg(N=("charge_off", "count"), ChargeOffs=("charge_off", "sum"))
        .reset_index()
    )
    grid["CO_Rate"] = (grid["ChargeOffs"] / grid["N"]).round(4)
    return grid, cuts_m1, cuts_m2


# ──────────────────────────────────────────────────────────────────────────────
# 7. TIER ASSIGNMENT  —  exhaustive search, 1.5x weighted-average CO rule
# ──────────────────────────────────────────────────────────────────────────────
def _weighted_co(cells: pd.DataFrame) -> float:
    total_n = cells["N"].sum()
    return cells["ChargeOffs"].sum() / total_n if total_n > 0 else 0.0

def _find_best_split(cells, n_tiers, mult):
    nc = len(cells)
    best_bounds, best_min_mult = None, -1.0
    for splits in combinations(range(1, nc), n_tiers - 1):
        bounds = (0,) + splits + (nc,)
        rates  = [_weighted_co(cells.iloc[bounds[i]:bounds[i+1]]) for i in range(n_tiers)]
        if any(r == 0.0 for r in rates[:-1]):
            continue
        mults = [rates[i+1] / rates[i] for i in range(n_tiers - 1)]
        if all(m >= mult for m in mults):
            min_m = min(mults)
            if min_m > best_min_mult:
                best_min_mult, best_bounds = min_m, bounds
    return best_bounds

def assign_tiers(grid: pd.DataFrame, n_tiers: int = N_TIERS,
                 mult: float = MULT) -> tuple:
    # Sort cells by CO rate; preserve original index for merge-back
    cells = grid.dropna(subset=["CO_Rate"]).copy()
    cells["_orig_idx"] = cells.index
    cells = cells.sort_values("CO_Rate", ascending=True).reset_index(drop=True)

    bounds = _find_best_split(cells, n_tiers, mult)
    if bounds is None:
        print(f"  WARNING: No {mult}x partition found — falling back to equal-count split.")
        nc     = len(cells)
        chunk  = nc // n_tiers
        bounds = tuple([i * chunk for i in range(n_tiers)] + [nc])

    tier_col, summary_rows, co_cuts = np.empty(len(cells), dtype=object), [], []
    for i in range(n_tiers):
        grp  = cells.iloc[bounds[i]:bounds[i+1]]
        name = TIER_ORDER[i]
        tier_col[bounds[i]:bounds[i+1]] = name
        rate = _weighted_co(grp)
        summary_rows.append({
            "Risk_Tier":   name,
            "N_Cells":     len(grp),
            "Accounts":    int(grp["N"].sum()),
            "ChargeOffs":  int(grp["ChargeOffs"].sum()),
            "Avg_CO_Rate": round(rate, 4),
            "CO_Rate_Min": round(grp["CO_Rate"].min(), 4),
            "CO_Rate_Max": round(grp["CO_Rate"].max(), 4),
        })
        co_cuts.append(round(grp["CO_Rate"].max(), 4))   # upper CO boundary of tier

    cells["Risk_Tier"] = tier_col

    # Merge tier labels back onto original grid via preserved index
    tier_map            = cells.set_index("_orig_idx")["Risk_Tier"]
    grid_tiered         = grid.copy()
    grid_tiered["Risk_Tier"] = tier_map.reindex(grid_tiered.index).values

    # Build summary
    summary = pd.DataFrame(summary_rows)
    rates   = summary["Avg_CO_Rate"].values
    summary["Multiplier_vs_Prev"] = ["-"] + [
        f"{rates[i] / rates[i-1]:.2f}x" for i in range(1, n_tiers)
    ]
    summary["CO_Rate_Pct"]    = (summary["Avg_CO_Rate"] * 100).round(1).astype(str) + "%"
    total_co                  = summary["ChargeOffs"].sum()
    summary["Pct_of_All_COs"] = (
        (summary["ChargeOffs"] / total_co * 100).round(1).astype(str) + "%"
    )

    return grid_tiered, summary, co_cuts


# ──────────────────────────────────────────────────────────────────────────────
# 8. SCORE GRID CHART
#    Coloured 5×5 heatmap — each cell annotated with N / #CO / CO%
#    Axes labelled with actual score cut points
#    Right panel shows CO-rate cut points per risk tier
# ──────────────────────────────────────────────────────────────────────────────
def _score_range_label(tier_label: int, cuts: np.ndarray) -> str:
    """
    Map a tier label (1-5, where 1=safest/highest score) to its score range.
    cuts from pd.qcut are in ascending order: [min, q20, q40, q60, q80, max].
    Tier label 1 → top quintile → cuts[4]–cuts[5]
    Tier label 5 → bottom quintile → cuts[0]–cuts[1]
    """
    idx  = 5 - int(tier_label)       # tier 1 → idx 4, tier 5 → idx 0
    lo   = cuts[idx]
    hi   = cuts[idx + 1]
    lo_s = f"{int(lo)}" if lo != -np.inf else f"{int(cuts[1]-1)}"
    hi_s = f"{int(hi)}" if hi != np.inf  else f"{int(cuts[-2]+1)}"
    return f"{lo_s}–{hi_s}"


def plot_score_grid(grid_tiered: pd.DataFrame, m1: str, m2: str,
                    cuts_m1: np.ndarray, cuts_m2: np.ndarray,
                    summary: pd.DataFrame, co_cuts: list) -> None:

    m1c, m2c = f"{m1}_tier", f"{m2}_tier"

    # ── layout: grid on left, tier legend on right ────────────────────────────
    fig = plt.figure(figsize=(13, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[3, 1.1], wspace=0.35)
    ax  = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")

    tier_labels_int = list(range(1, N_TIERS + 1))   # tier 1=safest, N_TIERS=riskiest

    # ── draw cells ────────────────────────────────────────────────────────────
    for _, row in grid_tiered.iterrows():
        m1_t  = int(row[m1c])
        m2_t  = int(row[m2c])
        tier  = row.get("Risk_Tier")
        color = TIER_PALETTE.get(tier, "#cccccc")

        # grid coordinates: x = m2_tier-1, y = m1_tier-1 (both 0-based)
        x, y = m2_t - 1, m1_t - 1

        rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white",
                              linewidth=2, zorder=1)
        ax.add_patch(rect)

        # cell annotations
        n_acc = int(row["N"])
        n_co  = int(row["ChargeOffs"])
        co_r  = row["CO_Rate"]

        ax.text(x + 0.5, y + 0.72, f"N = {n_acc:,}",
                ha="center", va="center", fontsize=7.5, color="black", zorder=2)
        ax.text(x + 0.5, y + 0.48, f"CO = {n_co:,}",
                ha="center", va="center", fontsize=7.5, color="black", zorder=2)
        ax.text(x + 0.5, y + 0.24, f"{co_r:.1%}",
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="black", zorder=2)

    # ── axes: score cut points as tick labels ─────────────────────────────────
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

    x_labels = [_score_range_label(t, cuts_m2) for t in tier_labels_int]
    y_labels = [_score_range_label(t, cuts_m1) for t in tier_labels_int]

    ax.set_xticks([i + 0.5 for i in range(5)])
    ax.set_xticklabels(
        [f"Tier {t}\n({x_labels[i]})" for i, t in enumerate(tier_labels_int)],
        fontsize=8
    )
    ax.set_yticks([i + 0.5 for i in range(5)])
    ax.set_yticklabels(
        [f"Tier {t}\n({y_labels[i]})" for i, t in enumerate(tier_labels_int)],
        fontsize=8
    )

    ax.set_xlabel(f"{m2} Score  →  Tier 1 = Safest, Tier 5 = Riskiest", fontsize=10)
    ax.set_ylabel(f"{m1} Score  →  Tier 1 = Safest, Tier 5 = Riskiest", fontsize=10)
    ax.set_title(f"Score Grid: {m1} × {m2}\n(volume · charge-offs · CO%  |  coloured by risk tier)",
                 fontsize=11, fontweight="bold")

    # ── right panel: tier legend + CO cut points ───────────────────────────────
    ax2.set_title("Risk Tier Key\n& CO Rate Cut Points",
                  fontsize=10, fontweight="bold", pad=8)

    y_pos = 0.97
    for idx, tier in enumerate(TIER_ORDER):
        row_s = summary[summary["Risk_Tier"] == tier]
        if row_s.empty:
            continue
        row_s = row_s.iloc[0]

        # Coloured patch + tier name
        patch = mpatches.FancyBboxPatch(
            (0.02, y_pos - 0.07), 0.96, 0.07,
            boxstyle="round,pad=0.01",
            facecolor=TIER_PALETTE[tier], edgecolor="white", linewidth=1.5,
            transform=ax2.transAxes, zorder=2
        )
        ax2.add_patch(patch)
        ax2.text(0.50, y_pos - 0.035, tier,
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=9, fontweight="bold", color="black", zorder=3)

        # Stats below patch
        co_lo = f"{row_s['CO_Rate_Min']:.1%}"
        co_hi = f"{row_s['CO_Rate_Max']:.1%}"
        mult_str = row_s["Multiplier_vs_Prev"]
        ax2.text(0.50, y_pos - 0.115,
                 f"CO range: {co_lo} – {co_hi}",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=7.8, color="#333333")
        ax2.text(0.50, y_pos - 0.155,
                 f"Avg CO: {row_s['CO_Rate_Pct']}  |  {mult_str}",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=7.5, color="#555555")
        ax2.text(0.50, y_pos - 0.195,
                 f"{row_s['Accounts']:,} accts  ·  {row_s['Pct_of_All_COs']} of COs",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=7.5, color="#555555")

        y_pos -= 0.22

    # CO rate cut-point dividers between tiers (skip last)
    for idx in range(len(TIER_ORDER) - 1):
        cutoff = co_cuts[idx]
        ax2.text(0.50, 0.97 - 0.22 * (idx + 1) + 0.01,
                 f"── CO cut: {cutoff:.1%} ──",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=7, color="#888888", style="italic")

    plt.savefig("score_grid_chart.png", bbox_inches="tight")
    plt.close()
    print("    Saved: score_grid_chart.png")


# ──────────────────────────────────────────────────────────────────────────────
# 9. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main():
    sep = "=" * 70
    print(sep)
    print("  CREDIT RISK SCORE PIPELINE — v2")
    print(sep)

    # ── simulate ──────────────────────────────────────────────────────────────
    print(f"\n[1] Simulating {N:,} accounts (target CO ≈ 25%) …")
    df = simulate_dataset(N)
    print(f"    Actual Charge-Off Rate : {df['charge_off'].mean():.2%}")

    # ── Lorenz curves ──────────────────────────────────────────────────────────
    print("\n[2] Lorenz Curves — all models")
    plot_lorenz_curves(df)

    print("\n[3] Lorenz Curves — per bin of best individual model")
    plot_lorenz_by_best_model_bins(df)

    # ── pair selection: CMI + JMI ─────────────────────────────────────────────
    print(f"\n[4] Pair Selection — CMI & JMI ({N_FOLDS}-fold CV, selected by JMI)")
    print("-" * 70)
    pair_rows = [compute_pair_metrics_cv(df, m1, m2) for m1, m2 in combinations(MODELS, 2)]
    pair_df   = (pd.DataFrame(pair_rows)
                   .sort_values("JMI_mean", ascending=False)
                   .reset_index(drop=True))
    pair_df["JMI_Rank"] = range(1, len(pair_df) + 1)
    pair_df["CMI_Rank"] = pair_df["CMI_mean"].rank(ascending=False, method="first").astype(int)

    print(pair_df.to_string(index=False))
    pair_df.to_csv("pair_rankings.csv", index=False)
    plot_pair_table(pair_df)

    best    = pair_df.iloc[0]   # highest JMI
    best_m1, best_m2 = [m.strip() for m in best["Pair"].split("+")]
    print(f"\n  ★  Best pair : {best['Pair']}")
    print(f"     JMI mean   = {best['JMI_mean']:.6f} bits  (selection criterion — total combined information)")
    print(f"     CMI mean   = {best['CMI_mean']:.6f} bits  (complementarity, CMI rank #{best['CMI_Rank']})")

    # ── score grid ────────────────────────────────────────────────────────────
    print(f"\n[5] Building 5×5 Score Grid — {best['Pair']}")
    grid, cuts_m1, cuts_m2 = build_score_grid(df, best_m1, best_m2)
    grid.to_csv("score_grid_raw.csv", index=False)

    m1c, m2c = f"{best_m1}_tier", f"{best_m2}_tier"
    pivot_co  = grid.pivot_table(index=m1c, columns=m2c,
                                  values="CO_Rate", aggfunc="mean")
    pivot_co.index.name = f"{best_m1} ↓ / {best_m2} →"
    print("\nCharge-Off Rate per cell (Tier 1=Safest, Tier 5=Riskiest):")
    print(pivot_co.round(3).to_string())

    # ── tier assignment ───────────────────────────────────────────────────────
    print(f"\n[6] Tier Assignment — {MULT}x CO Rate Rule")
    print("-" * 70)
    grid_tiered, summary, co_cuts = assign_tiers(grid)
    grid_tiered.to_csv("score_grid_tiered.csv", index=False)
    summary.to_csv("risk_tier_summary.csv", index=False)

    display_cols = ["Risk_Tier", "N_Cells", "Accounts", "CO_Rate_Pct",
                    "CO_Rate_Min", "CO_Rate_Max", "Multiplier_vs_Prev",
                    "Pct_of_All_COs"]
    print("\nRisk Tier Summary:")
    print(summary[display_cols].to_string(index=False))

    print("\nMultiplier validation (must be ≥ 1.50x):")
    rates = summary["Avg_CO_Rate"].values
    for i in range(1, N_TIERS):
        ratio  = rates[i] / rates[i - 1]
        status = "✓" if ratio >= MULT else "✗  FAILS"
        print(f"  {summary['Risk_Tier'].iloc[i-1]:12s} → "
              f"{summary['Risk_Tier'].iloc[i]:12s} : "
              f"{rates[i-1]:.1%} → {rates[i]:.1%}  ({ratio:.2f}x)  {status}")

    # ── score grid chart ───────────────────────────────────────────────────────
    print(f"\n[7] Score Grid Chart")
    plot_score_grid(grid_tiered, best_m1, best_m2,
                    cuts_m1, cuts_m2, summary, co_cuts)

    # ── output summary ─────────────────────────────────────────────────────────
    print("\n[8] Files written:")
    for f in ["lorenz_curves.png", "lorenz_by_best_model_bins.png",
              "pair_table.png",
              "pair_rankings.csv", "score_grid_raw.csv",
              "score_grid_tiered.csv", "risk_tier_summary.csv",
              "score_grid_chart.png"]:
        print(f"    {f}")

    print("\n" + sep)
    print("  DONE")
    print(sep)

    return df, pair_df, grid_tiered, summary


if __name__ == "__main__":
    df, pair_df, grid_tiered, summary = main()
