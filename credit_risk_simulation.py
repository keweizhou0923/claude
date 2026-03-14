"""
Credit Risk Score Pipeline — v2
================================
Changes from v1:
  - Pair selection: Conditional Mutual Information (CMI) + k-fold CV
    replaces logistic regression blending on training data
  - Tier assignment: exhaustive search for 1.5x weighted-average CO rate rule
    replaces hardcoded combined-score bins
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

N        = 50_000   # accounts
N_BINS   = 5        # quintile bins for CMI discretisation
N_FOLDS  = 5        # k-fold CV folds
MULT     = 1.5      # minimum tier-over-tier CO rate multiplier
N_TIERS  = 5        # number of risk tiers


# ──────────────────────────────────────────────────────────────────────────────
# 1. SIMULATE DATASET
# ──────────────────────────────────────────────────────────────────────────────
def simulate_dataset(n: int) -> pd.DataFrame:
    latent = np.random.normal(0, 1, n)
    prob   = 1 / (1 + np.exp(-(1.5 * latent - 0.5)))
    co     = np.random.binomial(1, prob, n)

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
    for col in ["Model_A", "Model_B", "Model_C", "Model_D", "Model_E"]:
        df.loc[np.random.rand(n) < 0.05, col] = np.nan
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. CONDITIONAL MUTUAL INFORMATION
#
# Theory recap:
#   I(Y ; B | A) = H(Y | A) - H(Y | A, B)
#
#   H(Y | A)    = Σ_a  p(a)   · H(Y | A=a)       [entropy of Y within each A-bin]
#   H(Y | A, B) = Σ_ab p(a,b) · H(Y | A=a, B=b)  [entropy of Y within each cell]
#
# We use the SYMMETRIC version:  CMI = [ I(Y;B|A) + I(Y;A|B) ] / 2
# so neither model is privileged as "primary".
#
# Bins are fit on the TRAINING fold and applied to the TEST fold
# to get an honest out-of-sample CMI estimate.
# ──────────────────────────────────────────────────────────────────────────────
def _h(p: float) -> float:
    """Binary entropy H(p) in bits. Returns 0 for p ∈ {0, 1}."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def _make_cuts(scores: np.ndarray, n_bins: int) -> np.ndarray:
    """Fit equal-population bin edges; extend to (-∞, +∞) for test-fold coverage."""
    _, cuts = pd.qcut(scores, q=n_bins, retbins=True, duplicates="drop")
    cuts[0], cuts[-1] = -np.inf, np.inf
    return cuts


def _apply_cuts(scores: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    return np.array(pd.cut(scores, bins=cuts, labels=False), dtype=float)


def _h_y_given_a(y: np.ndarray, a_bins: np.ndarray) -> float:
    """H(Y | A) — weighted average binary entropy within each A-bin."""
    n, h = len(y), 0.0
    for a_val in np.unique(a_bins[~np.isnan(a_bins)]):
        mask = a_bins == a_val
        h += mask.sum() / n * _h(y[mask].mean())
    return h


def _h_y_given_ab(y: np.ndarray, a_bins: np.ndarray, b_bins: np.ndarray) -> float:
    """H(Y | A, B) — weighted average binary entropy within each (A, B) cell."""
    n = len(y)
    df_t = pd.DataFrame({"y": y, "a": a_bins, "b": b_bins}).dropna()
    h = 0.0
    for (_, grp) in df_t.groupby(["a", "b"]):
        h += len(grp) / n * _h(grp["y"].mean())
    return h


def _cmi_one_direction(y, primary_bins, secondary_bins):
    """I(Y ; secondary | primary) = H(Y|primary) - H(Y|primary, secondary)."""
    return max(0.0, _h_y_given_a(y, primary_bins) - _h_y_given_ab(y, primary_bins, secondary_bins))


def compute_cmi_cv(df: pd.DataFrame, m1: str, m2: str) -> dict:
    """
    5-fold CV CMI.

    For each fold:
      1. Fit quintile cuts on the training split (80 % of data).
      2. Apply those cuts to the held-out test split (20 %).
      3. Compute symmetric CMI on the test split only.

    Reporting:
      CMI_mean  — average across folds (primary ranking metric)
      CMI_std   — fold-to-fold variability (stability check)
      CMI_min   — worst fold (conservative lower bound)
    """
    sub = df[[m1, m2, "charge_off"]].dropna().reset_index(drop=True)
    y, sa, sb = sub["charge_off"].values, sub[m1].values, sub[m2].values

    kf   = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    vals = []

    for tr_idx, te_idx in kf.split(y):
        # Fit bins on training fold
        cuts_a = _make_cuts(sa[tr_idx], N_BINS)
        cuts_b = _make_cuts(sb[tr_idx], N_BINS)

        # Apply to test fold
        a_bins = _apply_cuts(sa[te_idx], cuts_a)
        b_bins = _apply_cuts(sb[te_idx], cuts_b)
        y_te   = y[te_idx]

        # Symmetric: average both directions
        cmi = (_cmi_one_direction(y_te, a_bins, b_bins) +
               _cmi_one_direction(y_te, b_bins, a_bins)) / 2
        vals.append(cmi)

    return {
        "Pair":     f"{m1} + {m2}",
        "N_valid":  len(sub),
        "CMI_mean": round(np.mean(vals), 6),
        "CMI_std":  round(np.std(vals),  6),
        "CMI_min":  round(np.min(vals),  6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. SCORE GRID — 5 × 5 QUINTILE BINS
# ──────────────────────────────────────────────────────────────────────────────
def build_score_grid(df: pd.DataFrame, m1: str, m2: str, n_buckets: int = 5) -> pd.DataFrame:
    sub    = df[[m1, m2, "charge_off"]].dropna().copy()
    labels = list(range(n_buckets, 0, -1))   # 5 = safest, 1 = riskiest
    sub[f"{m1}_tier"] = pd.qcut(sub[m1], q=n_buckets, labels=labels, duplicates="drop")
    sub[f"{m2}_tier"] = pd.qcut(sub[m2], q=n_buckets, labels=labels, duplicates="drop")
    grid = (
        sub.groupby([f"{m1}_tier", f"{m2}_tier"], observed=True)
        .agg(N=("charge_off", "count"), ChargeOffs=("charge_off", "sum"))
        .reset_index()
    )
    grid["CO_Rate"] = (grid["ChargeOffs"] / grid["N"]).round(4)
    return grid


# ──────────────────────────────────────────────────────────────────────────────
# 4. TIER ASSIGNMENT — 1.5x WEIGHTED-AVERAGE CO RATE RULE
#
# Algorithm:
#   1. Sort all 25 cells by CO rate ascending (lowest risk first).
#   2. Enumerate every way to split the sorted list into N_TIERS contiguous
#      blocks — C(n_cells - 1, N_TIERS - 1) = C(24, 4) = 10,626 combinations.
#   3. For each candidate partition compute the weighted-average CO rate
#      per tier (accounts-weighted).
#   4. Keep only partitions where every consecutive tier pair satisfies
#      avg_CO(Tier k+1) >= MULT × avg_CO(Tier k).
#   5. Among valid partitions, pick the one that maximises the minimum
#      achieved multiplier (most conservative / most separated).
# ──────────────────────────────────────────────────────────────────────────────
def _weighted_co(cells: pd.DataFrame) -> float:
    return cells["ChargeOffs"].sum() / cells["N"].sum()


def _find_best_split(cells: pd.DataFrame, n_tiers: int, mult: float):
    """
    Exhaustive search. Returns boundary list [0, s1, s2, s3, s4, n_cells]
    or None if no valid partition exists.
    """
    nc = len(cells)
    best_bounds, best_min_mult = None, -1.0

    for splits in combinations(range(1, nc), n_tiers - 1):
        bounds = (0,) + splits + (nc,)
        rates  = [_weighted_co(cells.iloc[bounds[i]:bounds[i + 1]])
                  for i in range(n_tiers)]

        # Skip if any non-last tier has zero CO rate (can't compute ratio)
        if any(r == 0.0 for r in rates[:-1]):
            continue

        mults = [rates[i + 1] / rates[i] for i in range(n_tiers - 1)]
        if all(m >= mult for m in mults):
            min_m = min(mults)
            if min_m > best_min_mult:
                best_min_mult = min_m
                best_bounds   = bounds

    return best_bounds


def assign_tiers(grid: pd.DataFrame, n_tiers: int = N_TIERS,
                 mult: float = MULT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign each of the 25 score-grid cells to a named risk tier
    using the 1.5x weighted-average CO rate rule.

    Returns
    -------
    grid_tiered : original grid with 'Risk_Tier' column added
    summary     : one row per tier with CO rate, multiplier, coverage
    """
    # Drop cells with no accounts or NaN CO rate before partitioning
    cells = (grid.dropna(subset=["CO_Rate"])
                 .sort_values("CO_Rate", ascending=True)
                 .reset_index(drop=True))

    bounds = _find_best_split(cells, n_tiers, mult)

    if bounds is None:
        print(f"  WARNING: No partition satisfies {mult}x rule with {n_tiers} tiers.")
        print(f"           Falling back to equal-count split.")
        nc     = len(cells)
        chunk  = nc // n_tiers
        bounds = tuple([i * chunk for i in range(n_tiers)] + [nc])

    tier_names = ["Low Risk", "Medium-Low", "Medium", "Medium-High", "High Risk"]

    # Assign tier label to each cell
    tier_col = np.empty(len(cells), dtype=object)
    summary_rows = []
    for i in range(n_tiers):
        grp  = cells.iloc[bounds[i]:bounds[i + 1]]
        name = tier_names[i]
        tier_col[bounds[i]:bounds[i + 1]] = name
        summary_rows.append({
            "Risk_Tier":   name,
            "N_Cells":     len(grp),
            "Accounts":    int(grp["N"].sum()),
            "ChargeOffs":  int(grp["ChargeOffs"].sum()),
            "Avg_CO_Rate": round(_weighted_co(grp), 4),
        })

    cells = cells.copy()
    cells["Risk_Tier"] = tier_col

    # Build summary with multipliers and coverage
    summary = pd.DataFrame(summary_rows)
    rates   = summary["Avg_CO_Rate"].values
    summary["Multiplier_vs_Prev"] = ["-"] + [
        f"{rates[i] / rates[i - 1]:.2f}x" for i in range(1, n_tiers)
    ]
    summary["CO_Rate_Pct"]    = (summary["Avg_CO_Rate"] * 100).round(1).astype(str) + "%"
    total_co                  = summary["ChargeOffs"].sum()
    summary["Pct_of_All_COs"] = (
        (summary["ChargeOffs"] / total_co * 100).round(1).astype(str) + "%"
    )

    # Merge tier label back onto the original grid (preserving all 25 rows)
    tier_map   = cells[["CO_Rate", "Risk_Tier"]].drop_duplicates("CO_Rate")
    grid_tiered = grid.merge(tier_map, on="CO_Rate", how="left")

    return grid_tiered, summary


# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main():
    sep = "=" * 70
    print(sep)
    print("  CREDIT RISK SCORE PIPELINE — v2")
    print("  Pair Selection : CMI + Cross-Validation")
    print("  Tier Assignment: 1.5x Weighted-Average CO Rate Rule")
    print(sep)

    # ── Simulate ──────────────────────────────────────────────────────────────
    print(f"\n[1] Simulating {N:,} accounts …")
    df = simulate_dataset(N)
    print(f"    Overall Charge-Off Rate : {df['charge_off'].mean():.2%}")

    # ── CMI pair ranking ──────────────────────────────────────────────────────
    print(f"\n[2] CMI Pair Selection ({N_FOLDS}-fold CV, {N_BINS} quintile bins per score)")
    print(f"    Symmetric CMI = [ I(Y;B|A) + I(Y;A|B) ] / 2  (units: bits)")
    print("-" * 70)

    models   = ["Model_A", "Model_B", "Model_C", "Model_D", "Model_E"]
    cmi_rows = [compute_cmi_cv(df, m1, m2) for m1, m2 in combinations(models, 2)]
    cmi_df   = (pd.DataFrame(cmi_rows)
                  .sort_values("CMI_mean", ascending=False)
                  .reset_index(drop=True))

    print(cmi_df.to_string(index=False))
    cmi_df.to_csv("pair_cmi_rankings.csv", index=False)

    best    = cmi_df.iloc[0]
    best_m1, best_m2 = [m.strip() for m in best["Pair"].split("+")]
    print(f"\n  ★  Best pair : {best['Pair']}")
    print(f"     CMI mean   = {best['CMI_mean']:.6f} bits")
    print(f"     CMI min    = {best['CMI_min']:.6f} bits  (worst fold — conservative bound)")
    print(f"     CMI std    = {best['CMI_std']:.6f} bits  (fold stability)")

    # ── Score grid ────────────────────────────────────────────────────────────
    print(f"\n[3] 5×5 Score Grid  —  {best['Pair']}")
    print("    (Tier 5 = Safest, Tier 1 = Riskiest)")
    print("-" * 70)
    grid = build_score_grid(df, best_m1, best_m2)
    grid.to_csv("score_grid_raw.csv", index=False)

    m1c, m2c = f"{best_m1}_tier", f"{best_m2}_tier"
    pivot_co  = grid.pivot_table(index=m1c, columns=m2c, values="CO_Rate", aggfunc="mean")
    pivot_co.index.name = f"{best_m1} ↓ / {best_m2} →"
    print("\nCharge-Off Rate per cell:")
    print(pivot_co.round(3).to_string())

    # ── 1.5x tier assignment ──────────────────────────────────────────────────
    print(f"\n[4] Tier Assignment  —  {MULT}x Weighted-Average CO Rate Rule")
    print(f"    Exhaustive search over C(n_cells-1, {N_TIERS}-1) split points")
    print("-" * 70)
    grid_tiered, summary = assign_tiers(grid)
    grid_tiered.to_csv("score_grid_tiered.csv", index=False)
    summary.to_csv("risk_tier_summary.csv", index=False)

    # Tier summary table
    print("\nRisk Tier Summary:")
    display_cols = ["Risk_Tier", "N_Cells", "Accounts", "CO_Rate_Pct",
                    "Multiplier_vs_Prev", "Pct_of_All_COs"]
    print(summary[display_cols].to_string(index=False))

    # Multiplier validation
    print("\nMultiplier validation (must be ≥ 1.50x):")
    rates = summary["Avg_CO_Rate"].values
    for i in range(1, N_TIERS):
        ratio  = rates[i] / rates[i - 1]
        status = "✓" if ratio >= MULT else "✗  <-- FAILS constraint"
        print(f"  {summary['Risk_Tier'].iloc[i-1]:12s} → {summary['Risk_Tier'].iloc[i]:12s} : "
              f"{rates[i-1]:.1%} → {rates[i]:.1%}   ({ratio:.2f}x)  {status}")

    # Tier map on the grid
    if "Risk_Tier" in grid_tiered.columns:
        pivot_tier = grid_tiered.pivot_table(
            index=m1c, columns=m2c, values="Risk_Tier", aggfunc="first"
        )
        pivot_tier.index.name = f"{best_m1} ↓ / {best_m2} →"
        print(f"\nRisk Tier per cell:")
        print(pivot_tier.to_string())

    # ── Output files ──────────────────────────────────────────────────────────
    print("\n[5] Files written:")
    for f in ["pair_cmi_rankings.csv", "score_grid_raw.csv",
              "score_grid_tiered.csv", "risk_tier_summary.csv"]:
        print(f"    {f}")

    print("\n" + sep)
    print("  DONE")
    print(sep)

    return df, cmi_df, grid_tiered, summary


if __name__ == "__main__":
    df, cmi_df, grid_tiered, summary = main()
