"""
Credit Risk Score Simulation & Model Comparison
================================================
Simulates a credit portfolio dataset with:
  - Binary charge-off label (target)
  - 5 model scores (Model_A … Model_E) with varying predictive power

Then ranks every single model and every pair of models by:
  - AUC-ROC
  - KS Statistic (Kolmogorov-Smirnov)
  - Gini Coefficient
  - Top-decile Capture Rate (% of charge-offs caught in riskiest 10%)

The best two-model pair is used to build a combined score grid
showing Charge-Off Rate by Credit Risk Tier.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. SIMULATE DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
N = 50_000  # accounts

def simulate_dataset(n: int) -> pd.DataFrame:
    """
    Simulate a credit portfolio with a charge-off flag and 5 model scores.

    Scores are calibrated to reflect realistic predictive power differences:
      Model_A – Strong (AUC ~0.78)   internal bureau model
      Model_B – Strong (AUC ~0.75)   external vendor model
      Model_C – Moderate (AUC ~0.68) behavioural score
      Model_D – Weak    (AUC ~0.62)  thin-file proxy score
      Model_E – Noise   (AUC ~0.53)  experimental model
    """
    # Latent credit risk factor (higher → worse risk)
    latent_risk = np.random.normal(0, 1, n)

    # Charge-off probability driven by latent risk
    charge_off_prob = 1 / (1 + np.exp(-(1.5 * latent_risk - 0.5)))
    charge_off = np.random.binomial(1, charge_off_prob, n)

    # Model scores: scaled 300-850 (like a bureau score, higher = lower risk)
    def make_score(latent, signal_strength, noise_sd, score_min=300, score_max=850):
        raw = -signal_strength * latent + np.random.normal(0, noise_sd, n)
        # Normalise to [score_min, score_max]
        raw_norm = (raw - raw.min()) / (raw.max() - raw.min())
        return np.round(score_min + raw_norm * (score_max - score_min)).astype(int)

    df = pd.DataFrame({
        "account_id": np.arange(1, n + 1),
        "charge_off":  charge_off,
        # Higher signal → lower noise → stronger model
        "Model_A": make_score(latent_risk, signal_strength=2.5, noise_sd=0.6),
        "Model_B": make_score(latent_risk, signal_strength=2.2, noise_sd=0.8),
        "Model_C": make_score(latent_risk, signal_strength=1.6, noise_sd=1.2),
        "Model_D": make_score(latent_risk, signal_strength=1.1, noise_sd=1.6),
        "Model_E": make_score(latent_risk, signal_strength=0.4, noise_sd=2.2),
    })

    # Introduce ~5 % missing values per score to mimic real data
    for col in ["Model_A", "Model_B", "Model_C", "Model_D", "Model_E"]:
        mask = np.random.rand(n) < 0.05
        df.loc[mask, col] = np.nan

    return df


# ─────────────────────────────────────────────
# 2. EVALUATION METRICS
# ─────────────────────────────────────────────
def compute_ks(y_true, y_score) -> float:
    """Kolmogorov-Smirnov statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def compute_gini(auc: float) -> float:
    """Gini = 2 * AUC - 1."""
    return 2 * auc - 1


def top_decile_capture(y_true, y_score) -> float:
    """Fraction of charge-offs captured in the riskiest 10% of accounts."""
    df_tmp = pd.DataFrame({"y": y_true, "s": y_score})
    threshold = df_tmp["s"].quantile(0.10)   # bottom 10% of score = highest risk
    top_decile = df_tmp[df_tmp["s"] <= threshold]
    return top_decile["y"].sum() / max(y_true.sum(), 1)


def evaluate_single_model(y_true, y_score, name: str) -> dict:
    valid = ~np.isnan(y_score)
    yt, ys = y_true[valid], y_score[valid]
    # Invert score: low score = high risk → use negative for AUC
    neg_score = -ys
    auc  = roc_auc_score(yt, neg_score)
    ks   = compute_ks(yt, neg_score)
    gini = compute_gini(auc)
    tdc  = top_decile_capture(yt, ys)
    return {
        "Model": name,
        "N_valid": int(valid.sum()),
        "ChargeOff_Rate": round(yt.mean(), 4),
        "AUC":  round(auc,  4),
        "KS":   round(ks,   4),
        "Gini": round(gini, 4),
        "Top10pct_Capture": round(tdc, 4),
    }


def evaluate_combined_model(df: pd.DataFrame, m1: str, m2: str) -> dict:
    """
    Combine two models via logistic regression on available rows,
    then evaluate the blended score.
    """
    sub = df[[m1, m2, "charge_off"]].dropna()
    X = sub[[m1, m2]].values
    y = sub["charge_off"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=500)
    lr.fit(X_scaled, y)
    proba = lr.predict_proba(X_scaled)[:, 1]   # P(charge-off)

    auc  = roc_auc_score(y, proba)
    ks   = compute_ks(y, proba)
    gini = compute_gini(auc)
    tdc  = top_decile_capture(y, proba)

    return {
        "Model": f"{m1} + {m2}",
        "N_valid": len(sub),
        "ChargeOff_Rate": round(y.mean(), 4),
        "AUC":  round(auc,  4),
        "KS":   round(ks,   4),
        "Gini": round(gini, 4),
        "Top10pct_Capture": round(tdc, 4),
        "LR_coef_m1": round(lr.coef_[0][0], 4),
        "LR_coef_m2": round(lr.coef_[0][1], 4),
    }


# ─────────────────────────────────────────────
# 3. SCORE GRID BUILDER
# ─────────────────────────────────────────────
def build_score_grid(df: pd.DataFrame, m1: str, m2: str,
                     n_buckets: int = 5) -> pd.DataFrame:
    """
    Build a 2-D credit-risk tier grid using quintile buckets of m1 and m2.
    Each cell shows: account count, charge-off count, charge-off rate.
    """
    sub = df[[m1, m2, "charge_off"]].dropna().copy()

    # Quintile buckets (1 = riskiest, n_buckets = safest)
    labels = list(range(n_buckets, 0, -1))   # 5,4,3,2,1
    sub[f"{m1}_tier"] = pd.qcut(sub[m1], q=n_buckets, labels=labels, duplicates="drop")
    sub[f"{m2}_tier"] = pd.qcut(sub[m2], q=n_buckets, labels=labels, duplicates="drop")

    grid = (
        sub.groupby([f"{m1}_tier", f"{m2}_tier"], observed=True)
        .agg(
            N=("charge_off", "count"),
            ChargeOffs=("charge_off", "sum"),
        )
        .reset_index()
    )
    grid["CO_Rate"] = (grid["ChargeOffs"] / grid["N"]).round(4)

    # Assign a combined risk tier label
    grid["Combined_Score"] = (
        grid[f"{m1}_tier"].astype(int) + grid[f"{m2}_tier"].astype(int)
    )
    grid["Risk_Tier"] = pd.cut(
        grid["Combined_Score"],
        bins=[1, 3, 5, 7, 9, 10],
        labels=["High Risk", "Medium-High", "Medium", "Medium-Low", "Low Risk"],
        include_lowest=True,
    )
    return grid.sort_values([f"{m1}_tier", f"{m2}_tier"])


def summarise_risk_tiers(grid: pd.DataFrame) -> pd.DataFrame:
    """Roll up score grid to named risk tiers."""
    summary = (
        grid.groupby("Risk_Tier", observed=True)
        .agg(
            Accounts=("N", "sum"),
            ChargeOffs=("ChargeOffs", "sum"),
        )
        .reset_index()
    )
    summary["CO_Rate"] = (summary["ChargeOffs"] / summary["Accounts"]).round(4)
    summary["CO_Rate_Pct"] = (summary["CO_Rate"] * 100).round(2).astype(str) + "%"
    total_co = summary["ChargeOffs"].sum()
    summary["Pct_of_All_COs"] = (
        (summary["ChargeOffs"] / total_co * 100).round(1).astype(str) + "%"
    )
    return summary


# ─────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  CREDIT RISK SCORE SIMULATION & MODEL COMPARISON")
    print("=" * 70)

    # ── Simulate ──────────────────────────────
    print(f"\n[1] Simulating {N:,} accounts …")
    df = simulate_dataset(N)
    print(f"    Overall Charge-Off Rate : {df['charge_off'].mean():.2%}")
    print(f"    Missing values per score:")
    for col in ["Model_A", "Model_B", "Model_C", "Model_D", "Model_E"]:
        print(f"      {col}: {df[col].isna().sum():,} ({df[col].isna().mean():.1%})")

    # ── Single-model metrics ───────────────────
    print("\n[2] Single-Model Performance")
    print("-" * 70)
    models = ["Model_A", "Model_B", "Model_C", "Model_D", "Model_E"]
    single_results = []
    for m in models:
        res = evaluate_single_model(df["charge_off"].values, df[m].values, m)
        single_results.append(res)

    single_df = pd.DataFrame(single_results).set_index("Model")
    print(single_df[["AUC", "KS", "Gini", "Top10pct_Capture"]].to_string())
    single_df.to_csv("single_model_metrics.csv")

    # ── Pair model metrics ─────────────────────
    print("\n[3] All Two-Model Combination Performance")
    print("-" * 70)
    pair_results = []
    for m1, m2 in combinations(models, 2):
        res = evaluate_combined_model(df, m1, m2)
        pair_results.append(res)

    pair_df = pd.DataFrame(pair_results).set_index("Model")
    print(pair_df[["AUC", "KS", "Gini", "Top10pct_Capture"]].to_string())
    pair_df.to_csv("pair_model_metrics.csv")

    # ── Rank & select best pair ────────────────
    print("\n[4] Ranking & Selecting Best Two-Model Combination")
    print("-" * 70)
    # Composite rank: average rank across all four metrics
    ranked = pair_df[["AUC", "KS", "Gini", "Top10pct_Capture"]].copy()
    for col in ranked.columns:
        ranked[f"rank_{col}"] = ranked[col].rank(ascending=False)
    ranked["Composite_Rank"] = ranked[[c for c in ranked.columns if c.startswith("rank_")]].mean(axis=1)
    ranked = ranked.sort_values("Composite_Rank")

    print("\nPair Rankings (lower composite rank = better):")
    print(ranked[["AUC", "KS", "Gini", "Top10pct_Capture", "Composite_Rank"]].to_string())
    ranked.to_csv("pair_model_rankings.csv")

    best_pair_name = ranked.index[0]
    best_m1, best_m2 = [m.strip() for m in best_pair_name.split("+")]
    best_metrics = pair_df.loc[best_pair_name]

    print(f"\n  ★  Best pair: {best_pair_name}")
    print(f"     AUC={best_metrics['AUC']:.4f}  KS={best_metrics['KS']:.4f}"
          f"  Gini={best_metrics['Gini']:.4f}  Top-Decile={best_metrics['Top10pct_Capture']:.4f}")

    # ── Score grid for best pair ───────────────
    print(f"\n[5] Score Grid — {best_pair_name}")
    print("-" * 70)
    grid = build_score_grid(df, best_m1, best_m2, n_buckets=5)
    grid.to_csv("score_grid_detail.csv", index=False)

    tier_summary = summarise_risk_tiers(grid)
    print("\nCredit Risk Tier Summary:")
    print(tier_summary.to_string(index=False))
    tier_summary.to_csv("risk_tier_summary.csv", index=False)

    # Pretty pivot for presentation
    print(f"\nCharge-Off Rate Grid ({best_m1} Tier × {best_m2} Tier):")
    print("  (Tier 5 = Safest, Tier 1 = Riskiest)")
    pivot = grid.pivot_table(
        index=f"{best_m1}_tier",
        columns=f"{best_m2}_tier",
        values="CO_Rate",
        aggfunc="mean",
    )
    pivot.index.name   = f"{best_m1} ↓ / {best_m2} →"
    print(pivot.round(3).to_string())

    print("\n[6] Output files written:")
    for f in ["single_model_metrics.csv", "pair_model_metrics.csv",
              "pair_model_rankings.csv", "score_grid_detail.csv",
              "risk_tier_summary.csv"]:
        print(f"    {f}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    return df, ranked, grid, tier_summary


if __name__ == "__main__":
    df, ranked, grid, tier_summary = main()
