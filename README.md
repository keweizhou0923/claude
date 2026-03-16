# Credit Risk Score Simulation & Model Selection

## Goal

Identify the best two-model combination for credit risk scoring by simulating a realistic credit portfolio, evaluating all candidate model pairs, and building a score grid with clearly separated risk tiers.

## Dataset

A synthetic portfolio of **50,000 accounts** with a **~25% charge-off rate**. Five model scores of varying predictive power (Model_A through Model_E, from strong to near-random) are simulated alongside a binary charge-off outcome. Each score has ~5% missingness to reflect real-world data quality.

## Approach

### Step 1 — Individual Model Benchmarking
Each model is assessed independently using its **Lorenz curve** and **Gini coefficient** (ROC-based: `2 × AUC − 1`). This establishes a standalone performance baseline but does not determine which pair to use together.

### Step 2 — Pair Selection: JMI ranked, CMI as reference

Both metrics are computed in a single 5-fold CV loop and reported side-by-side. **Pairs are sorted and selected by highest JMI.**

| Metric | Formula | Measures | Role |
|---|---|---|---|
| **JMI** *(selection)* | `H(Y) − H(Y\|A,B)` | Total combined information the pair carries about charge-off | **Primary ranking** |
| **CMI** *(diagnostic)* | `[I(Y;B\|A) + I(Y;A\|B)] / 2` | Complementarity — new information each model adds beyond the other | Governance reference |

**Why rank by JMI?** JMI measures the pair's absolute discrimination ceiling — how much of the charge-off signal the two models can explain together. Selecting the highest-JMI pair maximises total predictive power.

**Why show CMI rank alongside?** CMI reveals whether the pair's power comes from two *complementary* models or from one dominant model carrying the other. A pair with JMI rank #1 but high CMI rank signals that both models are strong but largely redundant — useful context for model governance and vendor diversification decisions.

**Cross-validation** (5-fold) ensures both metrics are estimated on held-out data — bin boundaries are fit on the training fold and applied to the test fold only.

**Winner: Model_A + Model_B** (JMI rank #1). Both are individually strong models that together explain the most charge-off signal. Their CMI rank is low — both are driven by the same strong latent factor, making them largely redundant — a consideration for governance and vendor diversification.

### Step 3 — Score Grid Construction
The best pair is used to build a **5 × 5 score grid**: each model score is independently split into five equal-population quintile tiers (Tier 1 = safest/highest score, Tier 5 = riskiest/lowest score), producing 25 cells. Each cell reports account volume, charge-off count, and charge-off rate.

### Step 4 — Risk Tier Assignment (1.5x Rule)
The 25 cells are sorted by charge-off rate and partitioned into **5 named risk tiers** via an exhaustive search over all possible split points. The winning partition maximises the minimum tier-over-tier multiplier, subject to every consecutive tier pair satisfying:

```
Avg CO Rate (Tier k+1) ≥ 1.5 × Avg CO Rate (Tier k)
```

CO rates are **accounts-weighted averages** across all cells in a tier.

| Risk Tier | Avg CO Rate | Multiplier vs Prior |
|---|---|---|
| Low Risk | ~3% | — |
| Medium-Low | ~6% | ≥1.5× |
| Medium | ~13% | ≥1.5× |
| Medium-High | ~27% | ≥1.5× |
| High Risk | ~55%+ | ≥1.5× |

*(Exact values vary by simulation seed; all runs satisfy the 1.5× constraint.)*

## Outputs

| File | Description |
|---|---|
| `lorenz_curves.png` | Lorenz curves for all 5 models with Gini coefficients |
| `pair_table.png` | Pair ranking table — CMI and JMI side-by-side, ranked by CMI |
| `score_grid_chart.png` | Coloured 5×5 grid with volume, CO count, CO%, tier colours, and CO rate cut points |
| `pair_rankings.csv` | Raw CMI and JMI scores for all pairs |
| `score_grid_raw.csv` | 25-cell grid with CO rates |
| `score_grid_tiered.csv` | Grid with risk tier labels assigned |
| `risk_tier_summary.csv` | Tier-level summary with CO rates, multipliers, and coverage |
