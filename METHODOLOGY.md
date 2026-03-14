# Methodology — Credit Risk Score Simulation & Model Selection

## 1. Dataset Simulation

### Design
A single **latent risk factor** `Z ~ N(0,1)` drives both the charge-off outcome and all five model scores. This reflects the real-world structure where creditworthiness is an unobserved trait that models attempt to measure imperfectly.

```
Charge-off probability: p = sigmoid(1.5·Z − 1.5)
Charge-off outcome:     Y ~ Bernoulli(p)
Model score:            S = −signal·Z + noise,  scaled to [300, 850]
```

The negative sign on `signal` makes higher scores correspond to lower risk, consistent with bureau score conventions.

### Choice: CO Rate ≈ 25%
The intercept `−1.5` in the sigmoid was set to produce an overall charge-off rate of ~25%. This reflects a sub-prime or near-prime credit portfolio. The rate can be adjusted via the intercept parameter.

**Alternative considered:** The original simulation used `−0.5` giving ~42% CO, which is unrealistically high for most real portfolios.

### Choice: 5 Model Scores with Varying Signal
| Model | Signal | Noise | Design AUC | Character |
|---|---|---|---|---|
| Model_A | 2.5 | 0.6 | ~0.81 | Strong (e.g. internal bureau model) |
| Model_B | 2.2 | 0.8 | ~0.80 | Strong (e.g. external vendor model) |
| Model_C | 1.6 | 1.2 | ~0.75 | Moderate (e.g. behavioural score) |
| Model_D | 1.1 | 1.6 | ~0.67 | Weak (e.g. thin-file proxy) |
| Model_E | 0.4 | 2.2 | ~0.55 | Near-noise (e.g. experimental model) |

Because all scores are derived from the same `Z`, Model_A and Model_B are **highly correlated** — a key property the pair selection methodology is designed to detect.

### Choice: 5% Missingness Per Score
Random missingness is injected into each model score at 5% to simulate real-world data availability gaps (e.g. missing bureau pull, vendor downtime). This forces the pipeline to handle `dropna()` consistently.

**Alternatives:** Structured missingness (correlated with risk level), different rates per model, or no missingness for a clean simulation.

---

## 2. Individual Model Benchmarking — Lorenz Curves

### What It Measures
The Lorenz curve (Cumulative Accuracy Profile) plots:
- **X-axis:** Cumulative % of accounts, sorted from riskiest to safest (score ascending)
- **Y-axis:** Cumulative % of charge-offs captured

A perfect model captures all charge-offs in the first `bad_rate%` of accounts. A random model follows the diagonal.

### Gini Coefficient
The Gini in the legend is the **ROC-based Gini**:

```
Gini = 2 × ROC_AUC − 1
```

This is computed via `sklearn.metrics.roc_auc_score` on the negated scores (since high score = low risk, the negation aligns direction with "higher = more likely bad").

**Why not 2 × AUC_Lorenz − 1?**
The area under the CAP/Lorenz curve (`AUC_lorenz`) and the ROC AUC are different quantities at non-50% bad rates. For a perfect model with a 25% bad rate, `2 × AUC_lorenz − 1 = 0.75` while ROC Gini = 1.0. Using the CAP area underestimates Gini and makes models appear weaker than they are. The ROC-based formula is model-agnostic and the standard in credit risk.

**Alternative:** The Accuracy Ratio (AR) normalises the CAP area by the perfect model's area: `AR = (AUC_lorenz − 0.5) / (AUC_perfect − 0.5)`. AR and ROC Gini are closely related but not identical.

---

## 3. Pair Selection — Conditional Mutual Information (CMI)

### Why Not Logistic Regression Blending?
The v1 approach combined each pair with logistic regression and measured the blended AUC. This has three problems:
1. **Redundancy blind:** Two strong but correlated models produce a high blended AUC, appearing to be a good pair even though one adds little to the other.
2. **In-sample bias:** The LR is evaluated on the same data it was trained on, inflating AUC.
3. **Linear assumption:** LR only captures the linear portion of each model's incremental signal.

### CMI Definition

```
I(Y ; B | A) = H(Y | A) − H(Y | A, B)
```

- `H(Y | A)`: entropy of charge-off within each bin of Model A — how uncertain we still are about charge-off after knowing A's score
- `H(Y | A, B)`: entropy of charge-off within each (A-bin, B-bin) cell — residual uncertainty after knowing both scores
- The difference is the information B adds beyond A

**Symmetric version:** `CMI = [ I(Y;B|A) + I(Y;A|B) ] / 2` so neither model is treated as primary. This matters when the two models have different score distributions.

### Discretisation
Continuous scores are binned into **quintiles (5 bins)** before computing entropy. This is necessary because entropy is defined over discrete distributions.

**Choice: 5 bins (quintiles)**
- Fewer bins (e.g. 3) lose granularity; more bins (e.g. 10 deciles) require more data per cell to produce stable entropy estimates.
- With 50,000 accounts and 25 cells (5×5), each cell has ~2,000 accounts on average — sufficient for stable CO rate estimates with quintiles.
- For larger portfolios (500k+), deciles are appropriate.

### Cross-Validation (5-fold)
Bin boundaries are **fit on the training fold (80%)** and applied to the **test fold (20%)**. CMI is computed only on the test fold. This prevents in-sample inflation.

The `−inf / +inf` extension on bin edges ensures that test-fold scores outside the training range are always assigned to a valid bin rather than NaN.

**Reported metrics:**
- `CMI_mean`: average across 5 folds — primary ranking criterion
- `CMI_std`: fold-to-fold variability — stability check (low std = robust signal)
- `CMI_min`: worst fold — conservative lower bound for pair quality

### Result Interpretation
| Pair | CMI (bits) | Interpretation |
|---|---|---|
| Model_A + Model_E | 0.084 | E adds unique signal not in A |
| Model_A + Model_B | 0.016 | B is nearly redundant with A |

A and B are both driven by the same strong latent signal — they are highly correlated and one adds little beyond the other. E, despite being near-noise on its own, captures residual variance in Z that A does not, making it a genuinely complementary second model.

**Alternatives considered:**
- **Marginal AUC lift** (AUC of pair minus AUC of primary model alone): converges with CMI when a flexible combiner is used, but depends on the combiner choice and cannot detect non-linear complementarity with a linear combiner.
- **Score correlation penalty**: simple, penalises correlated pairs, but Pearson/Spearman correlation does not capture non-linear dependence.
- **Concordance/discordance analysis**: diagnostic tool — useful for explaining results to stakeholders after the pair is chosen.

---

## 4. Score Grid Construction

### Quintile Binning
Each model score is independently split into 5 equal-population quintiles. Labels run `[5, 4, 3, 2, 1]` from lowest to highest score bin, so:
- **Tier 1** = top quintile (highest scores) = safest accounts
- **Tier 5** = bottom quintile (lowest scores) = riskiest accounts

This produces a **5 × 5 grid of 25 cells**, each reporting:
- `N`: account count
- `ChargeOffs`: charge-off count
- `CO_Rate`: observed charge-off rate (ChargeOffs / N)

### Choice: 5 × 5 Grid
- Coarser grids (3×3 = 9 cells) are easier to communicate but lose risk differentiation within tiers.
- Finer grids (10×10 = 100 cells) provide more granularity but many cells become statistically unreliable with small portfolios.
- With 50k accounts and ~25% CO rate, each 5×5 cell averages ~2,000 accounts and ~500 charge-offs — sufficient for stable CO rate estimates.

**Rule of thumb:** Each cell should contain at least 30 charge-off events for a reliable CO rate estimate. At 25% CO rate, this requires at least 120 accounts per cell.

### Choice: Equal-Population (Quantile) Bins
The quintile cut points are determined by the **score distribution** of the portfolio, not by fixed score values.

**Pro:** Equal cell populations — no sparse cells.
**Con:** Bin boundaries shift when the score distribution changes (population drift). In production, consider fixing bin boundaries to specific score values after an initial calibration period.

**Alternatives:**
- **Fixed score cutpoints** (e.g. 300–450, 450–550, …): stable over time but may produce unequal populations.
- **WOE (Weight of Evidence) binning**: places bin boundaries where CO rate changes most sharply — maximally informative but prone to overfitting on small samples and requires recalibration when the distribution shifts.

---

## 5. Risk Tier Assignment — 1.5x Rule

### The Constraint
Each tier's **accounts-weighted average CO rate** must be at least 1.5× the tier below it:

```
Avg_CO(Tier k+1) ≥ 1.5 × Avg_CO(Tier k)
```

**Why accounts-weighted?** A tier with 10,000 accounts at 5% CO rate should dominate the tier's average over a 200-account cell at 6%. Unweighted averages can be skewed by small sparse cells.

**Why 1.5×?** This ensures each tier is operationally distinguishable — a pricing, limit, or decisioning action targeted at one tier has clearly different loss economics than the adjacent tier. The multiplier is a business parameter and can be raised (e.g. 2.0×) for sharper separation or lowered (e.g. 1.3×) if the portfolio CO range is compressed.

### Algorithm — Exhaustive Search
1. Sort all 25 cells by CO rate ascending (lowest risk first).
2. Enumerate all `C(24, 4) = 10,626` ways to place 4 dividers between the 25 sorted cells, creating 5 contiguous groups.
3. For each candidate partition, compute the accounts-weighted average CO rate per tier.
4. Retain only partitions where all 4 consecutive tier pairs satisfy the 1.5× constraint.
5. Among valid partitions, **select the one that maximises the minimum achieved multiplier** — the most conservative split, hardest to invalidate with small data shifts.

**Why exhaustive search?**
A greedy approach (satisfy one boundary at a time) finds *a* valid split but not necessarily the optimal one. The minimum-multiplier objective is not decomposable, so only the exhaustive search guarantees the global optimum. At 10,626 iterations over 25 rows, the runtime is negligible (<1 ms).

**Fallback:** If no valid 1.5× partition exists (can happen with a very compressed CO rate distribution), the algorithm falls back to an equal-count split and warns.

### Tier Naming and CO Rate Cut Points
The tier names (Low Risk → High Risk) are assigned in order from the lowest-CO-rate group. The CO rate cut points between tiers represent the **maximum CO rate of each tier's cells**, providing an interpretable boundary for risk classification.

| Transition | CO Rate Cut |
|---|---|
| Low Risk → Medium-Low | ≤ 3.0% |
| Medium-Low → Medium | ≤ 9.3% |
| Medium → Medium-High | ≤ 18.4% |
| Medium-High → High Risk | ≤ 33.8% |

**Alternative tier boundary approaches:**
- **Business-defined thresholds** (e.g. "High Risk = CO > 50%"): operationally intuitive, but may not be achievable with a given score pair.
- **Jenks natural breaks**: finds natural gaps in the CO rate distribution — good if the distribution is multimodal.
- **Decision tree on the 5×5 grid**: treats each cell as an observation and groups by minimising within-tier CO rate variance.

---

## 6. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| All models share one latent factor | Correlation structure is over-simplified | Real models draw from multiple risk dimensions (bureau, behavioural, income) |
| Random missingness | Real missingness is often correlated with risk (thin-file = higher risk) | Model missingness as a function of the latent factor |
| Bins fit and applied to same-period data | No concept of score drift | In production, fix bin boundaries after initial calibration; recalibrate annually |
| 1.5× rule applied to observed CO rates | Observed rates have sampling variance | Apply a confidence interval floor (lower bound of CI) for the constraint check |
| No monotonicity enforcement | Non-monotone cells in the grid are possible | Apply isotonic regression to smooth cell CO rates before tier assignment |
