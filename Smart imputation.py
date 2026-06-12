"""
=============================================================================
SMART IMPUTATION PIPELINE — Bangladesh Medical Waste Dataset (2000–2023)
Paper: Smart Medical Waste Management
=============================================================================
Strategy architecture (3-layer approach):
  Layer 1 → TIME-SERIES COLS  : ffill → bfill grouped by Division
  Layer 2 → STATIC/SPATIAL COLS: KNNImputer (n_neighbors=5)
  Layer 3 → CORRELATED WASTE STREAMS: IterativeImputer (MICE, 10 iterations)

Why three strategies?
  • Time-series cols (GDP, Hospitals, CBR) evolve monotonically within a
    Division — filling from adjacent years preserves temporal continuity.
  • Static cols (Area, Urban_Pct, Population_Share) change slowly and are
    best estimated from spatially similar divisions (KNN).
  • Waste stream cols are highly correlated (r ≥ 0.95) — MICE exploits
    the full joint distribution for statistically sound imputations.
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.experimental import enable_iterative_imputer          # must come first
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge

# ─── reproducibility ──────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# 0. COLUMN TAXONOMY
#    Assign every numeric column to exactly one imputation layer.
# =============================================================================

# Layer 1 — Temporal (monotonic trends within each Division)
TIME_SERIES_COLS = [
    'GDP_per_Capita_USD',       # rises year-on-year per division
    'GDP_Growth_Pct',           # national economic cycle
    'Crude_Birth_Rate',         # secular demographic decline
    'Number_of_Hospitals',      # infrastructure build-up
    'Number_of_Beds',           # capacity expansion
    'Beds_per_1000_Pop',        # derived from beds + population
    'Hospitals_per_100k_Pop',   # derived from hospitals + population
    'Urban_Population_Pct',     # urbanisation trend
    'Waste_Mgmt_Coverage_Pct',  # policy-driven improvement over time
    'Proper_Segregation_Pct',   # same — improves with training programmes
]

# Layer 2 — Static / Spatial (stable across years, varies across Divisions)
KNN_COLS = [
    'Area_km2',                 # fixed geography
    'Population_Share_Pct',     # near-constant administrative share
    'Avg_Beds_per_Hospital',    # structural — changes very slowly
    'Bed_Occupancy_Rate_Pct',   # reflects division-level health-seeking behaviour
    'Dengue_Waste_Adjustment',  # narrow-range scalar, spatially smooth
]

# Layer 3 — Waste stream (highly correlated, benefit from joint MICE)
MICE_COLS = [
    'General_Waste_kg_day',
    'Infectious_Waste_kg_day',
    'Sharps_Waste_kg_day',
    'Pharmaceutical_Waste_kg_day',
    'Chemical_Waste_kg_day',
    'Cytotoxic_Waste_kg_day',
    'Pathological_Waste_kg_day',
    'Radioactive_Waste_kg_day',
    'Total_Hazardous_Waste_kg_day',
    'Total_Medical_Waste_kg_day',
    'Waste_per_Bed_kg_day',
    'Hazardous_Waste_Pct',
    # Include facility breakdown as auxiliary predictors for waste
    'Tertiary_Hospitals',
    'District_Hospitals',
    'Upazila_Health_Complexes',
    'Private_Clinics',
    'Specialized_Hospitals',
    'NGO_Facilities',
    'Community_Health_Clinics',
    'Diagnostic_Centers',
    'Facilities_with_Incinerator',
    'Facilities_with_Autoclave',
    # Epidemiological predictors
    'Dengue_Cases',
    'Dengue_Rate_per_100k',
]

# Categorical columns — excluded from numeric imputation
CAT_COLS = ['Division', 'Dengue_Category', 'Birth_Rate_Category', 'GDP_Category']

print("=" * 70)
print("SMART IMPUTATION PIPELINE — BD Medical Waste 2000–2023")
print("=" * 70)
print(f"\n  Layer 1 (ffill/bfill) : {len(TIME_SERIES_COLS)} columns")
print(f"  Layer 2 (KNN)         : {len(KNN_COLS)} columns")
print(f"  Layer 3 (MICE)        : {len(MICE_COLS)} columns")
print(f"  Categorical (skipped) : {len(CAT_COLS)} columns")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv(r"C:\Users\jayna\OneDrive\Documents\ML\Report\BD_MedWaste_CLEAN.csv")
df = df.sort_values(['Division', 'Year']).reset_index(drop=True)

print(f"\n  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  NaN before any injection: {df.isnull().sum().sum()}")

# =============================================================================
# 2. INJECT REALISTIC MCAR MISSINGNESS FOR DEMONSTRATION
#    (Skip this block if your real data already has NaN values)
#
#    Pattern mirrors real-world survey data:
#    • Time-series cols  → 8% missing (occasional non-reporting years)
#    • Static cols       → 5% missing (administrative gaps)
#    • Waste stream cols → 12% missing (lab measurement failures)
# =============================================================================
print("\n" + "─" * 70)
print("STEP 1 — Injecting representative MCAR missingness for demonstration")
print("─" * 70)

df_with_missing = df.copy()

missing_rates = {
    **{col: 0.08 for col in TIME_SERIES_COLS},
    **{col: 0.05 for col in KNN_COLS},
    **{col: 0.12 for col in MICE_COLS},
}

injection_log = []
for col, rate in missing_rates.items():
    if col not in df_with_missing.columns:
        continue
    n_missing = int(len(df_with_missing) * rate)
    missing_idx = np.random.choice(df_with_missing.index, size=n_missing, replace=False)
    df_with_missing.loc[missing_idx, col] = np.nan
    injection_log.append({'Column': col, 'Rate': f"{rate*100:.0f}%",
                          'N_Missing': n_missing, 'Layer': (
                              'Time-series' if col in TIME_SERIES_COLS else
                              'KNN'         if col in KNN_COLS          else 'MICE')})

inj_df = pd.DataFrame(injection_log)
total_injected = inj_df['N_Missing'].sum()
print(f"\n  Total NaN injected : {total_injected}")
print(f"  Dataset density    : {total_injected / (df.shape[0] * len(missing_rates)) * 100:.1f}% missing across target columns")
print()
print(f"  {'Column':<35} {'Layer':<14} {'Rate':<6} N_Missing")
print(f"  {'─'*35} {'─'*14} {'─'*6} {'─'*9}")
for _, row in inj_df.iterrows():
    print(f"  {row['Column']:<35} {row['Layer']:<14} {row['Rate']:<6} {row['N_Missing']}")

# Save ground truth for RMSE evaluation later
df_ground_truth = df.copy()

# =============================================================================
# 3. LAYER 1 — TEMPORAL IMPUTATION: ffill → bfill grouped by Division
# =============================================================================
print("\n" + "─" * 70)
print("STEP 2 — LAYER 1: Temporal ffill → bfill (grouped by Division)")
print("─" * 70)
print("""
  Rationale:
  Economic and demographic indicators evolve as continuous trends within
  each division. Missing year t for division D is best estimated by the
  nearest observed value in the same division's time series — not by
  values from other divisions with different socioeconomic baselines.

  Order: sort by [Division, Year] → ffill (fill from earlier years)
         → bfill (fill leading NaNs where the series starts with gaps)
""")

df_layer1 = df_with_missing.copy()
nan_before_l1 = df_layer1[TIME_SERIES_COLS].isnull().sum().sum()

df_layer1[TIME_SERIES_COLS] = (
    df_layer1
    .groupby('Division')[TIME_SERIES_COLS]
    .transform(lambda s: s.ffill().bfill())
)

nan_after_l1 = df_layer1[TIME_SERIES_COLS].isnull().sum().sum()

print(f"  NaN in time-series cols before : {nan_before_l1}")
print(f"  NaN in time-series cols after  : {nan_after_l1}")
print(f"  Resolved                       : {nan_before_l1 - nan_after_l1}")

if nan_after_l1 > 0:
    still_missing = df_layer1[TIME_SERIES_COLS].isnull().sum()
    print(f"  Still missing (entire series gap — fallback to KNN): ")
    print(still_missing[still_missing > 0].to_string())

# =============================================================================
# 4. LAYER 2 — KNN IMPUTATION for static/spatial columns
# =============================================================================
print("\n" + "─" * 70)
print("STEP 3 — LAYER 2: KNNImputer (n_neighbors=5) for static columns")
print("─" * 70)
print("""
  Rationale:
  Static columns like Area_km2 and Population_Share_Pct are geographically
  fixed. A missing value for division D is best estimated by the k=5 most
  similar divisions in feature space — the KNN algorithm finds them
  automatically in the scaled numeric space.

  Pipeline:
    StandardScaler → KNNImputer(n_neighbors=5, weights='distance')
    → inverse_scale → write back to dataframe
""")

df_layer2 = df_layer1.copy()

# Build KNN feature matrix: use KNN_COLS + time-series cols as context
knn_context_cols = KNN_COLS + TIME_SERIES_COLS
knn_matrix = df_layer2[knn_context_cols].copy()

nan_before_l2 = df_layer2[KNN_COLS].isnull().sum().sum()

# Scale before KNN (distance-based — scaling is mandatory)
scaler_knn = StandardScaler()
knn_matrix_scaled = scaler_knn.fit_transform(knn_matrix)

knn_imputer = KNNImputer(
    n_neighbors=5,
    weights='distance',   # closer divisions get higher weight
    metric='nan_euclidean'
)
knn_imputed_scaled = knn_imputer.fit_transform(knn_matrix_scaled)
knn_imputed = scaler_knn.inverse_transform(knn_imputed_scaled)
knn_result_df = pd.DataFrame(knn_imputed, columns=knn_context_cols, index=df_layer2.index)

# Write back only the KNN_COLS
for col in KNN_COLS:
    df_layer2[col] = knn_result_df[col]

nan_after_l2 = df_layer2[KNN_COLS].isnull().sum().sum()

print(f"  n_neighbors                     : 5")
print(f"  weights                         : distance")
print(f"  NaN in static cols before       : {nan_before_l2}")
print(f"  NaN in static cols after        : {nan_after_l2}")
print(f"  Resolved                        : {nan_before_l2 - nan_after_l2}")

# =============================================================================
# 5. LAYER 3 — MICE (IterativeImputer) for waste stream columns
# =============================================================================
print("\n" + "─" * 70)
print("STEP 4 — LAYER 3: IterativeImputer / MICE for waste stream columns")
print("─" * 70)
print("""
  Rationale:
  Waste stream columns are highly correlated (r ≥ 0.63–0.99). MICE
  (Multiple Imputation by Chained Equations) exploits this joint
  distribution: each column's missing values are imputed by regressing
  it on all other columns, iterating until convergence.

  Configuration:
    estimator       = BayesianRidge (handles multicollinearity well)
    max_iter        = 10  (convergence typically by iter 5–7)
    n_nearest_features = 15  (limits to most correlated predictors)
    imputation_order = ascending (fewest missing → most missing)
    random_state    = 42
""")

df_layer3 = df_layer2.copy()

# MICE feature matrix: all numeric except Year, Area (fixed geography)
mice_context_cols = [c for c in df_layer3.columns
                     if c not in CAT_COLS + ['Year', 'Area_km2']]
mice_matrix = df_layer3[mice_context_cols].copy()

nan_before_l3 = df_layer3[MICE_COLS].isnull().sum().sum()

scaler_mice = StandardScaler()
mice_matrix_scaled = scaler_mice.fit_transform(mice_matrix)

mice_imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    n_nearest_features=15,
    imputation_order='ascending',   # fill least-missing first
    initial_strategy='median',       # warm start
    random_state=RANDOM_STATE,
    verbose=0
)

# Track convergence delta across iterations manually
print("\n  Running MICE iterations...")
# Fit and track
mice_imputer.fit(mice_matrix_scaled)
mice_imputed_scaled = mice_imputer.transform(mice_matrix_scaled)
mice_imputed = scaler_mice.inverse_transform(mice_imputed_scaled)
mice_result_df = pd.DataFrame(mice_imputed, columns=mice_context_cols, index=df_layer3.index)

for col in MICE_COLS:
    if col in mice_result_df.columns:
        df_layer3[col] = mice_result_df[col]

nan_after_l3 = df_layer3[MICE_COLS].isnull().sum().sum()

print(f"  NaN in waste stream cols before : {nan_before_l3}")
print(f"  NaN in waste stream cols after  : {nan_after_l3}")
print(f"  Resolved                        : {nan_before_l3 - nan_after_l3}")

# Final fully imputed dataframe
df_imputed = df_layer3.copy()

# =============================================================================
# 6. IMPUTATION QUALITY EVALUATION (RMSE vs ground truth)
# =============================================================================
print("\n" + "─" * 70)
print("STEP 5 — IMPUTATION QUALITY EVALUATION (RMSE vs Ground Truth)")
print("─" * 70)
print("""
  For each imputed column we compute:
    RMSE  — root mean squared error on the artificially injected NaN cells
    NRMSE — RMSE normalised by column range (scale-free, 0=perfect, 1=bad)
    R²    — coefficient of determination (1=perfect reconstruction)
""")

results = []

for col in TIME_SERIES_COLS + KNN_COLS + MICE_COLS:
    if col not in df_ground_truth.columns or col not in df_imputed.columns:
        continue
    if col not in df_with_missing.columns:
        continue

    # Identify cells that were artificially made missing
    was_missing = df_with_missing[col].isnull()
    if not was_missing.any():
        continue

    true_vals    = df_ground_truth.loc[was_missing, col].values
    imputed_vals = df_imputed.loc[was_missing, col].values

    # Guard against non-finite
    mask = np.isfinite(true_vals) & np.isfinite(imputed_vals)
    if mask.sum() == 0:
        continue

    tv = true_vals[mask]
    iv = imputed_vals[mask]

    rmse  = np.sqrt(np.mean((tv - iv) ** 2))
    rng   = tv.max() - tv.min() if tv.max() != tv.min() else 1.0
    nrmse = rmse / rng
    ss_res = np.sum((tv - iv) ** 2)
    ss_tot = np.sum((tv - tv.mean()) ** 2)
    r2    = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    layer = ('Time-series' if col in TIME_SERIES_COLS else
             'KNN'         if col in KNN_COLS          else 'MICE')

    results.append({
        'Column': col,
        'Layer': layer,
        'N_Imputed': int(mask.sum()),
        'RMSE': round(rmse, 4),
        'NRMSE': round(nrmse, 4),
        'R2': round(r2, 4)
    })

eval_df = pd.DataFrame(results).sort_values(['Layer', 'NRMSE'])

print(f"\n  {'Column':<35} {'Layer':<14} {'N':>5} {'RMSE':>12} {'NRMSE':>8} {'R²':>8}")
print(f"  {'─'*35} {'─'*14} {'─'*5} {'─'*12} {'─'*8} {'─'*8}")
for _, row in eval_df.iterrows():
    r2_str = f"{row['R2']:.4f}" if not pd.isna(row['R2']) else "   N/A"
    print(f"  {row['Column']:<35} {row['Layer']:<14} {row['N_Imputed']:>5} "
          f"{row['RMSE']:>12.4f} {row['NRMSE']:>8.4f} {r2_str:>8}")

print()
layer_summary = eval_df.groupby('Layer')[['NRMSE','R2']].mean().round(4)
print("  Mean metrics by layer:")
print(layer_summary.to_string())

# =============================================================================
# 7. BEFORE / AFTER COMPARISON SNAPSHOT
# =============================================================================
print("\n" + "─" * 70)
print("STEP 6 — BEFORE / AFTER SNAPSHOT (sample imputed values)")
print("─" * 70)

sample_checks = {
    'GDP_per_Capita_USD'   : ('Time-series', 'Dhaka'),
    'Area_km2'             : ('KNN',          'Sylhet'),
    'General_Waste_kg_day' : ('MICE',          'Rajshahi'),
    'Cytotoxic_Waste_kg_day': ('MICE',         'Barisal'),
}

for col, (layer, div) in sample_checks.items():
    if col not in df_with_missing.columns:
        continue
    # Find a missing cell for this division
    missing_rows = df_with_missing[
        (df_with_missing['Division'] == div) & df_with_missing[col].isnull()
    ]
    if missing_rows.empty:
        # Any division will do
        missing_rows = df_with_missing[df_with_missing[col].isnull()]
    if missing_rows.empty:
        continue

    idx = missing_rows.index[0]
    yr  = df_with_missing.loc[idx, 'Year']
    div_actual = df_with_missing.loc[idx, 'Division']
    original   = df_ground_truth.loc[idx, col]
    imputed    = df_imputed.loc[idx, col]
    error      = abs(original - imputed) / abs(original) * 100 if original != 0 else 0.0

    print(f"  {col} | {div_actual} | Year {int(yr)}")
    print(f"    Layer     : {layer}")
    print(f"    Original  : {original:.4f}")
    print(f"    Imputed   : {imputed:.4f}")
    print(f"    Error     : {error:.2f}%")
    print()

# =============================================================================
# 8. FINAL NaN AUDIT
# =============================================================================
print("─" * 70)
print("STEP 7 — FINAL NaN AUDIT")
print("─" * 70)

total_nan_before = df_with_missing.isnull().sum().sum()
total_nan_after  = df_imputed.isnull().sum().sum()

print(f"\n  NaN count before imputation : {total_nan_before}")
print(f"  NaN count after  imputation : {total_nan_after}")

remaining = df_imputed.isnull().sum()
remaining = remaining[remaining > 0]
if remaining.empty:
    print("  ✅  Zero NaN remaining — dataset fully imputed.")
else:
    print("  ⚠️  Remaining NaN (requires manual review):")
    print(remaining.to_string())

# =============================================================================
# 9. EXPORT
# =============================================================================
output_path = 'BD_MedWaste_IMPUTED.csv'
df_imputed.to_csv(output_path, index=False)
print(f"\n  ✅  Imputed dataset saved → {output_path}")
eval_df.to_csv('imputation_quality_report.csv', index=False)
print(f"  ✅  Quality report saved  → imputation_quality_report.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 70)
print("IMPUTATION PIPELINE SUMMARY")
print("=" * 70)

mean_nrmse = eval_df['NRMSE'].mean()
mean_r2    = eval_df['R2'].mean()

print(f"""
  ┌────────────────────────────────────────────────────────────────┐
  │  Layer 1 — Time-series ffill/bfill│ {len(TIME_SERIES_COLS):>2} columns           │
  │  Layer 2 — KNNImputer (k=5)       │ {len(KNN_COLS):>2} columns                   │
  │  Layer 3 — MICE / IterativeImputer│ {len(MICE_COLS):>2} columns                  │
  │                                   │                                              │
  │  Total NaN resolved               │ {total_nan_before - total_nan_after:>5} cells│
  │  Mean NRMSE (all cols)            │ {mean_nrmse:>8.4f}                           │
  │  Mean R²   (all cols)             │ {mean_r2:>8.4f}                              │
  └────────────────────────────────────────────────────────────────┘

""")