import pandas as pd
import numpy as np
import heapq
import json
import os
import warnings
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# 1. LOAD RAW DATA
print("=" * 60)
print("STEP 1 — LOADING RAW DATA")
print("=" * 60)

INPUT  = "D:\PRADEEP\DIGITIVITY\Delivery_optimizer\Delivery_Logistics.csv"
OUTDIR = "D:\PRADEEP\DIGITIVITY\Delivery_optimizer\outputs"
os.makedirs(OUTDIR, exist_ok=True)

raw = pd.read_csv(INPUT)
print(f"  Raw shape        : {raw.shape}")
print(f"  Columns          : {list(raw.columns)}")


# 2. DATA CLEANING
print("\n" + "=" * 60)
print("STEP 2 — DATA CLEANING")
print("=" * 60)

df = raw.copy()

# ── 2a. Fix delivery_id (all rows have 250.99 or 24750.01 as junk) ──
# Re-index with proper sequential IDs
df["delivery_id"] = range(1, len(df) + 1)
print(f"  Re-indexed delivery_id: 1 … {len(df)}")

# ── 2b. Fix corrupted time columns ──
# Values like '1970-01-01 00:00:00.000000008' → nanoseconds → hours
def ns_to_hours(series):
    """Extract the nanosecond integer from the epoch timestamp string."""
    return (
        pd.to_datetime(series, errors="coerce")
        .astype("int64")
        .clip(lower=0)
    )

df["delivery_time_hours"] = ns_to_hours(df["delivery_time_hours"])
df["expected_time_hours"] = ns_to_hours(df["expected_time_hours"])
# Values are nanoseconds stored as epoch; the meaningful unit is hours (1–24 range in raw)
# Divide by 1e9 to get seconds, but since max raw was ~16 ns → treat as literal hour values
# Re-extract: the original csv ns values ARE the hours (1 = 1 hr, 16 = 16 hrs)
df["delivery_time_hours"] = df["delivery_time_hours"]  // 1          # already int ns = hours
df["expected_time_hours"] = df["expected_time_hours"]  // 1
# Clamp to realistic range [1, 72]
df["delivery_time_hours"] = df["delivery_time_hours"].clip(1, 72)
df["expected_time_hours"] = df["expected_time_hours"].clip(1, 72)
print(f"  Fixed time columns — delivery_time range: "
      f"{df['delivery_time_hours'].min()} – {df['delivery_time_hours'].max()} hrs")

# ── 2c. Standardise text columns ──
text_cols = ["delivery_partner","package_type","vehicle_type","delivery_mode",
             "region","weather_condition","delayed","delivery_status"]
for c in text_cols:
    df[c] = df[c].str.strip().str.lower()

# ── 2d. Validate / fix delayed flag vs delivery_status ──
# If status == 'delayed' but delayed flag == 'no' → fix flag
mask = (df["delivery_status"] == "delayed") & (df["delayed"] == "no")
df.loc[mask, "delayed"] = "yes"
print(f"  Fixed delayed flag inconsistency: {mask.sum()} rows")

# ── 2e. Duplicate removal ──
before = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "delivery_id"])
df["delivery_id"] = range(1, len(df) + 1)   # re-index after drop
after  = len(df)
print(f"  Removed duplicates: {before - after} rows  ({after} remain)")

# ── 2f. Outlier detection & removal — Z-score + IQR ──
numeric_cols = ["distance_km", "package_weight_kg", "delivery_cost"]
cleaning_report = {}

for col in numeric_cols:
    z        = np.abs(stats.zscore(df[col]))
    q1, q3   = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr      = q3 - q1
    iqr_mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    z_mask   = z > 3

    combined = iqr_mask | z_mask
    n_out    = combined.sum()
    max_z    = z.max()

    cleaning_report[col] = {
        "max_z_score"   : round(float(max_z), 4),
        "iqr_outliers"  : int(iqr_mask.sum()),
        "z_outliers"    : int(z_mask.sum()),
        "total_removed" : int(n_out),
    }
    df = df[~combined].reset_index(drop=True)
    print(f"  [{col}]  max_z={max_z:.3f}  IQR_out={iqr_mask.sum()}  "
          f"Z_out={z_mask.sum()}  removed={n_out}")

df["delivery_id"] = range(1, len(df) + 1)
print(f"\n  Final clean shape: {df.shape}")

# Save cleaned data
clean_path = os.path.join(OUTDIR, "cleaned_data.csv")
df.to_csv(clean_path, index=False)
print(f"  Saved → {clean_path}")


# 3. FEATURE ENGINEERING
print("\n" + "=" * 60)
print("STEP 3 — FEATURE ENGINEERING")
print("=" * 60)

# Priority mapping
priority_map  = {"same day": "High", "express": "High",
                 "two day" : "Medium", "standard": "Low"}
priority_rank = {"High": 0, "Medium": 1, "Low": 2}

df["priority"]      = df["delivery_mode"].map(priority_map)
df["priority_rank"] = df["priority"].map(priority_rank)

# Time efficiency: ratio of expected to actual
df["time_efficiency"] = (df["expected_time_hours"] / df["delivery_time_hours"]).clip(0.1, 5)

# Cost per km
df["cost_per_km"] = df["delivery_cost"] / df["distance_km"].clip(lower=1)

# Delay binary
df["is_delayed"] = (df["delayed"] == "yes").astype(int)

# Composite delivery score (0–100) used for agent rating
df["perf_score"] = (
    df["delivery_rating"] / 5 * 40          # 40 pts: customer rating
    + (1 - df["is_delayed"]) * 35           # 35 pts: on-time
    + df["time_efficiency"].clip(0, 1) * 25 # 25 pts: efficiency
)

# Synthetic geo coordinates (region-based centroids + jitter for map display)
region_coords = {
    "north"  : (28.7,  77.1),
    "south"  : (12.9,  80.2),
    "east"   : (22.6,  88.4),
    "west"   : (23.0,  72.6),
    "central": (23.2,  77.4),
}
np.random.seed(42)
df["lat"] = df["region"].map(lambda r: region_coords[r][0]) + np.random.uniform(-3, 3, len(df))
df["lon"] = df["region"].map(lambda r: region_coords[r][1]) + np.random.uniform(-3, 3, len(df))

print(f"  Priority distribution:\n{df['priority'].value_counts().to_string()}")
print(f"  Avg time_efficiency : {df['time_efficiency'].mean():.3f}")
print(f"  Avg perf_score      : {df['perf_score'].mean():.2f}/100")


# 4. ML MODELS
print("\n" + "=" * 60)
print("STEP 4 — MACHINE LEARNING")
print("=" * 60)

le = LabelEncoder()
df_ml = df.copy()
for c in ["delivery_mode", "region", "weather_condition", "vehicle_type",
          "package_type", "delivery_partner"]:
    df_ml[c + "_enc"] = le.fit_transform(df_ml[c])

FEATURES = ["distance_km", "package_weight_kg", "delivery_cost",
            "delivery_rating", "priority_rank", "time_efficiency",
            "delivery_mode_enc", "region_enc", "weather_condition_enc",
            "vehicle_type_enc"]

X = df_ml[FEATURES]
y_cls = df_ml["is_delayed"]
y_reg = df_ml["delivery_cost"]

X_tr, X_te, yc_tr, yc_te = train_test_split(X, y_cls, test_size=0.2, random_state=42)
_, _, yr_tr, yr_te         = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# ── 4a. Random Forest: delay prediction ──
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, yc_tr)
rf_pred  = rf.predict(X_te)
rf_acc   = accuracy_score(yc_te, rf_pred)
rf_f1    = f1_score(yc_te, rf_pred)
print(f"  RandomForest Delay Classifier — Acc={rf_acc:.4f}  F1={rf_f1:.4f}")

# ── 4b. Gradient Boosting: cost prediction ──
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_tr, yr_tr)
gb_pred = gb.predict(X_te)
gb_r2   = r2_score(yr_te, gb_pred)
gb_mae  = mean_absolute_error(yr_te, gb_pred)
print(f"  GradientBoosting Cost Regressor — R²={gb_r2:.4f}  MAE=₹{gb_mae:.2f}")

# ── 4c. K-Means: route clustering (k=5) ──
km = KMeans(n_clusters=5, random_state=42, n_init=10)
df["route_cluster"] = km.fit_predict(df[["distance_km", "delivery_cost"]])
print(f"  KMeans (k=5) cluster sizes: {np.bincount(df['route_cluster']).tolist()}")

# Feature importances
feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

ml_results = {
    "rf_accuracy" : round(rf_acc, 4),
    "rf_f1"       : round(rf_f1, 4),
    "gb_r2"       : round(gb_r2, 4),
    "gb_mae"      : round(gb_mae, 2),
    "feature_importances": feat_imp.round(4).to_dict(),
}


# 5. AGENT ASSIGNMENT — greedy min-heap
print("\n" + "=" * 60)
print("STEP 5 — AGENT ASSIGNMENT")
print("=" * 60)

NUM_AGENTS = 3
AGENT_NAMES = ["Agent Alpha", "Agent Beta", "Agent Gamma"]

# Sort: priority first (High→Low), then by distance ascending
sorted_df = df.sort_values(["priority_rank", "distance_km"]).reset_index(drop=True)

# Min-heap: (total_distance, agent_idx)
heap = [(0.0, i) for i in range(NUM_AGENTS)]
heapq.heapify(heap)

agent_col = []
for dist in sorted_df["distance_km"]:
    total, agent_idx = heapq.heappop(heap)
    agent_col.append(AGENT_NAMES[agent_idx])
    heapq.heappush(heap, (total + dist, agent_idx))

sorted_df["assigned_agent"] = agent_col

# Re-merge back to original order
df = df.merge(
    sorted_df[["delivery_id", "assigned_agent"]],
    on="delivery_id", how="left"
)

# Print balance stats
for name in AGENT_NAMES:
    sub = df[df["assigned_agent"] == name]
    print(f"  {name}: {len(sub):,} deliveries | "
          f"dist={sub['distance_km'].sum():,.1f} km | "
          f"High={( sub['priority']=='High').sum()} | "
          f"delayed={(sub['is_delayed']==1).sum()}")


# 6. AGENT RATINGS
print("\n" + "=" * 60)
print("STEP 6 — AGENT RATINGS")
print("=" * 60)

agent_summary_rows = []
for name in AGENT_NAMES:
    sub = df[df["assigned_agent"] == name]
    n   = len(sub)

    on_time_rate  = 1 - sub["is_delayed"].mean()
    avg_rating    = sub["delivery_rating"].mean()
    avg_perf      = sub["perf_score"].mean()
    high_pct      = (sub["priority"] == "High").mean()
    total_dist    = sub["distance_km"].sum()
    avg_cost      = sub["delivery_cost"].mean()
    fail_rate     = (sub["delivery_status"] == "failed").mean()

    # Composite agent star rating (1–5)
    star = (
        on_time_rate * 2.0   # 2 pts
        + (avg_rating / 5) * 1.5   # 1.5 pts
        + avg_perf / 100 * 1.0     # 1 pt
        + (1 - fail_rate) * 0.5    # 0.5 pts
    )
    star = round(min(star, 5.0), 2)

    agent_summary_rows.append({
        "agent"          : name,
        "total_deliveries": n,
        "total_distance_km": round(total_dist, 1),
        "on_time_rate_%" : round(on_time_rate * 100, 2),
        "avg_customer_rating": round(avg_rating, 3),
        "high_priority_pct_%": round(high_pct * 100, 2),
        "avg_cost_per_delivery": round(avg_cost, 2),
        "failure_rate_%" : round(fail_rate * 100, 2),
        "avg_perf_score" : round(avg_perf, 2),
        "agent_star_rating": star,
    })
    print(f"  {name} ★ {star} — on_time={on_time_rate*100:.1f}%  "
          f"avg_rating={avg_rating:.2f}  fail={fail_rate*100:.1f}%")

agent_summary = pd.DataFrame(agent_summary_rows)
agent_path = os.path.join(OUTDIR, "agent_summary.csv")
agent_summary.to_csv(agent_path, index=False)
print(f"\n  Saved → {agent_path}")


# 7. ROUTE DATA FOR MAP (Leaflet.js)
print("\n" + "=" * 60)
print("STEP 7 — BUILDING ROUTE DATA")
print("=" * 60)

# Warehouse centroid (India logistics hub: Nagpur)
WAREHOUSE = {"lat": 21.1458, "lon": 79.0882, "name": "Warehouse (Nagpur Hub)"}

agent_colors = {
    "Agent Alpha": "#3b82f6",   # blue
    "Agent Beta" : "#f59e0b",   # amber
    "Agent Gamma": "#10b981",   # green
}

# For each agent sample up to 60 deliveries for the map (performance)
map_data = {"warehouse": WAREHOUSE, "agents": []}
for name in AGENT_NAMES:
    sub    = df[df["assigned_agent"] == name].copy()
    sample = sub.sample(min(60, len(sub)), random_state=42).sort_values("priority_rank")
    stops  = sample[["delivery_id","lat","lon","region","priority",
                      "distance_km","delivery_mode","delivery_rating",
                      "delivery_status","is_delayed","route_cluster"]].to_dict("records")

    # Aggregate per-region route path
    region_order = sample.groupby("region")["distance_km"].mean().sort_values().index.tolist()

    map_data["agents"].append({
        "name"     : name,
        "color"    : agent_colors[name],
        "stops"    : stops,
        "region_order": region_order,
        "summary"  : agent_summary[agent_summary["agent"] == name].to_dict("records")[0],
    })

route_json_path = os.path.join(OUTDIR, "route_data.json")
with open(route_json_path, "w") as f:
    json.dump(map_data, f, indent=2)
print(f"  Saved → {route_json_path}")


# 8. DELIVERY PLAN OUTPUT
print("\n" + "=" * 60)
print("STEP 8 — SAVING DELIVERY PLAN")
print("=" * 60)

plan_cols = ["delivery_id","assigned_agent","priority","region",
             "delivery_mode","distance_km","package_weight_kg",
             "delivery_cost","cost_per_km","delivery_rating",
             "delivery_status","is_delayed","perf_score","route_cluster","lat","lon"]
plan = df[plan_cols].copy()
plan_path = os.path.join(OUTDIR, "delivery_plan.csv")
plan.to_csv(plan_path, index=False)
print(f"  Saved → {plan_path}  ({len(plan):,} rows)")


# 9. CLEANING REPORT
report = {
    "raw_rows"        : int(raw.shape[0]),
    "clean_rows"      : int(df.shape[0]),
    "rows_removed"    : int(raw.shape[0] - df.shape[0]),
    "outlier_detail"  : cleaning_report,
    "ml_results"      : ml_results,
    "agent_summary"   : agent_summary.to_dict("records"),
}
report_path = os.path.join(OUTDIR, "cleaning_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"  Saved → {report_path}")

print("\n" + "=" * 60)
print("ALL DONE ✓")
print("=" * 60)
print(f"\nOutputs in {OUTDIR}/:")
for fname in ["cleaned_data.csv","delivery_plan.csv","agent_summary.csv",
              "route_data.json","cleaning_report.json"]:
    p = os.path.join(OUTDIR, fname)
    if os.path.exists(p):
        size_kb = os.path.getsize(p) // 1024
        print(f"  {fname:<30} {size_kb} KB")
