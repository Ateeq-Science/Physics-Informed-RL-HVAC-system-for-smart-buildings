import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = r"C:\\Users\\MOHAM\\CU_BEMS_preprocessed_wide_v2.csv"
MODEL_DIR = r"C:\\Users\\MOHAM\\saved_models_temp_sensitive_v2"
OUTPUT_CSV = "TOU_cost_results_focused_v6.csv"
PLOTS_DIR = "TOU_cost_plots_focused_v6"
os.makedirs(PLOTS_DIR, exist_ok=True)

SIM_MONTHS = {5: 'May', 6: 'June', 7: 'July', 8: 'August'}
FOCUSED_ACS = ["AC1(kW)", "AC4(kW)"]
TOU_PRICES = {
    "off_peak": 7.6,
    "mid_peak": 12.2,
    "on_peak": 15.8
}

SCENARIOS = {
    "baseline": (23, 25),
    "eco_mode": (24, 26),
    "comfort_mode": (23, 24),
    "aggressive_savings": (25, 27),
    "occupancy_based": (24, 26),
    "precooling": (22, 24),
    "humidity_focus": (23, 25),
    "relaxed_comfort": (26, 28),
    "night_setback": (27, 29),
    "dynamic_adjust": (24, 25)
}

def get_tou_price(hour, month, daytype=1):
    if daytype == 0:
        return TOU_PRICES["off_peak"]
    if 7 <= hour < 11 or 17 <= hour < 19:
        return TOU_PRICES["mid_peak"] if 5 <= month <= 10 else TOU_PRICES["on_peak"]
    elif 11 <= hour < 17:
        return TOU_PRICES["on_peak"] if 5 <= month <= 10 else TOU_PRICES["mid_peak"]
    else:
        return TOU_PRICES["off_peak"]

print("🔹 Loading dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df["Month"] = df["Date"].dt.month
df["Hour"] = df["Date"].dt.hour
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["DayType"] = np.where(df["DayOfWeek"] < 5, 1, 0)
df = df[df["Month"].isin(SIM_MONTHS.keys())].copy()
print(f"✅ Data filtered: {df.shape}")

results = []

for ac in FOCUSED_ACS:
    model_path = os.path.join(MODEL_DIR, f"{ac}_model.pkl")
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found for {ac}, skipping...")
        continue

    print(f"\n▶ Simulating costs for {ac}")
    model_info = joblib.load(model_path)
    model = model_info["model"]
    features = model_info["features"]

    X_base = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    baseline_pred = model.predict(X_base)

    for scenario, (low, high) in SCENARIOS.items():
        df_mode = df.copy()
        if "S1(degC)" in df_mode.columns:
            df_mode["S1(degC)"] = df_mode["S1(degC)"].clip(lower=low, upper=high)

        X_mode = df_mode[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_pred = model.predict(X_mode)

        costs = []
        temp_violations = 0
        total_hours = len(df_mode)

        for pred, row in zip(y_pred, df_mode.itertuples(index=False)):
            hour = int(row.Hour)
            month = int(row.Month)
            daytype = row.DayType
            price = get_tou_price(hour, month, daytype)

            costs.append((pred * price) / 100.0)

            if hasattr(row, 'S1(degC)') and not (low <= row._asdict()['S1(degC)'] <= high):
                temp_violations += 1

        df_mode[f"Cost_{scenario}"] = costs
        monthly_cost = df_mode.groupby("Month")[f"Cost_{scenario}"].sum()
        violation_pct = (temp_violations / total_hours) * 100

        for month, cost in monthly_cost.items():
            results.append([ac, SIM_MONTHS[month], scenario, cost, violation_pct])

results_df = pd.DataFrame(results, columns=["AC_Unit", "Month", "Scenario", "Monthly_Cost_$", "TempViolation_%"])
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Results saved: {OUTPUT_CSV}")

for ac in results_df["AC_Unit"].unique():
    data = results_df[results_df["AC_Unit"] == ac]
    pivot_data = data.pivot(index="Month", columns="Scenario", values="Monthly_Cost_$").fillna(0)
    ax = pivot_data.plot(kind="bar", figsize=(12, 6))
    plt.title(f"Monthly Electricity Costs ({ac})")
    plt.ylabel("Cost ($)")
    plt.xlabel("Month")
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{ac}_monthly_costs.png")
    plt.close()

overall = results_df.groupby(["Scenario", "Month"])["Monthly_Cost_$"].sum().unstack().T
ax = overall.plot(kind="bar", figsize=(12, 6))
plt.title("Overall Monthly Electricity Costs (AC1 & AC4)")
plt.ylabel("Cost ($)")
plt.xlabel("Month")
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=8)
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/overall_monthly_costs_focused.png")
plt.close()

print("🎉 Focused Simulation for AC1 & AC4 with Extended Scenarios Completed!")
