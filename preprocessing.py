import pandas as pd

INPUT_FILE = r"D:\NOTES\TMU\MRP\CU_BEMS_NASA_merged.csv"
OUTPUT_LONG = "CU_BEMS_preprocessed_long_v2.csv"
OUTPUT_WIDE = "CU_BEMS_preprocessed_wide_v2.csv"

lag_vars = [
    "AC1(kW)", "AC2(kW)", "AC3(kW)", "AC4(kW)", "AC5(kW)", "AC6(kW)",
    "AC7(kW)", "AC8(kW)", "AC9(kW)", "AC10(kW)", "AC12(kW)", "AC13(kW)",
    "AC14(kW)", "S1(degC)", "S1(RH%)", "Light(kW)", "Plug(kW)"
]
weather_cols = ["T2M", "RH2M", "T2MDEW", "ALLSKY_SFC_SW_DWN", "T2MWET", "PS", "WS50M"]

print("🔹 Loading dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False, parse_dates=["Date"])
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

df = df.sort_values(by=["Floor", "Zone", "Variable", "Date"]).reset_index(drop=True)

print("🔹 Computing lag features...")
df["Value_Lag1"] = df.groupby(["Floor", "Zone", "Variable"])["Value"].shift(1)
df["Value_Lag2"] = df.groupby(["Floor", "Zone", "Variable"])["Value"].shift(2)
df["Value_Lag1"] = df["Value_Lag1"].fillna(0)
df["Value_Lag2"] = df["Value_Lag2"].fillna(0)

df.to_csv(OUTPUT_LONG, index=False)
print(f"✅ Long format saved: {OUTPUT_LONG} ({df.shape[0]} rows)")

df_weather = df[["Floor", "Zone", "Date"] + weather_cols].drop_duplicates()

print("🔹 Pivoting dataset to wide format...")
df_wide = df.pivot_table(index=["Floor", "Zone", "Date"], 
                         columns="Variable", values="Value").reset_index()

df_lag1 = df[df["Variable"].isin(lag_vars)].pivot_table(
    index=["Floor", "Zone", "Date"], columns="Variable", values="Value_Lag1"
).add_suffix("_Lag1").reset_index()

df_lag2 = df[df["Variable"].isin(lag_vars)].pivot_table(
    index=["Floor", "Zone", "Date"], columns="Variable", values="Value_Lag2"
).add_suffix("_Lag2").reset_index()

df_wide = (
    df_wide.merge(df_lag1, on=["Floor", "Zone", "Date"], how="left")
           .merge(df_lag2, on=["Floor", "Zone", "Date"], how="left")
           .merge(df_weather, on=["Floor", "Zone", "Date"], how="left")
)

df_wide = df_wide.fillna(0)

df_wide.to_csv(OUTPUT_WIDE, index=False)
print(f"✅ Wide format saved: {OUTPUT_WIDE} ({df_wide.shape[0]} rows, {df_wide.shape[1]} columns)")

print("\n🎉 Preprocessing completed successfully with NASA weather data included!")
