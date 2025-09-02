import os
import pandas as pd

folder = r"C:\Users\MOHAM\Downloads\Cubems dataset"
output_path = "cubems_hourly_lagged_wide.csv"
all_data = []

def get_variable_type(variable_name):
    if any(x in variable_name for x in ["Light", "AC", "Plug"]):
        return "power"
    else:
        return "sensor"

for file in sorted(os.listdir(folder)):
    if file.endswith(".csv"):
        filepath = os.path.join(folder, file)
        print(f"\nProcessing {file}...")
        floor = file.replace(".csv", "")
        df = pd.read_csv(filepath)

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        invalid_rows = df["Date"].isna().sum()
        if invalid_rows > 0:
            print(f"⚠️ {invalid_rows} rows with invalid timestamps will be skipped.")
        df = df.dropna(subset=["Date"])

        df_long = df.melt(id_vars="Date", var_name="RawVariable", value_name="Value")
        df_long["Zone"] = df_long["RawVariable"].str.extract(r"(z\d+)")
        df_long["Variable"] = df_long["RawVariable"].str.extract(r"_(.*)")
        df_long["Floor"] = floor
        df_long = df_long.dropna(subset=["Zone"])

        df_long["Type"] = df_long["Variable"].apply(get_variable_type)

        df_long.set_index("Date", inplace=True)

        power_data = df_long[df_long["Type"] == "power"]
        power_hourly = (
            power_data
            .groupby(["Floor", "Zone", "Variable"])
            .resample("H")["Value"]
            .sum()
            .reset_index()
        )
        power_hourly["Value"] = power_hourly["Value"] / 60

        sensor_data = df_long[df_long["Type"] == "sensor"]
        sensor_hourly = (
            sensor_data
            .groupby(["Floor", "Zone", "Variable"])
            .resample("H")["Value"]
            .mean()
            .reset_index()
        )

        combined_hourly = pd.concat([power_hourly, sensor_hourly], ignore_index=True)
        combined_hourly = combined_hourly.loc[:, ["Floor", "Zone", "Variable", "Date", "Value"]]

        all_data.append(combined_hourly)

final_df = pd.concat(all_data, ignore_index=True)

final_df = final_df.sort_values(by=["Floor", "Zone", "Variable", "Date"])

final_df["Lag1"] = final_df.groupby(["Floor", "Zone", "Variable"])["Value"].shift(1)
final_df["Lag2"] = final_df.groupby(["Floor", "Zone", "Variable"])["Value"].shift(2)

df_pivot = final_df.pivot_table(
    index=["Floor", "Zone", "Date"],
    columns="Variable",
    values="Value"
).reset_index()

df_lag1 = final_df.pivot_table(
    index=["Floor", "Zone", "Date"],
    columns="Variable",
    values="Lag1"
).reset_index()
df_lag1 = df_lag1.add_suffix("_Lag1")
df_lag1.rename(columns={"Floor_Lag1": "Floor", "Zone_Lag1": "Zone", "Date_Lag1": "Date"}, inplace=True)

df_lag2 = final_df.pivot_table(
    index=["Floor", "Zone", "Date"],
    columns="Variable",
    values="Lag2"
).reset_index()
df_lag2 = df_lag2.add_suffix("_Lag2")
df_lag2.rename(columns={"Floor_Lag2": "Floor", "Zone_Lag2": "Zone", "Date_Lag2": "Date"}, inplace=True)

final_wide = df_pivot.merge(df_lag1, on=["Floor", "Zone", "Date"], how="left")
final_wide = final_wide.merge(df_lag2, on=["Floor", "Zone", "Date"], how="left")

final_wide["Hour"] = pd.to_datetime(final_wide["Date"]).dt.hour
final_wide["Month"] = pd.to_datetime(final_wide["Date"]).dt.month
final_wide["DayOfWeek"] = pd.to_datetime(final_wide["Date"]).dt.dayofweek
final_wide["DayType"] = final_wide["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)  # weekend=1

final_wide.to_csv(output_path, index=False)

print(f"\n✅ Preprocessing complete! Dataset saved to: {output_path}")
print(f"✅ Final shape: {final_wide.shape}")
