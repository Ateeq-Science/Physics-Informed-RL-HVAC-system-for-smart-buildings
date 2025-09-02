import pandas as pd

cubems = pd.read_csv(r"C:\Users\MOHAM\cubems_hourly_lagged_wide.csv")
cubems["Date"] = pd.to_datetime(cubems["Date"])

nasa = pd.read_csv(
    r"C:\Users\MOHAM\Downloads\POWER_Point_Hourly_20180701_20191231_013d74N_100d52E_LST.csv",
    skiprows=15
)

nasa["Date"] = pd.to_datetime(
    nasa[["YEAR", "MO", "DY", "HR"]].rename(
        columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"}
    )
)

nasa_clean = nasa.drop(columns=["YEAR", "MO", "DY", "HR"])

merged = pd.merge(
    cubems,
    nasa_clean,
    on="Date",
    how="inner"
)

print("Merged shape:", merged.shape)
print(merged.head())

merged.to_csv("CU_BEMS_NASA_merged.csv", index=False)

print("✅ Merged file saved successfully!")
