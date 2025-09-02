import os, math, joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = r"C:\Users\MOHAM\CU_BEMS_preprocessed_wide_v2.csv"
OUT_DIR = "saved_models_timeaware_v3"
FEATIMP_DIR = "feature_importance_v3"
PREDPLOT_DIR = "prediction_plots_v3"
METRICS_CSV = "model_metrics_timeaware_v3.csv"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FEATIMP_DIR, exist_ok=True)
os.makedirs(PREDPLOT_DIR, exist_ok=True)

SUMMER_MONTHS = [5, 6, 7, 8, 9]
ASHRAE_TEMP_COLS = ["S1(degC)", "S1(degC)_Lag1", "S1(degC)_Lag2"]

def cyclical_encode(series, max_val):
    
    x = 2 * math.pi * (series.astype(float) / max_val)
    return np.sin(x), np.cos(x)

def evaluate(y_true, y_pred, prefix):
    return {
        f"{prefix}_R2": round(r2_score(y_true, y_pred), 4),
        f"{prefix}_RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        f"{prefix}_MAE": round(mean_absolute_error(y_true, y_pred), 4)
    }

def time_split(df_sorted, frac_test=0.15, frac_val=0.15):
    n = len(df_sorted)
    n_test = int(n * frac_test)
    n_val = int(n * frac_val)
    n_train = n - n_test - n_val
    return (
        df_sorted.iloc[:n_train],
        df_sorted.iloc[n_train:n_train+n_val],
        df_sorted.iloc[n_train+n_val:]
    )

def is_dead_target(y, nonzero_thresh=0.01, std_thresh=1e-6):
    nz = (y > 0).sum()
    ratio = nz / max(1, len(y))
    return (ratio < nonzero_thresh) or (y.std() < std_thresh), ratio

df = pd.read_csv(DATA_PATH, low_memory=False, parse_dates=["Date"])
df = df.sort_values("Date").copy()

df["Hour"] = df["Date"].dt.hour
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["DayType"] = (df["DayOfWeek"] < 5).astype(int)
df = df[df["Month"].isin(SUMMER_MONTHS)].copy()

df["Hour_sin"], df["Hour_cos"]   = cyclical_encode(df["Hour"], 24)
df["Month_sin"], df["Month_cos"] = cyclical_encode(df["Month"], 12)

ac_targets = [
    c for c in df.columns
    if c.startswith("AC") and "(kW)" in c and "Lag" not in c and c not in ["AC11(kW)"]
]

lag1_cols = [c for c in df.columns if c.endswith("_Lag1")]
lag2_cols = [c for c in df.columns if c.endswith("_Lag2")]
weather_cols = ["T2M", "RH2M", "T2MDEW", "T2MWET", "ALLSKY_SFC_SW_DWN", "PS", "WS50M"]
time_cols = ["Hour","Month","DayOfWeek","DayType","Hour_sin","Hour_cos","Month_sin","Month_cos"]

for c in weather_cols + time_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df[weather_cols] = df[weather_cols].fillna(0)

rows = []
for ac in ac_targets:
    y_all = df[ac].fillna(0).copy()

    
    dead, nz_ratio = is_dead_target(y_all, nonzero_thresh=0.02)
    if dead or ac in ["AC12(kW)", "AC13(kW)", "AC14(kW)"]:
        print(f"⏭️  Skipping {ac}: non-zero ratio={nz_ratio:.3f} (or known near-zero).")
        continue

    print(f"\n🔧 Training time-aware model for {ac} (non-zero ratio={nz_ratio:.3f})")

    feat_list = []
    feat_list += [c for c in lag1_cols if c in df.columns]
    feat_list += [c for c in lag2_cols if c in df.columns]
    feat_list += [c for c in weather_cols if c in df.columns]
    feat_list += [c for c in time_cols if c in df.columns]
    feat_list += [c for c in ASHRAE_TEMP_COLS if c in df.columns]
    
    feat_list = list(dict.fromkeys(feat_list))

    tmp = df[["Date", ac] + feat_list].dropna(subset=feat_list).sort_values("Date").copy()
    X_all = tmp[feat_list].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_all = tmp[ac].fillna(0)

    tr, va, te = time_split(tmp, frac_test=0.15, frac_val=0.15)
    X_tr, y_tr = tr[feat_list], tr[ac]
    X_va, y_va = va[feat_list], va[ac]
    X_te, y_te = te[feat_list], te[ac]

    is_sparse = (y_all.eq(0).mean() > 0.3)
    if is_sparse:
        objective = "tweedie"
        extra = dict(objective="tweedie", tweedie_variance_power=1.2)
    else:
        objective = "regression"
        extra = dict(objective="regression")
    print(f"   ➜ Objective: {objective}")

    model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=20,
        reg_alpha=0.2,
        reg_lambda=0.6,
        random_state=42,
        **extra
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)]
    )

    joblib.dump({"model": model, "features": feat_list}, os.path.join(OUT_DIR, f"{ac}_model.pkl"))

    yhat_tr = model.predict(X_tr, num_iteration=model.best_iteration_)
    yhat_va = model.predict(X_va, num_iteration=model.best_iteration_)
    yhat_te = model.predict(X_te, num_iteration=model.best_iteration_)

    row = {"AC_Unit": ac, "Data_Points": len(tmp), "NonZero_Ratio": round(1 - float((y_all==0).mean()), 3)}
    row.update(evaluate(y_tr, yhat_tr, "Train"))
    row.update(evaluate(y_va, yhat_va, "Val"))
    row.update(evaluate(y_te, yhat_te, "Test"))
    rows.append(row)

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    topn = min(20, len(idx))
    top_feats = [feat_list[i] for i in idx[:topn]]

    plt.figure(figsize=(10, 6))
    plt.title(f"Top {topn} Features for {ac}")
    plt.bar(range(topn), importances[idx[:topn]], align="center")
    plt.xticks(range(topn), top_feats, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FEATIMP_DIR, f"{ac}_feature_importance.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    y_te_arr = y_te.values
    yhat_te_arr = yhat_te
    nplot = min(300, len(y_te_arr))
    plt.plot(y_te_arr[:nplot], label="Actual", alpha=0.8)
    plt.plot(yhat_te_arr[:nplot], label="Predicted", alpha=0.8)
    plt.title(f"{ac} — Actual vs Pred (last {nplot} of test)")
    plt.xlabel("Index (chronological)")
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PREDPLOT_DIR, f"{ac}_pred_plot.png"))
    plt.close()

pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)
print("\n✅ Time‑aware models trained. Saved bundles, importances, and metrics.")
