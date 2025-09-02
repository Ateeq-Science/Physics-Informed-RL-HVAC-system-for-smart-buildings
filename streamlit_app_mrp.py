import os
import math
import textwrap
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

mpl.rcParams["text.usetex"] = False

import plotly.express as px
import plotly.graph_objects as go

try:
    from streamlit_lottie import st_lottie
    HAS_LOTTIE = True
except Exception:
    HAS_LOTTIE = False

st.set_page_config(page_title="HVAC TOU Simulation — AC1 & AC4", layout="wide", page_icon="🧊")

PRIMARY = "#5BD1D7"
SECONDARY = "#A78BFA"
BG_DARK = "#0E1117"
CARD_BG = "#111827"
TEXT = "#E5E7EB"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, {BG_DARK} 0%, #0b0e13 100%);
  color: {TEXT};
}}
[data-testid="stSidebar"] {{
  background: #0b0e13;
}}
div.stButton > button:first-child {{
  background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
  color: black; border: 0; border-radius: 12px;
  padding: 0.7rem 1.05rem; transition: transform .06s ease-in-out;
  box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}}
div.stButton > button:first-child:hover {{ transform: translateY(-1px) scale(1.01); }}
.metric-card {{
  background: {CARD_BG}; border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px; padding: 1rem;
}}
.section-card {{
  background: {CARD_BG}; border: 1px solid rgba(255,255,255,0.06);
  border-radius: 18px; padding: 1.2rem 1.2rem .6rem 1.2rem; margin-bottom: 1rem;
}}
hr {{ border: none; height: 1px; background: rgba(255,255,255,0.08); }}
</style>
""", unsafe_allow_html=True)


LOTTIE_GEAR = {
  "v": "5.7.4","fr": 30,"ip": 0,"op": 90,"w": 200,"h": 200,"nm": "gear","ddd": 0,
  "assets": [],"layers": [{
    "ddd": 0,"ind": 1,"ty": 4,"nm": "gear","sr": 1,
    "ks": {"o":{"a":0,"k":100},"r":{"a":1,"k":[{"t":0,"s":[0]},{"t":90,"s":[360]}]},
           "p":{"a":0,"k":[100,100,0]},"a":{"a":0,"k":[0,0,0]},"s":{"a":0,"k":[100,100,100]}},
    "shapes": [{"ty":"el","p":{"a":0,"k":[0,0]},"s":{"a":0,"k":[120,120]},"nm":"circle","hd":False,
                "st":{"c":{"a":0,"k":[0.36,0.82,0.84,1]},"w":{"a":0,"k":8},"lc":1,"lj":1}}]
  }]
}

def header(logo_img):
    left, mid, right = st.columns([1.6, 1.1, 1])
    with left:
        st.markdown("## 🧊 HVAC TOU Simulation — CU-BEMS + NASA POWER")
        st.caption("**by Mohammed Ateeq**  •  **Supervisor:** Professor Alan Fung  •  Toronto Metropolitan University")
    with mid:
        if HAS_LOTTIE:
            st_lottie(LOTTIE_GEAR, height=110, speed=1, loop=True, quality="low")
    with right:
        if logo_img is not None:
            st.image(logo_img, caption="Toronto Metropolitan University", use_container_width=True)
        st.markdown(
            f"""<div class="metric-card" style="margin-top:.5rem;">
                <div style="font-size:0.9rem;opacity:0.8;">Scope</div>
                <div style="font-size:1.2rem;font-weight:700;">AC1 + AC4</div>
                <div style="font-size:0.9rem;opacity:0.8;margin-top:.5rem;">TOU editable • May–Aug</div>
            </div>""",
            unsafe_allow_html=True,
        )


SIM_MONTHS = {5:'May', 6:'June', 7:'July', 8:'August'}
MONTH_ORDER = ["May","June","July","August"]
ASHRAE_MIN, ASHRAE_MAX = 22.0, 29.0
PRESET_SCENARIOS = {
    "baseline (23–25°C)": (23.0, 25.0),
    "eco_mode (24–26°C)": (24.0, 26.0),
    "comfort_mode (23–24°C)": (23.0, 24.0),
    "aggressive_savings (25–27°C)": (25.0, 27.0),
    "precooling (22–24°C)": (22.0, 24.0),
    "relaxed (26–28°C)": (26.0, 28.0),
    "night_setback (27–29°C)": (27.0, 29.0),
}
TOU_DEFAULT = {"off_peak": 7.6, "mid_peak": 12.2, "on_peak": 15.8}

def ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns and not np.issubdtype(df["Date"].dtype, np.datetime64):
        with pd.option_context("mode.chained_assignment", None):
            try: df["Date"] = pd.to_datetime(df["Date"])
            except Exception: pass
    if "Hour" not in df.columns and "Date" in df.columns:
        df["Hour"] = df["Date"].dt.hour
    if "Month" not in df.columns and "Date" in df.columns:
        df["Month"] = df["Date"].dt.month
    if "DayOfWeek" not in df.columns and "Date" in df.columns:
        df["DayOfWeek"] = df["Date"].dt.dayofweek
    if "DayType" not in df.columns:
        df["DayType"] = np.where(df.get("DayOfWeek", 0) < 5, 1, 0)
    return df

def add_cyclical_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Hour" in df.columns:
        x = 2 * math.pi * (df["Hour"].astype(float) / 24.0)
        df["Hour_sin"] = np.sin(x)
        df["Hour_cos"] = np.cos(x)
    if "Month" in df.columns:
        x = 2 * math.pi * (df["Month"].astype(float) / 12.0)
        df["Month_sin"] = np.sin(x)
        df["Month_cos"] = np.cos(x)
    return df

def ensure_model_features(df: pd.DataFrame, model_feature_lists) -> pd.DataFrame:
    needed = set().union(*[set(flist) for flist in model_feature_lists])
    cyc_needed = {"Hour_sin","Hour_cos","Month_sin","Month_cos"}
    if len(needed.intersection(cyc_needed)) > 0:
        add_cyclical_columns(df)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.warning(f"{len(missing)} feature(s) missing; filled with 0 at predict time. Examples: {missing[:10]}")
    return df

def get_tou_price(hour: int, month: int, daytype: int, tou_prices: dict) -> float:
    
    if int(daytype) == 0:  # weekend
        return tou_prices["off_peak"]
    if 7 <= hour < 11 or 17 <= hour < 19:
        return tou_prices["mid_peak"] if 5 <= month <= 10 else tou_prices["on_peak"]
    elif 11 <= hour < 17:
        return tou_prices["on_peak"] if 5 <= month <= 10 else tou_prices["mid_peak"]
    else:
        return tou_prices["off_peak"]

def _pretty_scenario(name: str) -> str:
    n = name.replace("_", " ").strip()
    n = n[:1].upper() + n[1:]
    n = n.replace("(22 24", "(22–24").replace("(23 24", "(23–24") \
                 .replace("(23 25", "(23–25").replace("(24 26", "(24–26") \
                 .replace("(25 27", "(25–27").replace("(26 28", "(26–28") \
                 .replace("(27 29", "(27–29")
    if "°C" not in n: n = n.replace("C)", "°C)")
    return n

def _fmt_money(x):
    try: return f"${float(x):,.2f}"
    except Exception: return str(x)

def _sanitize_text(s: str) -> str:
    
    return (str(s).replace("\\", r"\\").replace("$", r"\$").replace("_", r"\_"))


def style_plotly_for_app(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
        legend_title_text="Scenario",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

def style_plotly_for_export(fig: go.Figure) -> go.Figure:
    
    fig_exp = go.Figure(fig)
    fig_exp.update_layout(
        template="plotly",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        legend_font_color="black",
    )
    fig_exp.update_yaxes(tickfont_color="black", title_font_color="black")
    fig_exp.update_xaxes(tickfont_color="black", title_font_color="black")
    return fig_exp


def make_grouped_cost_fig(df_per_ac):
    dfp = df_per_ac.copy()
    dfp["Month"] = pd.Categorical(dfp["Month"], categories=MONTH_ORDER, ordered=True)
    fig = px.bar(
        dfp, x="Month", y="Monthly_Cost_$", color="Scenario",
        barmode="group", category_orders={"Month": MONTH_ORDER},
        custom_data=["Monthly_Energy_kWh"]
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Scenario: %{legendgroup}<br>"
                      "Cost: $%{y:,.2f}<br>Energy: %{customdata[0]:,.0f} kWh<extra></extra>"
    )
    fig.update_layout(
        title="Monthly Electricity Cost (grouped by scenario)",
        yaxis_title="Cost ($)",
        xaxis_title="Month",
        bargap=0.15
    )
    fig.update_yaxes(separatethousands=True, tickprefix="$")
    return style_plotly_for_app(fig)

def make_grouped_energy_fig(df_per_ac):
    dfp = df_per_ac.copy()
    dfp["Month"] = pd.Categorical(dfp["Month"], categories=MONTH_ORDER, ordered=True)
    fig = px.bar(
        dfp, x="Month", y="Monthly_Energy_kWh", color="Scenario",
        barmode="group", category_orders={"Month": MONTH_ORDER},
        custom_data=["Monthly_Cost_$"]
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Scenario: %{legendgroup}<br>"
                      "Energy: %{y:,.0f} kWh<br>Cost: $%{customdata[0]:,.2f}<extra></extra>"
    )
    fig.update_layout(
        title="Monthly Energy (kWh) — grouped by scenario",
        yaxis_title="Energy (kWh)",
        xaxis_title="Month",
        bargap=0.15
    )
    fig.update_yaxes(separatethousands=True)
    return style_plotly_for_app(fig)


def fig_text_page(title, paragraphs, footer=None):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    y = 0.92
    plt.text(0.05, y, _sanitize_text(title), fontsize=20, fontweight="bold")
    y -= 0.06
    for p in paragraphs:
        if isinstance(p, list):
            for item in p:
                wrapped = textwrap.fill(f"• {_sanitize_text(item)}", 115)
                plt.text(0.06, y, wrapped, fontsize=12)
                y -= 0.05
            y -= 0.01
        else:
            wrapped = textwrap.fill(_sanitize_text(p), 120)
            plt.text(0.05, y, wrapped, fontsize=13)
            y -= 0.075
    if footer:
        plt.text(0.05, 0.04, _sanitize_text(footer), fontsize=9, color="gray")
    plt.axis("off"); return fig

def table_page(title, df, highlight_idx=0):
    df2 = df.copy()
    if "Total_Cost_$" in df2.columns:
        df2["Total_Cost_$"] = df2["Total_Cost_$"].map(_fmt_money)
    if "Savings_vs_Baseline_$" in df2.columns:
        df2["Savings_vs_Baseline_$"] = df2["Savings_vs_Baseline_$"].map(_fmt_money)
    if "Savings_%" in df2.columns:
        df2["Savings_%"] = df2["Savings_%"].map(lambda v: f"{float(v):.1f}%")
    if "Total_Energy_kWh" in df2.columns:
        df2["Total_Energy_kWh"] = df2["Total_Energy_kWh"].map(lambda v: f"{float(v):,.0f}")
    if "Scenario" in df2.columns:
        df2["Scenario"] = df2["Scenario"].apply(_pretty_scenario)

    fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=150)
    fig.patch.set_facecolor("white"); ax.axis("off")
    ax.text(0.02, 0.96, _sanitize_text(title), fontsize=20, fontweight="bold", va="top")

    table = ax.table(cellText=df2.values, colLabels=df2.columns,
                     cellLoc='center', colLoc='center', bbox=[0.02, 0.12, 0.96, 0.78])
    table.auto_set_font_size(False); table.set_fontsize(11)
    
    for c in range(df2.shape[1]):
        cell = table[(0, c)]
        cell.set_text_props(weight='bold'); cell.set_facecolor("#F2F2F2")
    
    if len(df2) > 0:
        for c in range(df2.shape[1]):
            cell = table[(1 + highlight_idx, c)]
            cell.set_text_props(weight='bold', color="#0b5394")
    return fig

def bars_from_series(title, series, ylabel):
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("Scenario")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    return fig

def title_page(pdf, header_img=None):
    fig = plt.figure(figsize=(11.69, 8.27), dpi=150)
    fig.patch.set_facecolor("white")
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.6, 0.25, 0.15], hspace=0.0)
    ax_title = fig.add_subplot(gs[0, :]); ax_title.axis("off")
    ax_title.text(0.05, 0.75, "HVAC TOU Scenario Simulator — AC1 & AC4",
                  fontsize=28, fontweight="bold", transform=ax_title.transAxes)
    ax_title.text(0.05, 0.55, "Toronto Metropolitan University • CU-BEMS + NASA POWER",
                  fontsize=16, transform=ax_title.transAxes)
    ax_title.text(0.05, 0.38, "by Mohammed Ateeq  •  Supervisor: Professor Alan Fung",
                  fontsize=14, transform=ax_title.transAxes)
    ax_title.text(0.05, 0.22, f"Report date: {datetime.now():%B %d, %Y}",
                  fontsize=12, color="gray", transform=ax_title.transAxes)
    ax_logo = fig.add_subplot(gs[1, 2]); ax_logo.axis("off")
    if header_img is not None:
        try:
            from PIL import Image
            img = Image.open(header_img) if not isinstance(header_img, str) else Image.open(header_img)
            ax_logo.imshow(img); ax_logo.axis("off")
        except Exception:
            pass
    pdf.savefig(fig); plt.close(fig)

def exec_summary(savings_tbl, overall_monthly):
    if len(savings_tbl) == 0:
        return ["No scenarios found."]
    best = savings_tbl.iloc[0]
    base_name = "baseline (23–25°C)" if "baseline (23–25°C)" in savings_tbl["Scenario"].values else savings_tbl["Scenario"].values[-1]
    lines = [
        "Objective: Compare electricity costs and energy across comfort bands for AC1 & AC4 under Ontario TOU pricing (May–Aug).",
        f"Best overall scenario: {_pretty_scenario(best['Scenario'])} with total cost {_fmt_money(best['Total_Cost_$'])}, "
        f"saving {_fmt_money(best['Savings_vs_Baseline_$'])} ({best['Savings_%']:.1f}%) vs {_pretty_scenario(base_name)}."
    ]
    try:
        base_month = overall_monthly[base_name]
        savings_by_month = {}
        for scen in overall_monthly.columns:
            if scen == base_name: continue
            savings_by_month[scen] = (base_month - overall_monthly[scen]).sum()
        if savings_by_month:
            top_scen = max(savings_by_month, key=savings_by_month.get)
            lines.append(f"Most impactful alternative overall: {_pretty_scenario(top_scen)}.")
    except Exception:
        pass
    lines.append("Note: AC1 & AC4 show meaningful temperature sensitivity; other units show minimal deltas.")
    return lines

def export_pdf(report_path, header_img, scenarios, tou, results, overall_monthly_cost, savings_tbl,
               figs_per_ac_plotly, totals_combined_df, monthly_energy_overall):
    offp, midp, onpk = tou
    with PdfPages(report_path) as pdf:
        
        title_page(pdf, header_img)

        
        pdf.savefig(fig_text_page("Executive Summary", exec_summary(savings_tbl, overall_monthly_cost))); plt.close()

        
        pdf.savefig(fig_text_page("About this Project", [
            ("We simulate electricity costs and energy for AC1 and AC4 under multiple thermal comfort scenarios using "
             "pre-trained LightGBM models on CU-BEMS data with NASA POWER weather. For each scenario, indoor "
             "temperature S1(degC) is clipped to a comfort band and hourly kW is priced with Ontario TOU rates. "
             "We aggregate to monthly totals (May–Aug)."),
            "We focus on AC1 & AC4 because they exhibit meaningful temperature sensitivity; scenario changes materially affect cost."
        ])); plt.close()

        scen_lines = [_pretty_scenario(n) for (n, _, _) in scenarios]
        tou_lines  = [f"Off-peak: {offp} c/kWh", f"Mid-peak: {midp} c/kWh", f"On-peak: {onpk} c/kWh"]
        pdf.savefig(fig_text_page("Scenarios & TOU Settings", [
            "Selected scenarios:", scen_lines,
            "TOU pricing (cents/kWh):", tou_lines,
            "Months included: May, June, July, August"
        ])); plt.close()

        
        for (ac, fig_cost_app, fig_kwh_app) in figs_per_ac_plotly:
            try:
                os.makedirs("exports/plots", exist_ok=True)
                img_cost = f"exports/plots/{ac}_cost_grouped.png"
                img_kwh  = f"exports/plots/{ac}_energy_grouped.png"

                
                style_plotly_for_export(go.Figure(fig_cost_app)).write_image(img_cost, scale=2, engine="kaleido")
                style_plotly_for_export(go.Figure(fig_kwh_app)).write_image(img_kwh, scale=2, engine="kaleido")

                
                fig = plt.figure(figsize=(11.69, 8.27), dpi=150)
                fig.patch.set_facecolor("white")
                ax_title = fig.add_axes([0.05, 0.94, 0.90, 0.05]); ax_title.axis("off")
                ax_title.text(0.0, 0.5, f"{ac} — Monthly Cost & Energy (Grouped)", fontsize=16, weight="bold", va="center")
                ax1 = fig.add_axes([0.07, 0.52, 0.86, 0.40]); ax1.axis("off")
                ax2 = fig.add_axes([0.07, 0.06, 0.86, 0.40]); ax2.axis("off")
                import matplotlib.image as mpimg
                ax1.imshow(mpimg.imread(img_cost)); ax2.imshow(mpimg.imread(img_kwh))
                pdf.savefig(fig); plt.close(fig)
            except Exception:
                pdf.savefig(fig_text_page(f"{ac} — Charts", ["Plotly images unavailable; charts omitted."])); plt.close()

        
        pdf.savefig(table_page("Total Energy & Cost by Scenario — AC1+AC4", totals_combined_df, highlight_idx=0)); plt.close()

        
        try:
            fig_e = plt.figure(figsize=(11.69, 8.27), dpi=150)
            fig_e.patch.set_facecolor("white")
            ax = fig_e.add_axes([0.08, 0.16, 0.85, 0.75])
            monthly_energy_overall.plot(kind="bar", ax=ax)
            ax.set_title("Monthly Energy (kWh) — AC1+AC4, grouped by scenario")
            ax.set_ylabel("Energy (kWh)"); ax.set_xlabel("Month")
            ax.legend(title="Scenario", fontsize=9)
            pdf.savefig(fig_e); plt.close(fig_e)
        except Exception:
            pdf.savefig(fig_text_page("Monthly Energy (kWh) — AC1+AC4", ["Chart unavailable."])); plt.close()

        
        try:
            tbl = monthly_energy_overall.T
            tbl["Total_Energy_kWh"] = tbl.sum(axis=1)
            tbl = tbl.reset_index().rename(columns={"index":"Scenario"})
            pdf.savefig(table_page("Monthly Energy (kWh) — Table (AC1+AC4)", tbl, highlight_idx=0)); plt.close()
        except Exception:
            pdf.savefig(fig_text_page("Monthly Energy Table — AC1+AC4", ["Table unavailable."])); plt.close()

        
        pdf.savefig(fig_text_page("Dataset Limitations & Impact on Models", [
            ("Low S1(degC) variance in most zones — clipping to comfort bands barely changes inputs, so predicted load "
             "is insensitive to scenarios. This is why only AC1/AC4 show clear effects."),
            ("Zero-inflated targets — many ACs are mostly off; predictors learn near-zero baselines and temperature features "
             "have limited leverage."),
            ("Temporal alignment — any timestamp drift or DST mismatch between CU-BEMS and NASA POWER weakens weather–load linkage."),
            "Seasonal scope — models cover May–Aug; shoulder/winter behavior requires retraining.",
            "Sensor gaps/noise — missing humidity/illuminance reduce feature richness and explainability.",
            "Lag dominance — where temperature isn’t informative, lag features dominate, limiting counterfactual sensitivity."
        ])); plt.close()
        pdf.savefig(fig_text_page("Future Work — Toward RL-Based Control", [
            ("Model-based RL: Use the trained AC predictors as the environment; actions are setpoint bands. "
             "Reward = −(TOU cost) − discomfort penalties (+ optional demand/carbon terms)."),
            "Algorithms: PPO or DQN with action masking; train with randomized weather/occupancy to avoid overfitting.",
            "Thermal dynamics: Add RC thermal model or surrogate to enable anticipatory precooling and peak shaving.",
            "Safety & evaluation: Off-policy replay and counterfactual tests before any live deployment.",
            "Explainability: SHAP for predictors and policy saliency to justify actions in peak periods."
        ])); plt.close()

st.sidebar.subheader("Paths & Branding")

data_path = st.sidebar.text_input("Dataset CSV", r"C:\Users\MOHAM\CU_BEMS_preprocessed_wide_v2.csv")
ac1_path  = st.sidebar.text_input("AC1 model",  r"C:\Users\MOHAM\saved_models_timeaware_v3\AC1(kW)_model.pkl")
ac4_path  = st.sidebar.text_input("AC4 model",  r"C:\Users\MOHAM\saved_models_timeaware_v3\AC4(kW)_model.pkl")

logo_path = st.sidebar.text_input("Header image (TMU logo path, optional)", "")
logo_upload = st.sidebar.file_uploader("Or upload header image", type=["png","jpg","jpeg","webp"])

st.sidebar.subheader("TOU (cents/kWh)")
offp = st.sidebar.number_input("Off-peak", value=float(TOU_DEFAULT["off_peak"]), step=0.1)
midp = st.sidebar.number_input("Mid-peak", value=float(TOU_DEFAULT["mid_peak"]), step=0.1)
onpk = st.sidebar.number_input("On-peak",  value=float(TOU_DEFAULT["on_peak"]),  step=0.1)
TOU = {"off_peak": offp, "mid_peak": midp, "on_peak": onpk}

header_img = None
if logo_upload is not None:
    header_img = logo_upload
elif logo_path and os.path.exists(logo_path):
    header_img = logo_path


header(header_img)

st.markdown(
    f"""
<div class="section-card">
  <h3>About this Project</h3>
  <p>
    This app predicts hourly AC load (kW) for <b>AC1</b> and <b>AC4</b> using pre-trained LightGBM models on CU-BEMS data merged with
    NASA POWER weather. It computes electricity <b>cost ($)</b> using Ontario <b>TOU</b> rates and aggregates both <b>cost</b> and
    <b>energy (kWh)</b> by month for user-selected comfort scenarios. We focus on AC1 & AC4 because they exhibit clear
    temperature sensitivity.
  </p>
</div>
""",
    unsafe_allow_html=True,
)


st.markdown("### 1) Load Dataset & Models (auto: AC1 & AC4)")
c1, c2, c3 = st.columns([1.4, 1.1, 1])
with c1:
    df = None
    upl_csv = st.file_uploader("Or upload dataset CSV", type="csv", key="df_up")
    if upl_csv is not None:
        try:
            df = pd.read_csv(upl_csv)
            st.success("Uploaded dataset loaded.")
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
    elif data_path and os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path, parse_dates=["Date"])
            st.info(f"Loaded dataset: {os.path.basename(data_path)}")
        except Exception as e:
            st.error(f"Failed to read dataset: {e}")
    else:
        st.warning("Provide a dataset path or upload a CSV.")

with c2:
    st.write("**Models in use:**")
    st.code(f"AC1: {ac1_path}\nAC4: {ac4_path}")
with c3:
    st.caption("Ensure both bundles contain: {'model', 'features'}")

if df is not None:
    df = ensure_time_cols(df)
    df = df[df["Month"].isin(list(SIM_MONTHS.keys()))].copy()

ac1_bundle = joblib.load(ac1_path) if os.path.exists(ac1_path) else None
ac4_bundle = joblib.load(ac4_path) if os.path.exists(ac4_path) else None
if (ac1_bundle is None) or (ac4_bundle is None):
    st.warning("Models not found — please check the paths above.")
    st.stop()

if df is not None:
    df = ensure_model_features(df, [ac1_bundle["features"], ac4_bundle["features"]])
    st.caption(f"Dataset shape after month filter & feature prep: {df.shape}")

st.markdown("### 2) Select Multiple Scenarios")
left, right = st.columns([1.3, 1])
with left:
    chosen_presets = st.multiselect(
        "Pick preset scenarios",
        list(PRESET_SCENARIOS.keys()),
        default=["baseline (23–25°C)", "eco_mode (24–26°C)", "precooling (22–24°C)"]
    )
with right:
    custom_toggle = st.checkbox("Add a Custom Scenario")
    custom_name, custom_low, custom_high, custom_valid = None, None, None, False
    if custom_toggle:
        st.write("**Custom (ASHRAE gate: 22–29 °C)**")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            custom_low = st.number_input("Lower (°C)", value=23.0, step=0.5)
        with cc2:
            custom_high = st.number_input("Upper (°C)", value=25.0, step=0.5)
        with cc3:
            custom_name = st.text_input("Scenario name", value="custom_23.0_25.0")
        errs = []
        if not (ASHRAE_MIN <= custom_low <= ASHRAE_MAX): errs.append(f"Lower must be {ASHRAE_MIN}–{ASHRAE_MAX}°C")
        if not (ASHRAE_MIN <= custom_high <= ASHRAE_MAX): errs.append(f"Upper must be {ASHRAE_MIN}–{ASHRAE_MAX}°C")
        if custom_low >= custom_high: errs.append("Lower must be strictly less than upper")
        if not custom_name.strip(): errs.append("Scenario name required")
        if errs:
            st.error("Custom scenario invalid:\n- " + "\n- ".join(errs))
        else:
            custom_valid = True


scenarios = []
for name in chosen_presets:
    low, high = PRESET_SCENARIOS[name]
    scenarios.append((name, low, high))
if custom_toggle and custom_valid:
    scenarios.append((custom_name.strip(), float(custom_low), float(custom_high)))
if "baseline (23–25°C)" not in [s[0] for s in scenarios]:
    scenarios.insert(0, ("baseline (23–25°C)", 23.0, 25.0))

st.markdown("---")


def run_scenario(df, ac_col, model_bundle, tou_prices, months_set, scenario_label, low, high):
    model = model_bundle["model"]
    feats = model_bundle["features"]
    df_m = df.copy()

    if "S1(degC)" in df_m.columns and low is not None and high is not None:
        df_m["S1(degC)"] = df_m["S1(degC)"].clip(lower=low, upper=high)

    X = df_m[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_pred_kw = model.predict(X)

    prices_c_kwh = df_m.apply(
        lambda r: get_tou_price(int(r.Hour), int(r.Month), int(r.DayType), tou_prices), axis=1
    ).values
    hourly_cost_dollars = (y_pred_kw * prices_c_kwh) / 100.0

    df_m["__cost__"] = hourly_cost_dollars
    df_m["__energy_kWh__"] = y_pred_kw

    g = df_m[df_m["Month"].isin(months_set)].groupby("Month")
    monthly_cost = g["__cost__"].sum()
    monthly_kwh  = g["__energy_kWh__"].sum()

    rows = []
    for m in sorted(monthly_cost.index):
        rows.append([
            ac_col,
            SIM_MONTHS.get(int(m), str(m)),
            scenario_label,
            float(monthly_cost.loc[m]),
            float(monthly_kwh.loc[m]),
        ])
    return pd.DataFrame(rows, columns=[
        "AC_Unit","Month","Scenario","Monthly_Cost_$","Monthly_Energy_kWh"
    ])

st.markdown("### 3) Run Simulations & Export")
go_btn = st.button("🚀 Run for Selected Scenarios (AC1 & AC4)")

def ensure_dirs():
    os.makedirs("exports/plots", exist_ok=True)
    os.makedirs("exports", exist_ok=True)

def ensure_report_path():
    ensure_dirs()
    return os.path.join("exports", "HVAC_TOUsim_AC1_AC4_Report.pdf")

if go_btn:
    if df is None:
        st.error("Dataset not loaded.")
        st.stop()

    ensure_dirs()
    months_set = set(SIM_MONTHS.keys())
    all_rows = []

    for (label, low, high) in scenarios:
        res1 = run_scenario(df, "AC1(kW)", ac1_bundle, TOU, months_set, label, low, high)
        res4 = run_scenario(df, "AC4(kW)", ac4_bundle, TOU, months_set, label, low, high)
        all_rows += [res1, res4]

    results = pd.concat(all_rows, ignore_index=True)

    st.subheader("Per-AC Monthly Cost & Energy (Grouped — clearer comparison)")

    figs_per_ac_plotly = []
    for ac in ["AC1(kW)", "AC4(kW)"]:
        sub = results[results["AC_Unit"] == ac].copy()
        sub["Month"] = pd.Categorical(sub["Month"], categories=MONTH_ORDER, ordered=True)

        st.markdown(f"**{ac}**")
        fig_cost = make_grouped_cost_fig(sub)
        st.plotly_chart(fig_cost, use_container_width=True)

        fig_kwh  = make_grouped_energy_fig(sub)
        st.plotly_chart(fig_kwh, use_container_width=True)

        try:
            style_plotly_for_export(go.Figure(fig_cost)).write_image(
                f"exports/plots/{ac}_monthly_costs_grouped.png", scale=2, engine="kaleido"
            )
            style_plotly_for_export(go.Figure(fig_kwh)).write_image(
                f"exports/plots/{ac}_monthly_energy_grouped.png", scale=2, engine="kaleido"
            )
        except Exception:
            pass

        figs_per_ac_plotly.append((ac, fig_cost, fig_kwh))

    st.subheader("Overall Monthly Cost by Scenario (AC1 + AC4)")
    overall_cost = results.groupby(["Scenario", "Month"])["Monthly_Cost_$"].sum().unstack().T.fillna(0)
    overall_cost = overall_cost.reindex(MONTH_ORDER)
    st.bar_chart(overall_cost)

    st.subheader("Overall Monthly Energy (kWh) by Scenario (AC1 + AC4)")
    overall_energy = results.groupby(["Scenario", "Month"])["Monthly_Energy_kWh"].sum().unstack().T.fillna(0)
    overall_energy = overall_energy.reindex(MONTH_ORDER)
    st.bar_chart(overall_energy)
    monthly_energy_overall = overall_energy.copy()

    baseline_key = "baseline (23–25°C)"
    if baseline_key not in overall_cost.columns:
        baseline_key = overall_cost.columns[0]

    totals_cost = results.groupby("Scenario")["Monthly_Cost_$"].sum().sort_values()
    totals_kwh  = results.groupby("Scenario")["Monthly_Energy_kWh"].sum().reindex(totals_cost.index)

    base_total_cost = float(totals_cost.loc[baseline_key])

    savings_tbl = pd.DataFrame({
        "Scenario": totals_cost.index,
        "Total_Energy_kWh": totals_kwh.values,
        "Total_Cost_$": totals_cost.values,
        "Savings_vs_Baseline_$": [base_total_cost - v for v in totals_cost.values],
        "Savings_%": [
            0.0 if base_total_cost == 0 else (base_total_cost - v) / base_total_cost * 100
            for v in totals_cost.values
        ],
    }).sort_values("Total_Cost_$", ascending=True)

    best_scen = savings_tbl.iloc[0]["Scenario"]

    def kpis_for_ac(ac):
        ac_df = results[results["AC_Unit"] == ac]
        total_kwh_by_scen  = ac_df.groupby("Scenario")["Monthly_Energy_kWh"].sum()
        total_cost_by_scen = ac_df.groupby("Scenario")["Monthly_Cost_$"].sum()

        base_cost = float(total_cost_by_scen.loc[baseline_key])
        best_cost = float(total_cost_by_scen.loc[best_scen])
        best_kwh  = float(total_kwh_by_scen.loc[best_scen])

        return best_kwh, best_cost, base_cost - best_cost, (0 if base_cost==0 else (base_cost-best_cost)/base_cost*100)

    st.subheader("KPIs — Best Scenario vs Baseline (per AC)")
    for ac in ["AC1(kW)", "AC4(kW)"]:
        best_kwh, best_cost, sav_dollars, sav_pct = kpis_for_ac(ac)
        st.markdown(
            f"""<div class="metric-card">
                <div style="font-size:1.0rem;margin-bottom:.35rem;"><b>{ac}</b> — {_pretty_scenario(best_scen)}</div>
                <div style="display:flex; gap:1rem; flex-wrap:wrap;">
                    <div><b>Total Energy:</b> {best_kwh:,.0f} kWh</div>
                    <div><b>Total Cost:</b> {_fmt_money(best_cost)}</div>
                    <div><b>Savings vs Baseline:</b> {_fmt_money(sav_dollars)} ({sav_pct:.1f}%)</div>
                </div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("#### Savings vs Baseline — By Month (AC1 + AC4) — Cost")
    base_monthly = overall_cost[baseline_key]
    comp_cost = overall_cost.copy()
    for col in comp_cost.columns:
        comp_cost[col] = base_monthly - comp_cost[col]
    st.bar_chart(comp_cost)

    st.download_button(
        "Download raw results (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="multi_scenario_results_ac1_ac4.csv",
        mime="text/csv"
    )

    report_path = ensure_report_path()

    totals_combined_df = savings_tbl.reset_index(drop=True).copy()
    export_pdf(
        report_path=report_path,
        header_img=header_img,
        scenarios=scenarios,
        tou=(offp, midp, onpk),
        results=results,
        overall_monthly_cost=overall_cost,
        savings_tbl=savings_tbl,
        figs_per_ac_plotly=figs_per_ac_plotly,
        totals_combined_df=totals_combined_df,
        monthly_energy_overall=monthly_energy_overall
    )
    st.success(f"PDF report saved: {report_path}")
    with open(report_path, "rb") as f:
        st.download_button("Download PDF Report",
                           data=f.read(),
                           file_name="HVAC_TOUsim_AC1_AC4_Report.pdf",
                           mime="application/pdf")

st.markdown("---")
st.markdown(
    f"""
<div class="section-card">
  <h3>Dataset Limitations & How They Affected Models</h3>
  <ul>
    <li><b>Low S1(degC) variance:</b> For many zones, indoor temperature barely moves, so clipping to comfort bands doesn’t change model inputs enough to alter predicted load. Scenario effects are therefore small outside AC1/AC4.</li>
    <li><b>Zero-inflated targets:</b> Several ACs are mostly off; models learn near-zero baselines well but have limited leverage from temperature or weather features.</li>
    <li><b>Time alignment sensitivity:</b> Misaligned or noisy timestamps between CU-BEMS and NASA POWER (e.g., DST, sensor polling) reduce apparent weather–load linkages, lowering feature importances for temperature variables.</li>
    <li><b>Seasonal scope:</b> The focus on May–Aug means models capture summer behavior; extension to shoulders/winter requires retraining and may change scenario economics.</li>
    <li><b>Feature dominance by lags:</b> Where temperature isn’t informative, lag features dominate — useful for forecasting but weak for “what-if” setpoint analysis.</li>
    <li><b>Sensor quality & gaps:</b> Missing/erratic humidity or illuminance readings force imputation/dropping, slightly capping achievable accuracy and interpretability.</li>
  </ul>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="section-card">
  <h3>Future Work — Practical Path to RL</h3>
  <ul>
    <li><b>Model-based RL environment:</b> Use the trained LightGBM predictors as a dynamics model; state = weather + time + indoor signals, actions = setpoint band.</li>
    <li><b>Reward:</b> negative electricity cost (TOU) − weighted discomfort penalty; optional demand/carbon terms.</li>
    <li><b>Algorithms:</b> PPO or DQN with discrete actions (OFF/LOW/MED/HIGH or comfort ranges), action masking for infeasible states.</li>
    <li><b>Thermal inertia:</b> Add a simple RC thermal model or a data-driven surrogate to enable precooling & load shifting.</li>
    <li><b>Safety & evaluation:</b> Policy constraints on comfort hours; offline evaluation with counterfactual replay before any live trial.</li>
    <li><b>Explainability:</b> SHAP on predictors + policy saliency to justify actions in peak periods.</li>
  </ul>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("© Toronto Metropolitan University — HVAC TOU Scenario Simulator • by Mohammed Ateeq • Supervisor: Professor Alan Fung")
