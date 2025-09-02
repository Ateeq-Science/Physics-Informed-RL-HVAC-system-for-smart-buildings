# Physics-Informed RL HVAC System for Smart Buildings

![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-brightgreen)
![LightGBM](https://img.shields.io/badge/lightgbm-4.6-orange)
![Status](https://img.shields.io/badge/status-active-success)

## 📖 Project Overview
This project explores **Reinforcement Learning (RL)** and **simulation-based approaches** for optimizing Heating, Ventilation, and Air Conditioning (HVAC) systems in smart buildings.  
It leverages the **CU-BEMS dataset** (Chulalongkorn University Building Energy Management System) merged with **NASA POWER weather data** to model energy consumption and evaluate cost savings under Ontario **Time-of-Use (TOU)** pricing.

We focus on **AC1 and AC4 units**, which demonstrated strong temperature sensitivity, and simulate multiple comfort–cost trade-off scenarios.

---

## 🚀 Features
- Data preprocessing pipeline for CU-BEMS and NASA POWER integration.  
- LightGBM-based predictive modeling of AC load.  
- **Streamlit App** with interactive scenario simulation.  
- TOU pricing simulation (Ontario Energy Board).  
- Automatic PDF report generation with cost & energy KPIs.  
- Ready for future RL integration (`ppo_train.py`, `hvac_env.py`).  

---

## 📂 Repository Structure
📦 Physics-Informed-RL-HVAC-system-for-smart-buildings
┣ 📜 1_cubems_data_merging.py
┣ 📜 2_nasa_cubems_merging.py
┣ 📜 3_preprocessing.py
┣ 📜 4_baseline_modelling.py
┣ 📜 5_tou_simulation.py
┣ 📜 6_streamlit_app.py
┣ 📜 hvac_env.py # future RL environment (not finalized)
┣ 📜 ppo_train.py # future RL PPO trainer (not finalized)
┣ 📜 requirements.txt
┣ 📜 README.md

yaml
Copy code

---

🖥️ Installation
Clone this repository:
git clone https://github.com/Ateeq-Science/Physics-Informed-RL-HVAC-system-for-smart-buildings.git
cd Physics-Informed-RL-HVAC-system-for-smart-buildings

Install dependencies:
pip install -r requirements.txt
Run the Streamlit app:

streamlit run 6_streamlit_app.py
📊 Results
Baseline scenario (23–25 °C) compared with multiple comfort bands.

AC1 & AC4 showed up to 49.7% cost savings under Aggressive Savings mode.

Streamlit dashboards provide interactive monthly cost and energy visualizations.

PDF reports summarize KPIs, scenario comparisons, and dataset limitations.

🔮 Future Work
Extend RL integration using PPO and DQN.

Multi-agent HVAC optimization across zones.

Incorporate renewable energy and demand response signals.

Real-world deployment and occupant feedback integration.

📚 References
ASHRAE (2020). ANSI/ASHRAE Standard 55.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM. NeurIPS.

Ontario Energy Board (2023). Time-of-Use Pricing.

Wei, T., Wang, Y., & Zhu, Q. (2017). Deep RL for HVAC Control. DAC.

Zhang, Z., Lam, K. P., & Hong, T. (2021). RL for Energy Efficiency. Building and Environment.

👤 Author
Mohammed Ateeq
M.Sc. Data Science & Analytics, Toronto Metropolitan University
Supervisor: Professor Alan Fung
