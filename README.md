# Physics-Informed-RL-HVAC-system-for-smart-buildings

# Physics-Informed RL HVAC System for Smart Buildings

## 📌 Overview
This project explores **Reinforcement Learning (RL)** for adaptive HVAC control in smart buildings.  
It combines the **CU-BEMS dataset** with **NASA POWER weather data**, predictive modeling with **LightGBM**, and a **Streamlit app** for TOU pricing simulation.

The focus is on **AC1 & AC4** units, which showed the strongest temperature sensitivity.  
An interactive dashboard enables scenario testing (Baseline, Eco, Aggressive) and generates automated PDF reports.

---

## 📂 Repository Structure
- `cubems_merge.py` → Merge CU-BEMS building data  
- `nasa_cubems_merge.py` → Merge CU-BEMS with NASA POWER weather data  
- `preprocessing.py` → Data cleaning & feature engineering  
- `baseline_modeling.py` → LightGBM baseline modeling (AC1, AC4)  
- `tou_simulation.py` → TOU simulation script  
- `streamlit_app_mrp.py` → Streamlit dashboard for scenario simulation  
- `hvac_env.py` / `ppo_train.py` → (Future work) RL environment and PPO training  

---

## ⚙️ Installation
```bash
git clone https://github.com/Ateeq-Science/Physics-Informed-RL-HVAC-system-for-smart-buildings.git
cd Physics-Informed-RL-HVAC-system-for-smart-buildings
pip install -r requirements.txt
