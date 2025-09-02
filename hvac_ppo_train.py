import tkinter as tk
from tkinter import messagebox
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from hvac_env import HVACEnv

DATA_PATH = r"C:\Users\MOHAM\CU_BEMS_modeling_dataset_with_orientation.csv"
MODEL_DIR = r"C:\Users\MOHAM\saved_models"

def run_training(selected_model):
    model_path = os.path.join(MODEL_DIR, f"{selected_model}(kW)_model.pkl")
    if not (os.path.exists(DATA_PATH) and os.path.exists(model_path)):
        messagebox.showerror("Missing Files", "Dataset or model not found.")
        return
    env = HVACEnv(DATA_PATH, model_path)
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    model.save(f"ppo_{selected_model}")

root = tk.Tk()
root.title("HVAC PPO Trainer")
tk.Label(root, text="Select AC Model").pack()
models = [f"AC{i}" for i in range(1, 16)]
var = tk.StringVar(root)
var.set(models[0])
tk.OptionMenu(root, var, *models).pack()
tk.Button(root, text="Start Training", command=lambda: run_training(var.get())).pack()
root.mainloop()
