import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import os
import joblib
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

try:
    import tensorflow as tf
except ImportError:
    tf = None

class EnergyPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Energy Consumption Predictor")
        
        self.root.geometry("900x700")
        self.root.configure(bg="white")
        
        # --- NEW: Bind the close button to our strict kill protocol ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        title_label = ttk.Label(root, text="Energy Consumption Prediction", font=("Helvetica", 18, "bold"), background="white")
        title_label.pack(pady=15)
        
        input_frame = ttk.Frame(root, padding="10")
        input_frame.pack(fill="x", padx=30)
        
        self.entries = {}
        
        countries = ['Germany', 'France', 'Netherlands', 'Italy', 'Spain', 'Sweden', 'Norway',
                     'Poland', 'Turkey', 'United Kingdom', 'United States', 'Canada', 'Brazil',
                     'India', 'China', 'Japan', 'Australia', 'South Africa', 'Mexico', 'Indonesia']
        months = [str(i) for i in range(1, 13)]
        days = ['0', '1', '2', '3', '4', '5', '6'] 
                     
        fields = [
            ("Country", "country", countries, None),
            ("Month (1-12)", "month", months, None),
            ("Day of Week (0=Mon, 6=Sun)", "day_of_week", days, None),
            ("Avg Temperature (°C)", "avg_temperature", "entry", "25.0"),
            ("Humidity (%)", "humidity", "entry", "60.0"),
            ("CO2 Emission", "co2_emission", "entry", "400.5"),
            ("Renewable Share (%)", "renewable_share", "entry", "15.0"),
            ("Urban Population (%)", "urban_population", "entry", "75.0"),
            ("Industrial Activity Index", "industrial_activity_index", "entry", "70.0"),
            ("Energy Price (USD)", "energy_price", "entry", "115.0")
        ]
        
        row, col = 0, 0
        for label_text, field_name, field_type, default_val in fields:
            field_container = ttk.Frame(input_frame)
            field_container.grid(row=row, column=col, padx=20, pady=8, sticky="ew")
            input_frame.columnconfigure(col, weight=1) 
            
            lbl = ttk.Label(field_container, text=label_text)
            lbl.pack(anchor="w", pady=(0, 2))
            
            if isinstance(field_type, list):
                entry = ttk.Combobox(field_container, values=field_type, state="readonly")
                entry.set(field_type[0])
            else:
                entry = ttk.Entry(field_container)
                if default_val:
                    entry.insert(0, default_val)
            
            entry.pack(fill="x")
            self.entries[field_name] = entry
            
            col += 1
            if col > 1: 
                col = 0
                row += 1
                
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)
        
        self.predict_btn = ttk.Button(btn_frame, text="Predict Energy Demand", command=self.predict)
        self.predict_btn.pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Clear Fields", command=self.clear_fields).pack(side="left", padx=10)
        
        self.result_label = ttk.Label(root, text="", font=("Helvetica", 16, "bold"), foreground="#27ae60", background="white")
        self.result_label.pack(pady=(5, 5))

        self.graph_frame = tk.Frame(root, bg="white")
        self.graph_frame.pack(fill="both", expand=True, padx=30, pady=10)
        self.canvas_widget = None 

        self.load_models()
        self.load_historical_data() 

    def load_historical_data(self):
        file_path = "Climate_Energy_Consumption_Dataset_2020_2024.csv"
        if not os.path.exists(file_path):
            file_path = "Climate & Energy Consumption 2020 - 2024.csv"
            if not os.path.exists(file_path):
                file_path = "../Climate_Energy_Consumption_Dataset_2020_2024.csv"
        try:
            self.hist_df = pd.read_csv(file_path)
            self.hist_df['date'] = pd.to_datetime(self.hist_df['date'])
            self.hist_df['month'] = self.hist_df['date'].dt.month
        except Exception:
            self.hist_df = None 

    def load_models(self):
        if tf is None:
            messagebox.showerror("Dependency Error", "TensorFlow is not installed.")
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "..", "Model", "saved_models")
        
        try:
            self.model = tf.keras.models.load_model(os.path.join(model_dir, 'lstm_energy_model.h5'))
            self.scaler_X = joblib.load(os.path.join(model_dir, 'scaler_X.pkl'))
            self.scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        except Exception as e:
            messagebox.showwarning("Model Load Error", f"Could not load AI models.\nError: {e}")
            self.predict_btn.config(state="disabled")

    def predict(self):
        try:
            self.result_label.config(text="Running LSTM Inference... Please wait.", foreground="#e74c3c")
            self.predict_btn.config(state="disabled")
            self.root.update_idletasks()
            
            country = self.entries['country'].get()
            month = int(self.entries['month'].get())
            day_of_week = int(self.entries['day_of_week'].get())
            temp = float(self.entries['avg_temperature'].get())
            humidity = float(self.entries['humidity'].get())
            co2 = float(self.entries['co2_emission'].get())
            renew = float(self.entries['renewable_share'].get())
            urban = float(self.entries['urban_population'].get())
            indus = float(self.entries['industrial_activity_index'].get())
            price = float(self.entries['energy_price'].get())
            
            month_sin = math.sin(2 * math.pi * month / 12)
            month_cos = math.cos(2 * math.pi * month / 12)
            country_encoded = self.label_encoder.transform([country])[0]
            
            feature_names = [
                'avg_temperature', 'humidity', 'co2_emission', 'renewable_share', 
                'urban_population', 'industrial_activity_index', 'energy_price', 
                'day_of_week', 'month_sin', 'month_cos', 'country_encoded'
            ]
            features_df = pd.DataFrame([[
                temp, humidity, co2, renew, urban, indus, price, 
                day_of_week, month_sin, month_cos, country_encoded
            ]], columns=feature_names)
            
            scaled_features = self.scaler_X.transform(features_df)
            
            TIME_STEPS = 7
            sequence = np.tile(scaled_features, (TIME_STEPS, 1)) 
            lstm_input = np.expand_dims(sequence, axis=0)        
            
            scaled_prediction = self.model.predict(lstm_input)
            
            scaled_pred_df = pd.DataFrame(scaled_prediction, columns=['energy_consumption'])
            actual_prediction = self.scaler_y.inverse_transform(scaled_pred_df)[0][0]
            
            final_text = f"Predicted Demand: {actual_prediction:,.2f} kWh"
            self.result_label.config(text=final_text, foreground="#27ae60")
            self.predict_btn.config(state="normal")
            
            self.plot_prediction(actual_prediction, country, month)
            
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all fields are filled with valid numbers.")
            self.result_label.config(text="")
            self.predict_btn.config(state="normal")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred:\n{e}")
            self.result_label.config(text="")
            self.predict_btn.config(state="normal")

    def plot_prediction(self, predicted_val, country, month):
        if self.canvas_widget:
            self.canvas_widget.destroy()

        baseline_val = 0
        if self.hist_df is not None:
            mask = (self.hist_df['country'] == country) & (self.hist_df['month'] == month)
            filtered_data = self.hist_df[mask]
            if not filtered_data.empty:
                baseline_val = filtered_data['energy_consumption'].mean()

        fig, ax = plt.subplots(figsize=(6, 3))
        
        categories = ['Historical Avg', 'AI Prediction']
        values = [baseline_val, predicted_val]
        colors = ['#95a5a6', '#27ae60'] 
        
        bars = ax.bar(categories, values, color=colors, width=0.5)
        
        ax.set_title(f"Demand Context: {country} in Month {month}", fontsize=12, pad=10)
        ax.set_ylabel("Energy (kWh)", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), 
                    f"{yval:,.0f}", ha='center', va='bottom', fontweight='bold')

        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    def clear_fields(self):
        for entry in self.entries.values():
            if isinstance(entry, ttk.Combobox):
                entry.current(0)
            else:
                entry.delete(0, tk.END)
        self.result_label.config(text="")
        if self.canvas_widget:
            self.canvas_widget.destroy()

    # --- NEW: Strict Kill Protocol ---
    def on_closing(self):
        """Forces matplotlib and TensorFlow to release memory and terminates the script."""
        plt.close('all') 
        self.root.quit()
        self.root.destroy()
        sys.exit(0) 

if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyPredictionApp(root)
    root.mainloop()