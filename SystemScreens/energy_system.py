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
        self.root.title("Intelligent Energy Demand Predictor")
        
        # Switched to a standard Widescreen Dashboard resolution
        self.root.geometry("1280x680")
        self.root.configure(bg="white")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        # --- LAYOUT WIDGETS (Side-by-Side) ---
        # Left Panel: Controls (Inputs & Buttons)
        self.left_panel = tk.Frame(root, bg="#f8f9fa", width=400, bd=1, relief="ridge")
        self.left_panel.pack(side="left", fill="y", padx=(10, 0), pady=10)
        self.left_panel.pack_propagate(False) # Prevents the panel from shrinking
        
        # Right Panel: Analytics (Text Result & Charts)
        self.right_panel = tk.Frame(root, bg="white")
        self.right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # --- LEFT PANEL: CONTROL CENTER ---
        title_label = ttk.Label(self.left_panel, text="Input Parameters", font=("Helvetica", 16, "bold"), background="#f8f9fa")
        title_label.pack(pady=(15, 10))
        
        input_frame = ttk.Frame(self.left_panel, style="TFrame")
        input_frame.pack(fill="x", padx=20)
        
        self.entries = {}
        countries = ['Germany', 'France', 'Netherlands', 'Italy', 'Spain', 'Sweden', 'Norway',
                     'Poland', 'Turkey', 'United Kingdom', 'United States', 'Canada', 'Brazil',
                     'India', 'China', 'Japan', 'Australia', 'South Africa', 'Mexico', 'Indonesia']
        months = [str(i) for i in range(1, 13)]
        days = ['0', '1', '2', '3', '4', '5', '6'] 
                     
        self.input_features = [
            ("Avg Temp (°C)", "avg_temperature", "25.0"),
            ("Humidity (%)", "humidity", "60.0"),
            ("CO2 Emission", "co2_emission", "400.5"),
            ("Renewable Share (%)", "renewable_share", "15.0"),
            ("Urban Pop (%)", "urban_population", "75.0"),
            ("Industrial Index", "industrial_activity_index", "70.0"),
            ("Energy Price ($)", "energy_price", "115.0")
        ]
                     
        fields = [
            ("Country", "country", countries, None),
            ("Month (1-12)", "month", months, None),
            ("Day of Week", "day_of_week", days, None)
        ] + [(lbl, col, "entry", dflt) for lbl, col, dflt in self.input_features]
        
        # Form layout: Label on left, Input on right (Vertical list)
        for i, (label_text, field_name, field_type, default_val) in enumerate(fields):
            lbl = ttk.Label(input_frame, text=label_text, background="#f8f9fa")
            lbl.grid(row=i, column=0, sticky="w", pady=8)
            
            if isinstance(field_type, list):
                entry = ttk.Combobox(input_frame, values=field_type, state="readonly", width=18)
                entry.set(field_type[0])
            else:
                entry = ttk.Entry(input_frame, width=21)
                if default_val:
                    entry.insert(0, default_val)
            
            entry.grid(row=i, column=1, sticky="e", pady=8, padx=(10,0))
            self.entries[field_name] = entry
                
        btn_frame = ttk.Frame(self.left_panel, style="TFrame")
        btn_frame.pack(pady=25)
        
        self.predict_btn = ttk.Button(btn_frame, text="Predict Demand", command=self.predict, width=15)
        self.predict_btn.pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_fields, width=8).pack(side="left", padx=5)

        # --- RIGHT PANEL: ANALYTICS DASHBOARD ---
        header_label = ttk.Label(self.right_panel, text="LSTM Inference Engine", font=("Helvetica", 20, "bold"), background="white")
        header_label.pack(pady=(15, 5))

        self.result_label = ttk.Label(self.right_panel, text="Ready for input...", font=("Helvetica", 22, "bold"), foreground="#7f8c8d", background="white")
        self.result_label.pack(pady=(10, 20))

        self.graph_frame = tk.Frame(self.right_panel, bg="white")
        self.graph_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas_widget = None 

        # --- LOAD DATA ---
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

    def perform_drift_analysis(self, country, user_inputs):
        if self.hist_df is None: return {}, []
        country_data = self.hist_df[self.hist_df['country'] == country]
        if country_data.empty: return {}, []

        z_scores = {}
        warnings = []
        for lbl, col, _ in self.input_features:
            val = user_inputs[col]
            mean_val = country_data[col].mean()
            std_val = country_data[col].std()
            if pd.isna(std_val) or std_val == 0: std_val = 1e-6 
                
            z_score = (val - mean_val) / std_val
            z_scores[lbl.split(" (")[0]] = z_score 
            
            if abs(z_score) > 2.0:
                direction = "high" if z_score > 0 else "low"
                warnings.append(f"• {lbl.split(' (')[0]} is unusually {direction} (Z: {z_score:+.1f})")
        return z_scores, warnings

    def predict(self):
        try:
            self.result_label.config(text="Running Inference...", foreground="#e74c3c")
            self.predict_btn.config(state="disabled")
            self.root.update_idletasks()
            
            inputs = {
                'country': self.entries['country'].get(),
                'month': int(self.entries['month'].get()),
                'day_of_week': int(self.entries['day_of_week'].get()),
                'avg_temperature': float(self.entries['avg_temperature'].get()),
                'humidity': float(self.entries['humidity'].get()),
                'co2_emission': float(self.entries['co2_emission'].get()),
                'renewable_share': float(self.entries['renewable_share'].get()),
                'urban_population': float(self.entries['urban_population'].get()),
                'industrial_activity_index': float(self.entries['industrial_activity_index'].get()),
                'energy_price': float(self.entries['energy_price'].get())
            }
            
            z_scores, drift_warnings = self.perform_drift_analysis(inputs['country'], inputs)
            if drift_warnings:
                warning_text = "CONCEPT DRIFT WARNING:\nExtreme outliers detected. Prediction may be volatile:\n\n" + "\n".join(drift_warnings)
                messagebox.showwarning("Data Drift Detected", warning_text)
            
            month_sin = math.sin(2 * math.pi * inputs['month'] / 12)
            month_cos = math.cos(2 * math.pi * inputs['month'] / 12)
            country_encoded = self.label_encoder.transform([inputs['country']])[0]
            
            feature_names = [
                'avg_temperature', 'humidity', 'co2_emission', 'renewable_share', 
                'urban_population', 'industrial_activity_index', 'energy_price', 
                'day_of_week', 'month_sin', 'month_cos', 'country_encoded'
            ]
            features_df = pd.DataFrame([[
                inputs['avg_temperature'], inputs['humidity'], inputs['co2_emission'], 
                inputs['renewable_share'], inputs['urban_population'], 
                inputs['industrial_activity_index'], inputs['energy_price'], 
                inputs['day_of_week'], month_sin, month_cos, country_encoded
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
            
            self.plot_prediction(actual_prediction, inputs['country'], inputs['month'], z_scores)
            
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all fields are valid numbers.")
            self.result_label.config(text="Input Error", foreground="#e74c3c")
            self.predict_btn.config(state="normal")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred:\n{e}")
            self.result_label.config(text="Prediction Failed", foreground="#e74c3c")
            self.predict_btn.config(state="normal")

    def plot_prediction(self, predicted_val, country, month, z_scores):
        if self.canvas_widget:
            self.canvas_widget.destroy()

        baseline_val = 0
        if self.hist_df is not None:
            mask = (self.hist_df['country'] == country) & (self.hist_df['month'] == month)
            filtered_data = self.hist_df[mask]
            if not filtered_data.empty:
                baseline_val = filtered_data['energy_consumption'].mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [1, 1.5]})
        
        categories = ['Historical Avg', 'AI Prediction']
        values = [baseline_val, predicted_val]
        colors = ['#95a5a6', '#27ae60'] 
        
        bars = ax1.bar(categories, values, color=colors, width=0.5)
        ax1.set_title(f"Demand vs Norm", fontsize=13, pad=10)
        ax1.set_ylabel("Energy (kWh)", fontsize=11)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), 
                    f"{yval:,.0f}", ha='center', va='bottom', fontweight='bold', fontsize=11)

        if z_scores:
            features = list(z_scores.keys())
            scores = list(z_scores.values())
            
            bar_colors = ['#e74c3c' if s > 0 else '#3498db' for s in scores]
            
            y_pos = np.arange(len(features))
            ax2.barh(y_pos, scores, color=bar_colors, alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(features, fontsize=10)
            ax2.axvline(x=0, color='black', linewidth=1)
            
            ax2.axvspan(2, max(3, max(scores)+0.5), color='#ff9999', alpha=0.2)
            ax2.axvspan(min(-3, min(scores)-0.5), -2, color='#99ccff', alpha=0.2)
            
            ax2.set_title("Input Deviation (Why did the AI predict this?)", fontsize=13, pad=10)
            ax2.set_xlabel("Deviation from Historical Mean (Z-Score)", fontsize=11)
            ax2.invert_yaxis() 
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

        fig.tight_layout(pad=3.0)
        
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
        self.result_label.config(text="Ready for input...", foreground="#7f8c8d")
        if self.canvas_widget:
            self.canvas_widget.destroy()

    def on_closing(self):
        plt.close('all') 
        self.root.quit()
        self.root.destroy()
        sys.exit(0) 

if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyPredictionApp(root)
    root.mainloop()