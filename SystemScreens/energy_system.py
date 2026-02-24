import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os
import joblib
import math

try:
    import tensorflow as tf
except ImportError:
    tf = None

class EnergyPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Energy Consumption Predictor")
        
        # Changed window to be wider and shorter (850x550)
        self.root.geometry("850x550")
        self.root.configure(bg="white")
        
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        title_label = ttk.Label(root, text="Energy Consumption Prediction", font=("Helvetica", 18, "bold"), background="white")
        title_label.pack(pady=15)
        
        # Grid Container
        input_frame = ttk.Frame(root, padding="10")
        input_frame.pack(fill="both", expand=True, padx=30)
        
        self.entries = {}
        
        # Pre-defined options for dropdowns
        countries = ['Germany', 'France', 'Netherlands', 'Italy', 'Spain', 'Sweden', 'Norway',
                     'Poland', 'Turkey', 'United Kingdom', 'United States', 'Canada', 'Brazil',
                     'India', 'China', 'Japan', 'Australia', 'South Africa', 'Mexico', 'Indonesia']
        months = [str(i) for i in range(1, 13)]
        days = ['0', '1', '2', '3', '4', '5', '6'] 
                     
        fields = [
            ("Country", "country", countries),
            ("Month (1-12)", "month", months),
            ("Day of Week (0=Mon, 6=Sun)", "day_of_week", days),
            ("Avg Temperature (°C)", "avg_temperature", "entry"),
            ("Humidity (%)", "humidity", "entry"),
            ("CO2 Emission", "co2_emission", "entry"),
            ("Renewable Share (%)", "renewable_share", "entry"),
            ("Urban Population (%)", "urban_population", "entry"),
            ("Industrial Activity Index", "industrial_activity_index", "entry"),
            ("Energy Price (USD)", "energy_price", "entry")
        ]
        
        # Generate UI using a 2-Column Grid
        row, col = 0, 0
        for label_text, field_name, field_type in fields:
            # Create a mini-container for each label+input pair
            field_container = ttk.Frame(input_frame)
            field_container.grid(row=row, column=col, padx=20, pady=8, sticky="ew")
            input_frame.columnconfigure(col, weight=1) # Distributes width evenly
            
            lbl = ttk.Label(field_container, text=label_text)
            lbl.pack(anchor="w", pady=(0, 2))
            
            if isinstance(field_type, list):
                entry = ttk.Combobox(field_container, values=field_type, state="readonly")
                entry.set(field_type[0])
            else:
                entry = ttk.Entry(field_container)
            
            entry.pack(fill="x")
            self.entries[field_name] = entry
            
            # Logic to wrap to the next row after 2 columns
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
        self.result_label.pack(pady=(5, 15))

        self.load_models()

    def load_models(self):
        """Loads the saved LSTM model and preprocessing scalers."""
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
            
            import pandas as pd
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
            
            final_text = f"Predicted Demand:\n{actual_prediction:,.2f} kWh"
            self.result_label.config(text=final_text, foreground="#27ae60")
            self.predict_btn.config(state="normal")
            
            messagebox.showinfo("Inference Complete", f"The AI has processed the data.\n\n{final_text}")
            
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all fields are filled with valid numbers.")
            self.result_label.config(text="")
            self.predict_btn.config(state="normal")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred:\n{e}")
            self.result_label.config(text="")
            self.predict_btn.config(state="normal")

    def clear_fields(self):
        for entry in self.entries.values():
            if isinstance(entry, ttk.Combobox):
                entry.current(0)
            else:
                entry.delete(0, tk.END)
        self.result_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyPredictionApp(root)
    root.mainloop()