import tkinter as tk
from tkinter import ttk, messagebox

class EnergyPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Energy Consumption Predictor")
        self.root.geometry("500x700")
        
        title_label = ttk.Label(root, text="Energy Consumption Prediction", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=15)
        
        input_frame = ttk.Frame(root, padding="20")
        input_frame.pack(fill="both", expand=True)
        
        self.entries = {}
        
        # Dropdown options extracted from the dataset
        countries = ['Germany', 'France', 'Netherlands', 'Italy', 'Spain', 'Sweden', 'Norway',
                     'Poland', 'Turkey', 'United Kingdom', 'United States', 'Canada', 'Brazil',
                     'India', 'China', 'Japan', 'Australia', 'South Africa', 'Mexico', 'Indonesia']
                     
        fields = [
            ("Country", "country", countries),
            ("Month (1-12)", "month", "entry"),
            ("Day of Week (0=Mon, 6=Sun)", "day_of_week", "entry"),
            ("Avg Temperature (°C)", "avg_temperature", "entry"),
            ("Humidity (%)", "humidity", "entry"),
            ("CO2 Emission", "co2_emission", "entry"),
            ("Renewable Share (%)", "renewable_share", "entry"),
            ("Urban Population (%)", "urban_population", "entry"),
            ("Industrial Activity Index", "industrial_activity_index", "entry"),
            ("Energy Price (USD)", "energy_price", "entry")
        ]
        
        # Generate UI
        for label_text, field_name, field_type in fields:
            lbl = ttk.Label(input_frame, text=label_text)
            lbl.pack(anchor="w", pady=(5, 2))
            
            if isinstance(field_type, list):
                entry = ttk.Combobox(input_frame, values=field_type, state="readonly", width=47)
                entry.set(field_type[0])
            else:
                entry = ttk.Entry(input_frame, width=50)
            
            entry.pack(anchor="w", pady=(0, 5))
            self.entries[field_name] = entry
            
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Predict Energy Demand", command=self.predict).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Clear Fields", command=self.clear_fields).pack(side="left", padx=10)
        
        self.result_label = ttk.Label(root, text="", font=("Helvetica", 14, "bold"), foreground="#27ae60")
        self.result_label.pack(pady=10)

    def predict(self):
        data = {}
        for field, entry in self.entries.items():
            val = entry.get().strip()
            if not val:
                messagebox.showerror("Error", f"Missing value for {field}")
                return
            data[field] = val
            
        # Placeholder for actual model inference
        messagebox.showinfo("Processing", "Data collected successfully!\n\n(In the next step, this will be passed to the LSTM Engine in Model/train_model.py)")
        self.result_label.config(text="Predicted Demand: ~7,450 kWh\nStatus: Normal")

    def clear_fields(self):
        for entry in self.entries.values():
            if isinstance(entry, ttk.Combobox):
                entry.current(0)
            else:
                entry.delete(0, tk.END)
        self.result_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    app = EnergyPredictionApp(root)
    root.mainloop()