import tkinter as tk
from tkinter import ttk, messagebox

class EnergyPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Energy Consumption Predictor")
        self.root.geometry("450x750") # Adjusted size for better fit
        
        # Title
        title_label = ttk.Label(root, text="Energy Consumption Prediction", 
                                font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create frame for inputs
        input_frame = ttk.Frame(root, padding="10")
        input_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.entries = {}
        
        # Defined fields matching the dataset: (Label, Dictionary Key, Type/Options)
        fields = [
            ("Month (1-12)", "Month", "entry"),
            ("Hour (0-23)", "Hour", "entry"),
            ("Day of Week", "DayOfWeek", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
            ("Holiday", "Holiday", ["Yes", "No"]),
            ("Temperature (°C)", "Temperature", "entry"),
            ("Humidity (%)", "Humidity", "entry"),
            ("Square Footage", "SquareFootage", "entry"),
            ("Occupancy", "Occupancy", "entry"),
            ("HVAC Usage", "HVACUsage", ["On", "Off"]),
            ("Lighting Usage", "LightingUsage", ["On", "Off"]),
            ("Renewable Energy", "RenewableEnergy", "entry")
        ]
        
        # Generate UI dynamically
        for label_text, field_name, field_type in fields:
            # Create Label
            label = ttk.Label(input_frame, text=label_text)
            label.pack(anchor="w", pady=(5, 0))
            
            # Create Input (Combobox for categorical, Entry for numerical)
            if isinstance(field_type, list):
                entry = ttk.Combobox(input_frame, values=field_type, state="readonly", width=37)
                entry.set(field_type[0]) # Set default value
            else:
                entry = ttk.Entry(input_frame, width=40)
            
            entry.pack(anchor="w", pady=(0, 5))
            self.entries[field_name] = entry
        
        # Buttons frame
        button_frame = ttk.Frame(root)
        button_frame.pack(pady=15)
        
        predict_btn = ttk.Button(button_frame, text="Predict Energy", 
                                command=self.predict)
        predict_btn.pack(side="left", padx=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear", 
                              command=self.clear_fields)
        clear_btn.pack(side="left", padx=5)
        
        # Result label
        self.result_label = ttk.Label(root, text="", font=("Arial", 12, "bold"), foreground="blue")
        self.result_label.pack(pady=10)
    
    def get_inputs(self):
        """Collect input data from UI"""
        data = {}
        for field, entry in self.entries.items():
            value = entry.get().strip()
            if not value:
                messagebox.showerror("Error", f"Please fill in the '{field}' field.")
                return None
            data[field] = value
        return data
    
    def predict(self):
        """Handle prediction"""
        data = self.get_inputs()
        if data is None:
            return
        
        # Placeholder for model prediction
        messagebox.showinfo("Success", f"Input data collected successfully!\n\n(Ready to pass to LSTM model)")
        
        # Simulated Output Update
        self.result_label.config(text="Prediction: [Model output pending...]")
    
    def clear_fields(self):
        """Clear all input fields"""
        for field_name, entry in self.entries.items():
            if isinstance(entry, ttk.Combobox):
                entry.current(0) # Reset to first option
            else:
                entry.delete(0, tk.END)
        self.result_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Optional: Add a simple theme if available
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
        
    app = EnergyPredictionApp(root)
    root.mainloop()