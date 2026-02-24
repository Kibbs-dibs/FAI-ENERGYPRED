import tkinter as tk
from tkinter import ttk, messagebox

class EnergyPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Energy Demand Prediction System")
        self.root.geometry("1280x720")
        self.root.configure(bg="white")
        
        # Apply strict Black and White theme
        self.style = ttk.Style()
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')
            
        self.style.configure("TFrame", background="white")
        self.style.configure("TLabel", background="white", foreground="black", font=("Helvetica", 11))
        self.style.configure("Title.TLabel", font=("Helvetica", 28, "bold"), background="white", foreground="black")
        self.style.configure("TButton", font=("Helvetica", 12, "bold"), foreground="black", padding=10)
        self.style.configure("Result.TLabel", font=("Helvetica", 16, "bold"), background="white", foreground="black")

        self.entries = {}
        self.create_widgets()

    def create_widgets(self):
        # Main container to center everything
        main_container = ttk.Frame(self.root, style="TFrame")
        main_container.place(relx=0.5, rely=0.5, anchor="center")
        
        # Title
        title_label = ttk.Label(main_container, text="LSTM Energy Consumption Predictor", style="Title.TLabel")
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 40))
        
        # Input Frame (Holds the 2 columns)
        input_frame = ttk.Frame(main_container, style="TFrame")
        input_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Defined fields matching the dataset - Split into two columns for 1280x720 layout
        fields_col1 = [
            ("Month (1-12)", "Month", "entry"),
            ("Hour (0-23)", "Hour", "entry"),
            ("Day of Week", "DayOfWeek", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
            ("Holiday", "Holiday", ["Yes", "No"]),
            ("Temperature (°C)", "Temperature", "entry"),
            ("Humidity (%)", "Humidity", "entry")
        ]
        
        fields_col2 = [
            ("Square Footage", "SquareFootage", "entry"),
            ("Occupancy", "Occupancy", "entry"),
            ("HVAC Usage", "HVACUsage", ["On", "Off"]),
            ("Lighting Usage", "LightingUsage", ["On", "Off"]),
            ("Renewable Energy", "RenewableEnergy", "entry")
        ]
        
        # Helper function to generate input columns dynamically
        def create_column(fields, parent, col_index):
            for row_idx, (label_text, field_name, field_type) in enumerate(fields):
                # Container for each input pair to control vertical spacing
                field_container = ttk.Frame(parent, style="TFrame")
                field_container.grid(row=row_idx, column=col_index, padx=30, pady=10, sticky="w")
                
                label = ttk.Label(field_container, text=label_text)
                label.pack(anchor="w", pady=(0, 5))
                
                # Create dropdowns for categorical data, standard entries for numerical
                if isinstance(field_type, list):
                    entry = ttk.Combobox(field_container, values=field_type, state="readonly", width=37, font=("Helvetica", 11))
                    entry.set(field_type[0])
                else:
                    entry = ttk.Entry(field_container, width=39, font=("Helvetica", 11))
                
                entry.pack(anchor="w")
                self.entries[field_name] = entry

        # Build Column 1 and Column 2
        create_column(fields_col1, input_frame, 0)
        create_column(fields_col2, input_frame, 1)
        
        # Buttons Frame
        button_frame = ttk.Frame(main_container, style="TFrame")
        button_frame.grid(row=2, column=0, columnspan=2, pady=40)
        
        predict_btn = ttk.Button(button_frame, text="Predict Energy", command=self.predict, width=20)
        predict_btn.grid(row=0, column=0, padx=10)
        
        clear_btn = ttk.Button(button_frame, text="Clear Fields", command=self.clear_fields, width=15)
        clear_btn.grid(row=0, column=1, padx=10)
        
        close_btn = ttk.Button(button_frame, text="Close Window", command=self.root.quit, width=15)
        close_btn.grid(row=0, column=2, padx=10)
        
        # Result Label
        self.result_label = ttk.Label(main_container, text="", style="Result.TLabel")
        self.result_label.grid(row=3, column=0, columnspan=2, pady=10)

    def get_inputs(self):
        """Collect and validate input data from UI"""
        data = {}
        for field, entry in self.entries.items():
            value = entry.get().strip()
            if not value:
                messagebox.showerror("Validation Error", f"Please fill in the '{field}' field.")
                return None
            data[field] = value
        return data

    def predict(self):
        """Handle prediction execution"""
        data = self.get_inputs()
        if data is None:
            return
        
        # Display processing status to prevent user from spam-clicking
        self.result_label.config(text="Processing through LSTM Model...")
        self.root.update_idletasks()
        
        # In a real integration, the 'data' dictionary would be cast to a DataFrame, 
        # scaled, reshaped into a 3D Tensor, and passed to model.predict() here.
        
        # Simulated Output Sequence
        messagebox.showinfo("Data Collected", "Input data validated successfully.\nPassing tensors to LSTM...")
        self.result_label.config(text="Predicted Energy Consumption: [ Output Pending ] kWh")

    def clear_fields(self):
        """Reset all input fields to their default state"""
        for field_name, entry in self.entries.items():
            if isinstance(entry, ttk.Combobox):
                entry.current(0)
            else:
                entry.delete(0, tk.END)
        self.result_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    # Enforce minimum size so the 2-column grid doesn't collapse
    root.minsize(1000, 600)
    app = EnergyPredictionApp(root)
    root.mainloop()