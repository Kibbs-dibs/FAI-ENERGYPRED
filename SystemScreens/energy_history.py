import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os

class EnergyHistoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Historical Data Viewer")
        self.root.geometry("1100x650")
        self.root.configure(bg="white")
        
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        # UI Styling
        style.configure("Treeview", rowheight=25, font=("Helvetica", 10))
        style.configure("Treeview.Heading", font=("Helvetica", 11, "bold"))
        
        # Header
        title = ttk.Label(root, text="Historical Energy Database", font=("Helvetica", 20, "bold"), background="white")
        title.pack(pady=(15, 5))
        
        self.df = None
        self.display_columns = []
        
        # Top Controls Frame
        controls_frame = ttk.Frame(root, style="TFrame")
        controls_frame.pack(fill="x", padx=20, pady=10)
        
        # --- NEW: Multiple Filter Dropdowns ---
        # 1. Country Filter
        ttk.Label(controls_frame, text="Country:", background="white", font=("Helvetica", 11)).pack(side="left")
        self.country_filter = ttk.Combobox(controls_frame, state="readonly", width=18)
        self.country_filter.pack(side="left", padx=(5, 15))
        self.country_filter.bind("<<ComboboxSelected>>", self.apply_filters)
        
        # 2. Year Filter
        ttk.Label(controls_frame, text="Year:", background="white", font=("Helvetica", 11)).pack(side="left")
        self.year_filter = ttk.Combobox(controls_frame, state="readonly", width=10)
        self.year_filter.pack(side="left", padx=(5, 15))
        self.year_filter.bind("<<ComboboxSelected>>", self.apply_filters)
        
        # 3. Month Filter
        ttk.Label(controls_frame, text="Month:", background="white", font=("Helvetica", 11)).pack(side="left")
        self.month_filter = ttk.Combobox(controls_frame, state="readonly", width=10)
        self.month_filter.pack(side="left", padx=(5, 15))
        self.month_filter.bind("<<ComboboxSelected>>", self.apply_filters)
        
        # Reset Button
        ttk.Button(controls_frame, text="Reset Filters", command=self.reset_filters).pack(side="left", padx=10)
        
        # Data Count Label
        self.count_label = ttk.Label(controls_frame, text="", background="white", font=("Helvetica", 10, "italic"), foreground="#7f8c8d")
        self.count_label.pack(side="right", padx=10)

        # Table Frame
        table_frame = ttk.Frame(root)
        table_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame)
        y_scroll.pack(side="right", fill="y")
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal")
        x_scroll.pack(side="bottom", fill="x")
        
        # Treeview (Data Table)
        self.tree = ttk.Treeview(table_frame, yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.tree.pack(fill="both", expand=True)
        
        y_scroll.config(command=self.tree.yview)
        x_scroll.config(command=self.tree.xview)
        
        self.load_data()

    def load_data(self):
        # Handle potential variations in the file name based on your directory
        file_path = "Climate_Energy_Consumption_Dataset_2020_2024.csv"
        if not os.path.exists(file_path):
            file_path = "Climate & Energy Consumption 2020 - 2024.csv"
            if not os.path.exists(file_path):
                file_path = "../Climate_Energy_Consumption_Dataset_2020_2024.csv"
            
        try:
            self.df = pd.read_csv(file_path)
            
            # --- FEATURE ENGINEERING: Extract Year and Month for filtering ---
            temp_dates = pd.to_datetime(self.df['date'])
            self.df['Year'] = temp_dates.dt.year.astype(str)
            self.df['Month'] = temp_dates.dt.month.astype(str)
            
            # Setup Columns dynamically (excluding the hidden Year/Month helper columns)
            self.display_columns = [col for col in self.df.columns if col not in ['Year', 'Month']]
            self.tree["columns"] = self.display_columns
            self.tree["show"] = "headings"
            
            for col in self.display_columns:
                self.tree.heading(col, text=col.replace("_", " ").title())
                self.tree.column(col, width=120, anchor="center")
            
            # Populate filter dropdowns dynamically from the data
            self.country_filter['values'] = ["All"] + sorted(self.df['country'].unique().tolist())
            self.country_filter.current(0)
            
            self.year_filter['values'] = ["All"] + sorted(self.df['Year'].unique().tolist())
            self.year_filter.current(0)
            
            # Sort months numerically rather than alphabetically
            self.month_filter['values'] = ["All"] + sorted(self.df['Month'].unique().tolist(), key=int)
            self.month_filter.current(0)
            
            self.populate_table(self.df)
            
        except Exception as e:
            messagebox.showerror("Data Error", f"Failed to load dataset: {e}")

    def populate_table(self, dataframe):
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Cap display at 1000 rows to prevent Tkinter from freezing
        display_df = dataframe.head(1000)
        
        for index, row in display_df.iterrows():
            formatted_row = []
            for col in self.display_columns:
                val = row[col]
                # Format numbers to 2 decimal places for cleaner viewing
                if isinstance(val, float):
                    formatted_row.append(f"{val:.2f}")
                else:
                    formatted_row.append(val)
                    
            self.tree.insert("", "end", values=formatted_row)
            
        total = len(dataframe)
        showing = len(display_df)
        self.count_label.config(text=f"Showing {showing} of {total} records")

    def apply_filters(self, event=None):
        """Applies all active filters cumulatively."""
        filtered_df = self.df.copy()
        
        country = self.country_filter.get()
        year = self.year_filter.get()
        month = self.month_filter.get()
        
        if country != "All":
            filtered_df = filtered_df[filtered_df['country'] == country]
            
        if year != "All":
            filtered_df = filtered_df[filtered_df['Year'] == year]
            
        if month != "All":
            filtered_df = filtered_df[filtered_df['Month'] == month]
            
        self.populate_table(filtered_df)

    def reset_filters(self):
        """Resets all dropdowns to 'All' and reloads the full dataset."""
        self.country_filter.current(0)
        self.year_filter.current(0)
        self.month_filter.current(0)
        self.populate_table(self.df)

if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyHistoryApp(root)
    root.mainloop()