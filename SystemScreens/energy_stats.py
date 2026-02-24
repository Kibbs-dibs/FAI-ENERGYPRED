import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

class EnergyStatsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exploratory Data Analysis")
        self.root.geometry("1280x800") # Increased base size
        
        # Force window to maximize depending on OS
        try:
            self.root.state('zoomed') # Works on Windows
        except tk.TclError:
            self.root.attributes('-zoomed', True) # Works on Linux/Mac
            
        self.root.configure(bg="white")
        
        # Bind the close button (X) to a strict kill protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        style = ttk.Style()
        if 'clam' in style.theme_names(): 
            style.theme_use('clam')
        
        self.df = None
        self.load_data()
        
        title = ttk.Label(root, text="Data Analytics Dashboard", font=("Helvetica", 24, "bold"), background="white")
        title.pack(pady=20)
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=30, pady=20)
        
        if self.df is not None:
            self.create_tabs()

    def load_data(self):
        file_path = "../Climate_Energy_Consumption_Dataset_2020_2024.csv"
        if not os.path.exists(file_path):
            file_path = "../Climate_Energy_Consumption_Dataset_2020_2024.csv"
            
        try:
            self.df = pd.read_csv(file_path)
        except Exception as e:
            messagebox.showerror("Data Error", f"Failed to load dataset: {e}")

    def create_tabs(self):
        # Tab 1: Correlation Matrix
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Correlation Heatmap")
        self.plot_correlation(tab1)
        
        # Tab 2: Country Comparison
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Demand by Country")
        self.plot_country_bar(tab2)

    def plot_correlation(self, parent):
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_country_bar(self, parent):
        fig, ax = plt.subplots(figsize=(10, 8))
        avg_demand = self.df.groupby('country')['energy_consumption'].mean().sort_values(ascending=False)
        
        # Fixed Seaborn Warning: Added hue and legend=False
        sns.barplot(x=avg_demand.values, y=avg_demand.index, hue=avg_demand.index, palette="viridis", legend=False, ax=ax)
        
        ax.set_title("Average Energy Consumption by Country", fontsize=16)
        ax.set_xlabel("Energy Consumption (kWh)", fontsize=12)
        ax.set_ylabel("") # Remove default 'country' label for cleaner look
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def on_closing(self):
        """Strictly terminate to prevent matplotlib from hanging the parent process"""
        plt.close('all') # Kill all matplotlib figures
        self.root.quit()
        self.root.destroy()
        sys.exit(0) # Force Python to close this script entirely

if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyStatsApp(root)
    root.mainloop()