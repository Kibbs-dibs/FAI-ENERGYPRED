import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class EnergyStatsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exploratory Data Analysis - Energy Statistics")
        self.root.geometry("1280x720")
        self.root.configure(bg="white")
        
        # Apply strict Black and White theme
        self.style = ttk.Style()
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')
            
        self.style.configure("TFrame", background="white")
        self.style.configure("TLabel", background="white", foreground="black")
        self.style.configure("Title.TLabel", font=("Helvetica", 24, "bold"), background="white", foreground="black")
        self.style.configure("TNotebook", background="white", borderwidth=0)
        self.style.configure("TNotebook.Tab", background="white", foreground="black", font=("Helvetica", 12, "bold"), padding=[10, 5])
        self.style.map("TNotebook.Tab", background=[("selected", "#e0e0e0")])
        self.style.configure("TButton", font=("Helvetica", 12), foreground="black")

        self.df = None
        self.load_data()
        self.create_widgets()

    def load_data(self):
        """Load and preprocess the dataset for visualization"""
        try:
            # Load the dataset
            self.df = pd.read_csv("Energy_consumption_dataset.csv")
            
            # Create a numerical copy for the correlation matrix
            self.df_numeric = self.df.copy()
            
            # Map categorical to numeric for correlation
            mapping_dict = {'Yes': 1, 'No': 0, 'On': 1, 'Off': 0}
            for col in ['Holiday', 'HVACUsage', 'LightingUsage']:
                if col in self.df_numeric.columns:
                    self.df_numeric[col] = self.df_numeric[col].map(mapping_dict)
            
            # Drop purely categorical columns that can't be easily linearly correlated
            if 'DayOfWeek' in self.df_numeric.columns:
                self.df_numeric.drop(columns=['DayOfWeek'], inplace=True)
                
        except FileNotFoundError:
            messagebox.showerror("File Not Found", "Could not find 'Energy_consumption_dataset.csv'.\nPlease ensure it is in the same directory.")
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Data Error", f"An error occurred while processing data:\n{e}")
            self.root.quit()

    def create_widgets(self):
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", pady=10, padx=20)
        
        title = ttk.Label(header_frame, text="Energy Consumption Statistics & Analysis", style="Title.TLabel")
        title.pack(side="left")
        
        close_btn = ttk.Button(header_frame, text="Close Window", command=self.root.quit)
        close_btn.pack(side="right")
        
        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create Tabs
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text="Distribution Analysis")
        self.notebook.add(self.tab2, text="Correlation Heatmap")
        self.notebook.add(self.tab3, text="Temperature vs Energy")
        
        # Populate Tabs
        if self.df is not None:
            self.plot_distribution(self.tab1)
            self.plot_correlation(self.tab2)
            self.plot_scatter(self.tab3)

    def draw_canvas(self, fig, parent_frame):
        """Helper to draw matplotlib figure on tkinter canvas"""
        fig.patch.set_facecolor('white') # Force white background
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_distribution(self, parent):
        """Plot a histogram showing the distribution of Energy Consumption"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(self.df['EnergyConsumption'], bins=50, kde=True, color="black", ax=ax)
        
        ax.set_title("Distribution of Energy Consumption", fontsize=16, color="black")
        ax.set_xlabel("Energy Consumption (kWh)", fontsize=12, color="black")
        ax.set_ylabel("Frequency", fontsize=12, color="black")
        
        # Style axes
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
            
        fig.tight_layout()
        self.draw_canvas(fig, parent)

    def plot_correlation(self, parent):
        """Plot a correlation heatmap to show feature relationships"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate correlation matrix
        corr = self.df_numeric.corr()
        
        # Draw heatmap (using grayscale/black/white colormap to adhere to theme)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greys", cbar=True, ax=ax, 
                    annot_kws={"color": "black", "size": 9})
        
        ax.set_title("Feature Correlation Heatmap", fontsize=16, color="black")
        ax.tick_params(colors="black", labelsize=10)
        
        fig.tight_layout()
        self.draw_canvas(fig, parent)

    def plot_scatter(self, parent):
        """Plot Scatter plot of Temperature vs Energy Consumption by HVAC usage"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # We will use markers instead of colors to keep the black and white theme
        hvac_on = self.df[self.df['HVACUsage'] == 'On']
        hvac_off = self.df[self.df['HVACUsage'] == 'Off']
        
        ax.scatter(hvac_on['Temperature'], hvac_on['EnergyConsumption'], 
                   color="black", marker="o", alpha=0.6, label="HVAC On")
        ax.scatter(hvac_off['Temperature'], hvac_off['EnergyConsumption'], 
                   color="gray", marker="x", alpha=0.6, label="HVAC Off")
        
        ax.set_title("Impact of Temperature & HVAC on Energy", fontsize=16, color="black")
        ax.set_xlabel("Temperature (Celsius)", fontsize=12, color="black")
        ax.set_ylabel("Energy Consumption (kWh)", fontsize=12, color="black")
        
        ax.legend(facecolor="white", edgecolor="black", labelcolor="black")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
            
        fig.tight_layout()
        self.draw_canvas(fig, parent)

if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyStatsApp(root)
    root.mainloop()