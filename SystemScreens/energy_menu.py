import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os

class MainMenuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Energy Demand Prediction System - Main Menu")
        self.root.geometry("1280x720")
        
        # Set the main window background to white
        self.root.configure(bg="white")
        
        # Apply theme and styling
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        # Configure custom styles: White backgrounds, Black text
        style.configure("TFrame", background="white")
        style.configure("Title.TLabel", font=("Helvetica", 36, "bold"), foreground="black", background="white")
        style.configure("Subtitle.TLabel", font=("Helvetica", 16), foreground="black", background="white")
        style.configure("Menu.TButton", font=("Helvetica", 14, "bold"), foreground="black", padding=15)
        
        self.create_widgets()

    def create_widgets(self):
        # Main container to center everything
        main_container = ttk.Frame(self.root, style="TFrame")
        main_container.place(relx=0.5, rely=0.5, anchor="center")
        
        # Header Section
        title = ttk.Label(main_container, text="Facility Energy Intelligence", style="Title.TLabel")
        title.pack(pady=(0, 10))
        
        subtitle = ttk.Label(main_container, 
                             text="LSTM-Based Energy Demand Forecasting & Analytics", 
                             style="Subtitle.TLabel")
        subtitle.pack(pady=(0, 50))
        
        # Buttons Section
        btn_width = 30
        
        # 1. Prediction System Button
        btn_predict = ttk.Button(main_container, 
                                 text="Energy Prediction System", 
                                 style="Menu.TButton",
                                 width=btn_width,
                                 command=self.open_prediction_system)
        btn_predict.pack(pady=10)
        
        # 2. EDA & Statistics Button
        btn_stats = ttk.Button(main_container, 
                               text="Statistics & EDA", 
                               style="Menu.TButton",
                               width=btn_width,
                               command=self.open_stats)
        btn_stats.pack(pady=10)
        
        # 3. History Button
        btn_history = ttk.Button(main_container, 
                                 text="Data History", 
                                 style="Menu.TButton",
                                 width=btn_width,
                                 command=self.open_history)
        btn_history.pack(pady=10)
        
        # Footer / Quit Button
        btn_quit = ttk.Button(main_container, 
                              text="Exit Application",
                              style="Menu.TButton",
                              width=15,
                              command=self.root.quit)
        btn_quit.pack(pady=(50, 0))

    def run_script(self, script_name):
        """Helper method to run external python scripts safely"""
        
        # FORCE UI REFRESH: This ensures the button visually clicks 
        # and doesn't freeze the main menu before opening the heavy subprocess
        self.root.update_idletasks() 
        
        if os.path.exists(script_name):
            try:
                # sys.executable ensures it uses the same Python interpreter
                # Popen runs the script asynchronously (won't freeze the menu)
                subprocess.Popen([sys.executable, script_name])
            except Exception as e:
                messagebox.showerror("Execution Error", f"Failed to open {script_name}.\nError: {e}")
        else:
            messagebox.showwarning("File Not Found", 
                                   f"The file '{script_name}' does not exist in the current directory yet.\n\n"
                                   "Please create it to enable this feature.")

    def open_prediction_system(self):
        self.run_script("energy_system.py")

    def open_stats(self):
        self.run_script("energy_stats.py")

    def open_history(self):
        self.run_script("energy_history.py")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Optional: Set minimum window size
    root.minsize(800, 600)
    
    app = MainMenuApp(root)
    root.mainloop()