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
                               text="Exploratory Data Analysis (Stats)", 
                               style="Menu.TButton",
                               width=btn_width,
                               command=self.open_stats)
        btn_stats.pack(pady=10)
        
        # 3. History Button
        btn_history = ttk.Button(main_container, 
                                 text="Submission History", 
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

    def run_script(self, script_name, close_menu=False):
        """Helper method to run external python scripts safely in the same folder"""
        
        # Force UI refresh to prevent the button from freezing while launching subprocess
        self.root.update_idletasks() 
        
        # Dynamically get the exact directory where this energy_menu.py file is located
        current_folder = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_folder, script_name)
        
        if os.path.exists(script_path):
            try:
                # sys.executable ensures the new script uses the exact same Python interpreter
                # Popen runs the script asynchronously
                subprocess.Popen([sys.executable, script_path])
                
                # Close the main menu window if requested
                if close_menu:
                    self.root.destroy()
                    
            except Exception as e:
                messagebox.showerror("Execution Error", f"Failed to open {script_name}.\nError: {e}")
        else:
            messagebox.showwarning("File Not Found", 
                                   f"The file '{script_name}' does not exist in the folder:\n{current_folder}\n\n"
                                   "Please ensure all files are in the same directory.")

    def open_prediction_system(self):
        self.run_script("energy_system.py", close_menu=True)

    def open_stats(self):
        self.run_script("energy_stats.py", close_menu=True)

    def open_history(self):
        self.run_script("energy_history.py", close_menu=True)

if __name__ == "__main__":
    root = tk.Tk()
    
    # Set minimum window size to prevent UI squeezing
    root.minsize(800, 600)
    
    app = MainMenuApp(root)
    root.mainloop()