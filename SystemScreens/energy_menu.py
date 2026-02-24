import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os

class MainMenuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Energy Demand Prediction System - Main Menu")
        self.root.geometry("1000x600")
        self.root.configure(bg="white")
        
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        style.configure("TFrame", background="white")
        style.configure("Title.TLabel", font=("Helvetica", 32, "bold"), foreground="#2c3e50", background="white")
        style.configure("Subtitle.TLabel", font=("Helvetica", 14), foreground="#7f8c8d", background="white")
        style.configure("Menu.TButton", font=("Helvetica", 12, "bold"), padding=10)
        
        self.create_widgets()

    def create_widgets(self):
        main_container = ttk.Frame(self.root, style="TFrame")
        main_container.place(relx=0.5, rely=0.5, anchor="center")
        
        title = ttk.Label(main_container, text="Energy Consumption Predicktion System", style="Title.TLabel")
        title.pack(pady=(0, 5))
        
        self.subtitle = ttk.Label(main_container, text="LSTM-Based Energy Demand Forecasting", style="Subtitle.TLabel")
        self.subtitle.pack(pady=(0, 40))
        
        btn_width = 25
        
        # Store buttons in a list so we can disable them during loading
        # Store buttons in a list so we can disable them during loading
        self.buttons = []
        
        btn_predict = ttk.Button(main_container, text="Start Predickting", style="Menu.TButton", width=btn_width, command=self.open_prediction_system)
        btn_predict.pack(pady=10)
        self.buttons.append(btn_predict)
        
        btn_stats = ttk.Button(main_container, text="Statistics & EDA", style="Menu.TButton", width=btn_width, command=self.open_stats)
        btn_stats.pack(pady=10)
        self.buttons.append(btn_stats)

        # ADDED: Data History Button
        btn_history = ttk.Button(main_container, text="Historical Data Viewer", style="Menu.TButton", width=btn_width, command=self.open_history)
        btn_history.pack(pady=10)
        self.buttons.append(btn_history)
        
        btn_quit = ttk.Button(main_container, text="Exit", style="Menu.TButton", width=15, command=self.root.quit)
        btn_quit.pack(pady=(40, 0))
        self.buttons.append(btn_quit)

    def run_script(self, script_name):
        if os.path.exists(script_name):
            try:
                # 1. Show Loading State (Text + Hourglass cursor + Disable buttons)
                self.original_text = self.subtitle.cget("text")
                self.subtitle.config(text=f"Loading module... This may take a few seconds.", foreground="#e74c3c")
                self.root.config(cursor="watch") # Hourglass cursor
                for btn in self.buttons:
                    btn.config(state="disabled")
                
                self.root.update_idletasks() # Force UI to update immediately
                
                # 2. Launch process asynchronously so the UI doesn't freeze
                self.active_process = subprocess.Popen([sys.executable, script_name])
                
                # 3. Start a non-blocking loop to check when the process finishes
                self.check_process()
                
            except Exception as e:
                messagebox.showerror("Execution Error", f"Failed to open {script_name}.\nError: {e}")
                self.restore_menu()
        else:
            messagebox.showwarning("File Not Found", f"The file '{script_name}' does not exist.")

    def check_process(self):
        # .poll() returns None if the process is still running
        if self.active_process.poll() is None:
            # Check again in 200 milliseconds
            self.root.after(200, self.check_process)
        else:
            # Process finished, restore the menu
            self.restore_menu()

    def restore_menu(self):
        # Reset text, cursor, and re-enable buttons
        self.subtitle.config(text=self.original_text, foreground="#7f8c8d")
        self.root.config(cursor="") 
        for btn in self.buttons:
            btn.config(state="normal")

    def open_prediction_system(self):
        self.run_script("energy_system.py")

    def open_stats(self):
        self.run_script("energy_stats.py")
    
    def open_history(self):
        self.run_script("energy_history.py")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainMenuApp(root)
    root.mainloop()