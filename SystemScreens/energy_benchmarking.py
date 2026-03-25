import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import joblib
import threading

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TIME_STEPS = 7


def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :-1])
        y.append(data[i + time_steps, -1])
    return np.array(X), np.array(y)


class BenchmarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Benchmarking & Comparative Evaluation")
        self.root.geometry("1280x800")
        try:
            self.root.state('zoomed')
        except tk.TclError:
            self.root.attributes('-zoomed', True)
        self.root.configure(bg="white")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        ttk.Label(root, text="Model Benchmarking & Comparative Evaluation",
                  font=("Helvetica", 22, "bold"), background="white").pack(pady=(15, 2))
        ttk.Label(root, text="LSTM vs Naive Persistence vs Linear Regression vs Random Forest",
                  font=("Helvetica", 12), foreground="#7f8c8d", background="white").pack()

        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(pady=12)

        self.run_btn = ttk.Button(ctrl_frame, text="▶  Run Benchmark", width=20,
                                   command=self.run_benchmark_thread)
        self.run_btn.pack(side="left", padx=10)

        self.status_label = ttk.Label(ctrl_frame,
                                       text="Ready. Click 'Run Benchmark' to start evaluation.",
                                       font=("Helvetica", 10, "italic"),
                                       foreground="#7f8c8d", background="white")
        self.status_label.pack(side="left", padx=10)

        self.progress = ttk.Progressbar(root, mode='indeterminate', length=500)
        self.progress.pack(pady=4)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)

        for tab_name in ["Performance Metrics", "Prediction Comparison",
                         "Residual Analysis", "Error Over Time"]:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=tab_name)
            setattr(self, f"tab_{tab_name.replace(' ', '_').lower()}", frame)

        self.y_actual = None
        self.results = {}

    # ── Threading ──────────────────────────────────────────────────────────

    def run_benchmark_thread(self):
        if not TF_AVAILABLE:
            messagebox.showerror("Dependency Error", "TensorFlow is not installed.")
            return
        self.run_btn.config(state="disabled")
        self.update_status("Loading data and models...", "#e74c3c")
        self.progress.start(10)
        threading.Thread(target=self.run_benchmark, daemon=True).start()

    def update_status(self, msg, color="#e74c3c"):
        self.root.after(0, lambda: self.status_label.config(text=msg, foreground=color))

    def run_benchmark(self):
        try:
            self.update_status("Step 1/5: Loading & preprocessing dataset...")
            (X_train_flat, X_test_flat, y_train,
             y_test, X_train_seq, X_test_seq, scaler_y) = self.load_and_preprocess()

            self.update_status("Step 2/5: Running LSTM inference...")
            lstm_preds, y_actual = self.evaluate_lstm(X_test_seq, y_test, scaler_y)

            self.update_status("Step 3/5: Evaluating Naive Persistence baseline...")
            naive_preds = self.naive_persistence(y_test, scaler_y)

            self.update_status("Step 4/5: Training Linear Regression...")
            lr_preds = self.train_lr(X_train_flat, y_train, X_test_flat, scaler_y)

            self.update_status("Step 5/5: Training Random Forest...")
            rf_preds = self.train_rf(X_train_flat, y_train, X_test_flat, scaler_y)

            self.y_actual = y_actual
            self.results = {
                'LSTM (Our Model)': lstm_preds,
                'Naive Persistence': naive_preds,
                'Linear Regression': lr_preds,
                'Random Forest': rf_preds,
            }
            self.root.after(0, self.display_results)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Benchmark Error", f"{e}"))
            self.root.after(0, self.reset_ui)

    # ── Data Loading & Preprocessing ───────────────────────────────────────

    def load_and_preprocess(self):
        df = None
        for path in ["../Climate_Energy_Consumption_Dataset_2020_2024.csv",
                     "Climate_Energy_Consumption_Dataset_2020_2024.csv"]:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        if df is None:
            raise FileNotFoundError("Dataset CSV not found.")

        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "..", "Model", "saved_models")

        label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        scaler_X = joblib.load(os.path.join(model_dir, 'scaler_X.pkl'))
        scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))

        df['country_encoded'] = label_encoder.transform(df['country'])
        df = df.drop(['date', 'country', 'month'], axis=1)
        cols = [c for c in df.columns if c != 'energy_consumption'] + ['energy_consumption']
        df = df[cols]

        X_scaled = scaler_X.transform(df.drop('energy_consumption', axis=1))
        y_scaled = scaler_y.transform(df[['energy_consumption']])
        scaled_data = np.hstack((X_scaled, y_scaled))

        X_seq, y_seq = create_sequences(scaled_data, TIME_STEPS)
        split_idx = int(len(X_seq) * 0.8)

        X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
        X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)

        return X_train_flat, X_test_flat, y_train, y_test, X_train_seq, X_test_seq, scaler_y

    def evaluate_lstm(self, X_test_seq, y_test, scaler_y):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "..", "Model", "saved_models")
        model = tf.keras.models.load_model(os.path.join(model_dir, 'lstm_energy_model.h5'))
        y_pred_scaled = model.predict(X_test_seq, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        return y_pred, y_actual

    def naive_persistence(self, y_test, scaler_y):
        y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        preds = np.roll(y_actual, 1)
        preds[0] = y_actual[0]
        return preds

    def train_lr(self, X_train, y_train, X_test, scaler_y):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()

    def train_rf(self, X_train, y_train, X_test, scaler_y):
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()

    # ── Metrics ────────────────────────────────────────────────────────────

    def compute_metrics(self, y_actual, y_pred):
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mape = np.mean(np.abs((y_actual - y_pred) / (np.abs(y_actual) + 1e-6))) * 100
        ss_res = np.sum((y_actual - y_pred) ** 2)
        r2 = 1 - ss_res / np.sum((y_actual - np.mean(y_actual)) ** 2)
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

    # ── Display ────────────────────────────────────────────────────────────

    def display_results(self):
        self.progress.stop()
        self.update_status("✅ Benchmark complete! LSTM results validated against 3 baselines.", "#27ae60")

        all_metrics = {name: self.compute_metrics(self.y_actual, preds)
                       for name, preds in self.results.items()}

        self.build_metrics_tab(all_metrics)
        self.build_prediction_tab()
        self.build_residual_tab()
        self.build_error_tab()
        self.run_btn.config(state="normal")

    def embed(self, fig, parent):
        outer = tk.Frame(parent, bg="white")
        outer.pack(fill="both", expand=True)

        v_scroll = ttk.Scrollbar(outer, orient="vertical")
        v_scroll.pack(side="right", fill="y")
        h_scroll = ttk.Scrollbar(outer, orient="horizontal")
        h_scroll.pack(side="bottom", fill="x")

        tk_canvas = tk.Canvas(outer, bg="white",
                            yscrollcommand=v_scroll.set,
                            xscrollcommand=h_scroll.set)
        tk_canvas.pack(side="left", fill="both", expand=True)

        v_scroll.config(command=tk_canvas.yview)
        h_scroll.config(command=tk_canvas.xview)

        fig_canvas = FigureCanvasTkAgg(fig, master=tk_canvas)
        fig_canvas.draw()
        fig_widget = fig_canvas.get_tk_widget()
        tk_canvas.create_window((0, 0), window=fig_widget, anchor="nw")

        # KEY FIX: Update scroll region AFTER widget fully renders
        def update_scrollregion(event=None):
            tk_canvas.update_idletasks()
            tk_canvas.config(scrollregion=tk_canvas.bbox("all"))

        fig_widget.bind("<Configure>", update_scrollregion)

        # Scroll UP/DOWN with mouse wheel
        def on_scroll_y(event):
            tk_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Scroll LEFT/RIGHT with Shift + mouse wheel
        def on_scroll_x(event):
            tk_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        tk_canvas.bind("<MouseWheel>", on_scroll_y)
        tk_canvas.bind("<Shift-MouseWheel>", on_scroll_x)
        fig_widget.bind("<MouseWheel>", on_scroll_y)
        fig_widget.bind("<Shift-MouseWheel>", on_scroll_x)

    def build_metrics_tab(self, all_metrics):
        tab = self.tab_performance_metrics
        for w in tab.winfo_children():
            w.destroy()

        model_names = list(all_metrics.keys())
        short_names = ['LSTM', 'Naive', 'Lin. Reg.', 'Rand. Forest']
        colors = ['#27ae60', '#95a5a6', '#3498db', '#e67e22']

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Model Comparison — Lower is Better (except R²)",
                     fontsize=14, fontweight='bold')

        for ax, (key, label, title) in zip(axes, [
            ('MAE', 'kWh', 'Mean Absolute Error (MAE)'),
            ('RMSE', 'kWh', 'Root Mean Squared Error (RMSE)'),
            ('MAPE', '%', 'Mean Abs. Percentage Error (MAPE)'),
        ]):
            vals = [all_metrics[m][key] for m in model_names]
            bars = ax.bar(short_names, vals, color=colors, alpha=0.85, width=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.015,
                        f"{val:,.1f}", ha='center', fontweight='bold', fontsize=10)
            bars[0].set_edgecolor('#2c3e50')
            bars[0].set_linewidth(2.5)
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_ylabel(label, fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        self.embed(fig, tab)

        tree_frame = ttk.Frame(tab)
        tree_frame.pack(fill="x", padx=20, pady=(0, 10))
        cols = ['Model', 'MAE (kWh)', 'RMSE (kWh)', 'MAPE (%)', 'R² Score', 'Rank']
        tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=4)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor='center', width=170)

        ranked = sorted(all_metrics.items(), key=lambda x: x[1]['MAE'])
        for rank, (name, m) in enumerate(ranked, 1):
            tag = 'best' if rank == 1 else ('second' if rank == 2 else 'normal')
            tree.insert("", "end", values=[
                name,
                f"{m['MAE']:,.2f}",
                f"{m['RMSE']:,.2f}",
                f"{m['MAPE']:.2f}%",
                f"{m['R2']:.4f}",
                f"#{rank}"
            ], tags=(tag,))
        tree.tag_configure('best', foreground='#27ae60',
                           font=('Helvetica', 10, 'bold'))
        tree.tag_configure('second', foreground='#e67e22')
        tree.pack(fill="x")

    def build_prediction_tab(self):
        tab = self.tab_prediction_comparison
        slice_len = 150
        colors = ['#27ae60', '#95a5a6', '#3498db', '#e67e22']

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Actual vs Predicted — First {slice_len} Test Samples",
                     fontsize=14, fontweight='bold')
        axes = axes.flatten()

        for i, (name, preds) in enumerate(self.results.items()):
            axes[i].plot(self.y_actual[:slice_len], color='#2c3e50',
                         linewidth=1.5, label='Actual', alpha=0.85)
            axes[i].plot(preds[:slice_len], color=colors[i],
                         linestyle='--', linewidth=1.5, label=name, alpha=0.9)
            axes[i].set_title(name, fontsize=12, fontweight='bold')
            axes[i].set_xlabel("Time (Days)", fontsize=10)
            axes[i].set_ylabel("Energy (kWh)", fontsize=10)
            axes[i].legend(fontsize=9)
            axes[i].grid(True, linestyle=':', alpha=0.5)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

        plt.tight_layout()
        self.embed(fig, tab)

    def build_residual_tab(self):
        tab = self.tab_residual_analysis
        colors = ['#27ae60', '#95a5a6', '#3498db', '#e67e22']

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Residual Distribution — Ideal: Symmetric Bell Curve Centered at 0",
                     fontsize=13, fontweight='bold')
        axes = axes.flatten()

        for i, (name, preds) in enumerate(self.results.items()):
            residuals = self.y_actual - preds
            axes[i].hist(residuals, bins=50, color=colors[i],
                         alpha=0.7, edgecolor='white', density=True)
            pd.Series(residuals).plot.kde(ax=axes[i], color='#2c3e50', linewidth=2)
            axes[i].axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')
            axes[i].set_title(f"{name} — Residual Distribution", fontsize=12, fontweight='bold')
            axes[i].set_xlabel("Residual (kWh)", fontsize=10)
            axes[i].set_ylabel("Density", fontsize=10)
            axes[i].legend(fontsize=9)
            axes[i].text(0.97, 0.95,
                         f"Mean: {np.mean(residuals):,.0f}\nStd: {np.std(residuals):,.0f}",
                         transform=axes[i].transAxes, ha='right', va='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

        plt.tight_layout()
        self.embed(fig, tab)

    def build_error_tab(self):
        tab = self.tab_error_over_time
        colors = ['#27ae60', '#95a5a6', '#3498db', '#e67e22']

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle("Error Analysis Over Test Sequence", fontsize=14, fontweight='bold')

        for i, (name, preds) in enumerate(self.results.items()):
            axes[0].plot(np.abs(self.y_actual - preds), color=colors[i],
                         label=name, alpha=0.65, linewidth=1)
        axes[0].set_title("Absolute Error Over Time — All Models", fontsize=13)
        axes[0].set_xlabel("Test Sample Index", fontsize=11)
        axes[0].set_ylabel("Absolute Error (kWh)", fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, linestyle=':', alpha=0.5)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        for i, (name, preds) in enumerate(self.results.items()):
            cumulative_mae = (np.cumsum(np.abs(self.y_actual - preds))
                              / (np.arange(len(preds)) + 1))
            axes[1].plot(cumulative_mae, color=colors[i], label=name, linewidth=2)
        axes[1].set_title("Cumulative MAE — Convergence Analysis", fontsize=13)
        axes[1].set_xlabel("Test Sample Index", fontsize=11)
        axes[1].set_ylabel("Cumulative MAE (kWh)", fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, linestyle=':', alpha=0.5)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        self.embed(fig, tab)

    def reset_ui(self):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.update_status("Error occurred. Please try again.", "#e74c3c")

    def on_closing(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = BenchmarkApp(root)
    root.mainloop()