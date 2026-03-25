import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys
import joblib
import datetime
import threading

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TIME_STEPS = 7


def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :-1])
        y.append(data[i + time_steps, -1])
    return np.array(X), np.array(y)


class ReportGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Performance Report Generator")
        self.root.geometry("820x640")
        self.root.configure(bg="white")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        ttk.Label(root, text="📊 Performance Report Generator",
                  font=("Helvetica", 22, "bold"), background="white").pack(pady=(25, 4))
        ttk.Label(root,
                  text="Generates a comprehensive multi-page PDF evaluation report of the LSTM system",
                  font=("Helvetica", 11), foreground="#7f8c8d", background="white").pack(pady=(0, 18))

        # Options
        opt_frame = ttk.LabelFrame(root, text="Report Sections", padding=15)
        opt_frame.pack(fill="x", padx=40, pady=8)

        self.inc_arch = tk.BooleanVar(value=True)
        self.inc_metrics = tk.BooleanVar(value=True)
        self.inc_charts = tk.BooleanVar(value=True)
        self.inc_feat = tk.BooleanVar(value=True)
        self.inc_data = tk.BooleanVar(value=True)

        checks = [
            (self.inc_arch, "Model Architecture Summary"),
            (self.inc_metrics, "Evaluation Metrics (MAE, RMSE, MAPE, R²)"),
            (self.inc_charts, "Actual vs Predicted Chart + Scatter"),
            (self.inc_feat, "Feature Importance Analysis"),
            (self.inc_data, "Dataset Overview & Statistical Analysis"),
        ]
        for i, (var, label) in enumerate(checks):
            ttk.Checkbutton(opt_frame, text=label, variable=var).grid(
                row=i, column=0, sticky="w", pady=4)

        # Save path
        path_frame = ttk.Frame(root)
        path_frame.pack(fill="x", padx=40, pady=10)
        ttk.Label(path_frame, text="Save to:", font=("Helvetica", 11),
                  background="white").pack(side="left")
        self.output_path = tk.StringVar(
            value=os.path.join(os.getcwd(), "LSTM_Energy_Report.pdf"))
        ttk.Entry(path_frame, textvariable=self.output_path, width=52).pack(
            side="left", padx=10)
        ttk.Button(path_frame, text="Browse", command=self.browse_path).pack(side="left")

        # Generate button
        self.gen_btn = ttk.Button(root, text="Generate PDF Report",
                                   width=28, command=self.generate_thread)
        self.gen_btn.pack(pady=14)

        self.progress = ttk.Progressbar(root, mode='indeterminate', length=450)
        self.progress.pack(pady=4)

        self.status_label = ttk.Label(root,
                                       text="Ready. Configure options and click Generate.",
                                       font=("Helvetica", 10, "italic"),
                                       foreground="#7f8c8d", background="white")
        self.status_label.pack(pady=4)

        # Log
        log_frame = ttk.LabelFrame(root, text="Generation Log", padding=10)
        log_frame.pack(fill="both", expand=True, padx=40, pady=10)
        self.log_text = tk.Text(log_frame, height=8, font=("Courier", 9),
                                 bg="#f8f9fa", state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def log(self, message):
        self.log_text.config(state="normal")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {message}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.update_idletasks()

    def browse_path(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
            initialfile="LSTM_Energy_Report.pdf")
        if path:
            self.output_path.set(path)

    def generate_thread(self):
        if not TF_AVAILABLE:
            messagebox.showerror("Dependency Error", "TensorFlow is not installed.")
            return
        self.gen_btn.config(state="disabled")
        self.progress.start(10)
        threading.Thread(target=self.generate_report, daemon=True).start()

    def generate_report(self):
        try:
            output_path = self.output_path.get()
            self.root.after(0, lambda: self.log("Initializing report generation..."))
            self.root.after(0, lambda: self.status_label.config(
                text="Processing... please wait.", foreground="#e74c3c"))

            self.root.after(0, lambda: self.log("Loading dataset and evaluating LSTM..."))
            df_raw, y_actual, y_pred, feat_imp, feat_names = self.load_and_evaluate()

            self.root.after(0, lambda: self.log(f"Writing PDF → {output_path}"))

            with PdfPages(output_path) as pdf:
                self.page_title(pdf)
                self.root.after(0, lambda: self.log("✅  Page 1: Title page"))

                if self.inc_arch.get():
                    self.page_architecture(pdf)
                    self.root.after(0, lambda: self.log("✅  Page 2: Architecture summary"))

                if self.inc_metrics.get():
                    self.page_metrics(pdf, y_actual, y_pred)
                    self.root.after(0, lambda: self.log("✅  Page 3: Evaluation metrics"))

                if self.inc_charts.get():
                    self.page_prediction_chart(pdf, y_actual, y_pred)
                    self.root.after(0, lambda: self.log("✅  Page 4: Prediction charts"))

                if self.inc_feat.get():
                    self.page_feature_importance(pdf, feat_imp, feat_names)
                    self.root.after(0, lambda: self.log("✅  Page 5: Feature importance"))

                if self.inc_data.get():
                    self.page_data_analysis(pdf, df_raw)
                    self.root.after(0, lambda: self.log("✅  Page 6: Dataset analysis"))

            self.root.after(0, lambda: self.log(f"\n🎉  Report saved to:\n    {output_path}"))
            self.root.after(0, lambda: self.status_label.config(
                text="✅  Report generated successfully!", foreground="#27ae60"))
            self.root.after(0, lambda: messagebox.showinfo(
                "Success", f"PDF report generated!\n\nSaved to:\n{output_path}"))

        except Exception as e:
            self.root.after(0, lambda: self.log(f"❌  Error: {e}"))
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Report generation failed:\n{e}"))

        self.root.after(0, self.reset_ui)

    # ── Data Loading ────────────────────────────────────────────────────────

    def load_and_evaluate(self):
        df_raw = None
        for path in ["../Climate_Energy_Consumption_Dataset_2020_2024.csv",
                     "Climate_Energy_Consumption_Dataset_2020_2024.csv"]:
            if os.path.exists(path):
                df_raw = pd.read_csv(path)
                break
        if df_raw is None:
            raise FileNotFoundError("Dataset CSV not found.")

        df = df_raw.copy()
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
        X_test_seq = X_seq[split_idx:]
        y_test = y_seq[split_idx:]

        model = tf.keras.models.load_model(
            os.path.join(model_dir, 'lstm_energy_model.h5'))
        y_pred_scaled = model.predict(X_test_seq, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        X_rf = df.drop('energy_consumption', axis=1)
        rf.fit(X_rf, df['energy_consumption'])
        feat_imp = rf.feature_importances_
        feat_names = X_rf.columns.tolist()

        return df_raw, y_actual, y_pred, feat_imp, feat_names

    # ── PDF Pages ───────────────────────────────────────────────────────────

    def page_title(self, pdf):
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('#2c3e50')
        props = dict(ha='center', transform=fig.transFigure)
        fig.text(0.5, 0.72, "LSTM Energy Demand Prediction System",
                 fontsize=26, fontweight='bold', color='white', **props)
        fig.text(0.5, 0.62, "Performance Evaluation Report",
                 fontsize=20, color='#27ae60', **props)
        fig.text(0.5, 0.50, "CT032-3-3 Further Artificial Intelligence",
                 fontsize=14, color='#bdc3c7', **props)
        fig.text(0.5, 0.44, "Asia Pacific University of Technology and Innovation (APU)",
                 fontsize=12, color='#bdc3c7', **props)
        fig.text(0.5, 0.30,
                 f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
                 fontsize=11, color='#95a5a6', **props)
        ax = fig.add_axes([0.2, 0.565, 0.6, 0.003])
        ax.set_facecolor('#27ae60')
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def page_architecture(self, pdf):
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("Model Architecture Summary", fontsize=18, fontweight='bold', y=0.97)
        ax = fig.add_axes([0, 0, 1, 0.92])
        ax.axis('off')
        rows = [
            ["Input Layer", "Shape: (7, 11) — 7 time steps × 11 features",
             "Receives 3D sequential tensor"],
            ["LSTM Layer 1", "64 units, return_sequences=True",
             "Captures long-range temporal dependencies"],
            ["Dropout 1", "Rate: 20%", "Regularization — prevents overfitting"],
            ["LSTM Layer 2", "32 units, return_sequences=False",
             "Refines sequential feature extraction"],
            ["Dropout 2", "Rate: 20%", "Second regularization layer"],
            ["Dense Layer", "16 units, ReLU activation",
             "Non-linear transformation of LSTM output"],
            ["Output Layer", "1 unit, linear", "Single continuous kWh prediction"],
            ["", "", ""],
            ["Optimizer", "Adam", "Adaptive learning rate"],
            ["Loss Function", "Mean Squared Error (MSE)", "Penalizes large errors"],
            ["Epochs", "20", "Training iterations"],
            ["Batch Size", "64", "Samples per gradient update"],
            ["Train/Test Split", "80% / 20% (Chronological)", "Prevents data leakage"],
            ["Time Steps", "7-day Sliding Window", "Sequential context window"],
        ]
        table = ax.table(cellText=rows,
                         colLabels=["Component", "Configuration", "Purpose"],
                         cellLoc='left', loc='center',
                         bbox=[0.02, 0.02, 0.96, 0.90])
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('#dee2e6')
            if r == 0:
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#f8f9fa')
            if c == 0:
                cell.set_text_props(fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def page_metrics(self, pdf, y_actual, y_pred):
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mape = np.mean(np.abs((y_actual - y_pred) / (np.abs(y_actual) + 1e-6))) * 100
        r2 = 1 - np.sum((y_actual - y_pred) ** 2) / np.sum(
            (y_actual - np.mean(y_actual)) ** 2)

        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("Model Evaluation Metrics — LSTM Test Set Performance",
                     fontsize=17, fontweight='bold', y=0.97)

        metrics = [
            ("MAE", f"{mae:,.2f} kWh", "Avg absolute daily error", '#27ae60'),
            ("RMSE", f"{rmse:,.2f} kWh", "Penalises large outlier errors", '#e74c3c'),
            ("MAPE", f"{mape:.2f}%", "Mean absolute % error", '#3498db'),
            ("R² Score", f"{r2:.4f}", "Variance explained by model", '#e67e22'),
        ]
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3,
                               top=0.88, bottom=0.08, left=0.08, right=0.95)
        for i, (name, value, desc, color) in enumerate(metrics):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            ax.set_facecolor(color)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.patch.set_visible(True)  
            ax.text(0.5, 0.65, name, ha='center', fontsize=22, fontweight='bold',
                    color='white', transform=ax.transAxes)
            ax.text(0.5, 0.38, value, ha='center', fontsize=18, color='white',
                    transform=ax.transAxes)
            ax.text(0.5, 0.12, desc, ha='center', fontsize=10, color='white',
                    alpha=0.85, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def page_prediction_chart(self, pdf, y_actual, y_pred):
        fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
        fig.suptitle("Actual vs Predicted Energy Demand",
                     fontsize=16, fontweight='bold')

        slice_len = 150
        axes[0].plot(y_actual[:slice_len], color='#95a5a6', linewidth=2,
                     label='Actual Demand', alpha=0.9)
        axes[0].plot(y_pred[:slice_len], color='#27ae60', linestyle='--',
                     linewidth=2, label='LSTM Prediction', alpha=0.9)
        axes[0].fill_between(range(slice_len), y_actual[:slice_len],
                             y_pred[:slice_len], alpha=0.12, color='#27ae60')
        axes[0].set_title(f"Time-Series Comparison — First {slice_len} Test Days",
                          fontsize=12)
        axes[0].set_ylabel("Energy Consumption (kWh)", fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        axes[1].scatter(y_actual, y_pred, alpha=0.25, s=8, color='#3498db')
        lims = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
        axes[1].plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_title("Actual vs Predicted Scatter — All Test Samples", fontsize=12)
        axes[1].set_xlabel("Actual Energy (kWh)", fontsize=11)
        axes[1].set_ylabel("Predicted Energy (kWh)", fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def page_feature_importance(self, pdf, importances, feat_names):
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        fig.suptitle("Feature Importance Analysis — Random Forest Regressor",
                     fontsize=16, fontweight='bold')

        sorted_idx = np.argsort(importances)
        colors = ['#27ae60' if importances[i] >= np.mean(importances) else '#95a5a6'
                  for i in sorted_idx]
        bars = ax.barh(np.array(feat_names)[sorted_idx], importances[sorted_idx],
                       color=colors, alpha=0.88)
        ax.axvline(np.mean(importances), color='#e74c3c', linestyle='--',
                   linewidth=1.5, label=f"Mean: {np.mean(importances):.3f}")
        for bar, val in zip(bars, importances[sorted_idx]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va='center', fontsize=9)
        ax.set_title("Features ranked by predictive contribution to energy_consumption",
                     fontsize=12)
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def page_data_analysis(self, pdf, df):
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        fig.suptitle("Dataset Overview & Statistical Analysis",
                     fontsize=16, fontweight='bold')

        df_t = df.copy()
        df_t['date'] = pd.to_datetime(df_t['date'])
        df_t['Month'] = df_t['date'].dt.month
        df_t['Year'] = df_t['date'].dt.year

        counts = df_t['country'].value_counts()
        axes[0, 0].barh(counts.index, counts.values, color='#3498db', alpha=0.8)
        axes[0, 0].set_title("Data Records per Country", fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel("Number of Records", fontsize=10)
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)

        monthly = df_t.groupby('Month')['energy_consumption'].mean()
        axes[0, 1].plot(range(1, 13), monthly.values, marker='o',
                        color='#27ae60', linewidth=2.5)
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                                   rotation=30, fontsize=8)
        axes[0, 1].set_title("Global Monthly Average Energy", fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel("Avg Energy (kWh)", fontsize=10)
        axes[0, 1].grid(True, linestyle=':', alpha=0.5)
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)

        hb = axes[1, 0].hexbin(df_t['avg_temperature'], df_t['energy_consumption'],
                               gridsize=30, cmap='YlOrRd', mincnt=1)
        plt.colorbar(hb, ax=axes[1, 0], label='Count')
        axes[1, 0].set_title("Temperature vs Energy (Density)", fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel("Avg Temperature (°C)", fontsize=10)
        axes[1, 0].set_ylabel("Energy Consumption (kWh)", fontsize=10)

        summary = [
            ["Total Records", f"{len(df_t):,}"],
            ["Countries", f"{df_t['country'].nunique()}"],
            ["Date Range", f"{df_t['date'].min().strftime('%Y-%m-%d')} → "
                           f"{df_t['date'].max().strftime('%Y-%m-%d')}"],
            ["Mean Energy", f"{df_t['energy_consumption'].mean():,.2f} kWh"],
            ["Max Energy", f"{df_t['energy_consumption'].max():,.2f} kWh"],
            ["Min Energy", f"{df_t['energy_consumption'].min():,.2f} kWh"],
            ["Features (engineered)", "11"],
            ["Missing Values", f"{df_t.isnull().sum().sum()}"],
        ]
        axes[1, 1].axis('off')
        table = axes[1, 1].table(
            cellText=summary, colLabels=['Attribute', 'Value'],
            cellLoc='center', loc='center', bbox=[0, 0.05, 1, 0.90])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#f8f9fa')
        axes[1, 1].set_title("Dataset Summary", fontsize=12, fontweight='bold', pad=15)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def reset_ui(self):
        self.progress.stop()
        self.gen_btn.config(state="normal")

    def on_closing(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = ReportGeneratorApp(root)
    root.mainloop()