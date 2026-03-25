import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys


class AdvancedEDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Analytics Dashboard")
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

        self.df = None
        self.load_data()

        ttk.Label(root, text="Advanced Analytics Dashboard",
                  font=("Helvetica", 24, "bold"), background="white").pack(pady=(15, 2))
        ttk.Label(root, text="Extended Exploratory Data Analysis — Energy Consumption Dataset",
                  font=("Helvetica", 12), foreground="#7f8c8d", background="white").pack(pady=(0, 10))

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=30, pady=15)

        if self.df is not None:
            self.prepare_data()
            self.create_tabs()

    def load_data(self):
        paths = [
            "../Climate_Energy_Consumption_Dataset_2020_2024.csv",
            "Climate_Energy_Consumption_Dataset_2020_2024.csv",
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    self.df = pd.read_csv(path)
                    return
                except Exception as e:
                    messagebox.showerror("Data Error", f"Failed to load dataset: {e}")
                    return
        messagebox.showerror("Data Error", "Dataset file not found. Ensure the CSV is in the project root.")

    def prepare_data(self):
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['Year'] = self.df['date'].dt.year
        self.df['Month'] = self.df['date'].dt.month

        def get_season(m):
            if m in [12, 1, 2]: return 'Winter'
            elif m in [3, 4, 5]: return 'Spring'
            elif m in [6, 7, 8]: return 'Summer'
            else: return 'Autumn'

        self.df['Season'] = self.df['Month'].apply(get_season)

    def embed_figure(self, fig, parent):
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

    def create_tabs(self):
        tabs = [
            ("Monthly Trend", self.plot_monthly_trend),
            ("Seasonal Distribution", self.plot_seasonal),
            ("Energy vs Climate", self.plot_energy_vs_climate),
            ("Year-over-Year", self.plot_yoy),
            ("Feature Correlations", self.plot_feature_correlations),
            ("Distribution Analysis", self.plot_distribution),
        ]
        for tab_name, plot_func in tabs:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=tab_name)
            plot_func(frame)

    # --- Tab 1: Monthly Trend ---
    def plot_monthly_trend(self, parent):
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly = self.df.groupby(['Year', 'Month'])['energy_consumption'].mean().reset_index()
        palette = sns.color_palette("tab10", len(monthly['Year'].unique()))

        for i, year in enumerate(sorted(monthly['Year'].unique())):
            d = monthly[monthly['Year'] == year]
            ax.plot(d['Month'], d['energy_consumption'], marker='o', label=str(year),
                    color=palette[i], linewidth=2.5, markersize=6)

        ax.set_title("Average Monthly Energy Consumption by Year", fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Avg Energy Consumption (kWh)", fontsize=12)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.legend(title="Year", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        self.embed_figure(fig, parent)

    # --- Tab 2: Seasonal Distribution ---
    def plot_seasonal(self, parent):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
        season_colors = {'Spring': '#2ecc71', 'Summer': '#e74c3c',
                         'Autumn': '#e67e22', 'Winter': '#3498db'}

        season_data = [self.df[self.df['Season'] == s]['energy_consumption'].values
                       for s in season_order]
        bp = axes[0].boxplot(season_data, labels=season_order, patch_artist=True, notch=True)
        for patch, season in zip(bp['boxes'], season_order):
            patch.set_facecolor(season_colors[season])
            patch.set_alpha(0.75)
        axes[0].set_title("Energy Consumption Distribution by Season", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Energy Consumption (kWh)", fontsize=11)
        axes[0].grid(True, linestyle=':', alpha=0.5)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        season_means = self.df.groupby('Season')['energy_consumption'].mean().reindex(season_order)
        bars = axes[1].bar(season_order, season_means.values,
                           color=[season_colors[s] for s in season_order], alpha=0.85, width=0.5)
        for bar, val in zip(bars, season_means.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(season_means) * 0.012,
                         f"{val:,.0f}", ha='center', fontweight='bold', fontsize=11)
        axes[1].set_title("Mean Energy Consumption by Season", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("Mean Energy Consumption (kWh)", fontsize=11)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        self.embed_figure(fig, parent)

    # --- Tab 3: Energy vs Climate Variables ---
    def plot_energy_vs_climate(self, parent):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].scatter(self.df['avg_temperature'], self.df['energy_consumption'],
                        alpha=0.25, s=12, color='#e74c3c')
        z = np.polyfit(self.df['avg_temperature'].dropna(),
                       self.df.loc[self.df['avg_temperature'].notna(), 'energy_consumption'], 1)
        x_line = np.linspace(self.df['avg_temperature'].min(), self.df['avg_temperature'].max(), 100)
        axes[0].plot(x_line, np.poly1d(z)(x_line), "k--", linewidth=2, label="Trend Line")
        axes[0].set_title("Energy Consumption vs Average Temperature", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Average Temperature (°C)", fontsize=11)
        axes[0].set_ylabel("Energy Consumption (kWh)", fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        axes[1].scatter(self.df['industrial_activity_index'], self.df['energy_consumption'],
                        alpha=0.25, s=12, color='#8e44ad')
        z2 = np.polyfit(self.df['industrial_activity_index'].dropna(),
                        self.df.loc[self.df['industrial_activity_index'].notna(), 'energy_consumption'], 1)
        x_line2 = np.linspace(self.df['industrial_activity_index'].min(),
                               self.df['industrial_activity_index'].max(), 100)
        axes[1].plot(x_line2, np.poly1d(z2)(x_line2), "k--", linewidth=2, label="Trend Line")
        axes[1].set_title("Energy Consumption vs Industrial Activity Index", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Industrial Activity Index", fontsize=11)
        axes[1].set_ylabel("Energy Consumption (kWh)", fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        self.embed_figure(fig, parent)

    # --- Tab 4: Year-over-Year ---
    def plot_yoy(self, parent):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        yearly_global = self.df.groupby('Year')['energy_consumption'].mean().reset_index()
        bars = axes[0].bar(yearly_global['Year'].astype(str), yearly_global['energy_consumption'],
                           color='#27ae60', alpha=0.85, width=0.5)
        for bar, (_, row) in zip(bars, yearly_global.iterrows()):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(yearly_global['energy_consumption']) * 0.012,
                         f"{row['energy_consumption']:,.0f}", ha='center', fontweight='bold', fontsize=11)
        axes[0].set_title("Global Average Annual Energy Consumption", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Mean Energy Consumption (kWh)", fontsize=11)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        top5 = self.df.groupby('country')['energy_consumption'].mean().nlargest(5).index
        top5_data = (self.df[self.df['country'].isin(top5)]
                     .groupby(['country', 'Year'])['energy_consumption'].mean().reset_index())
        palette = sns.color_palette("Set2", 5)
        for i, country in enumerate(top5):
            cdata = top5_data[top5_data['country'] == country]
            axes[1].plot(cdata['Year'].astype(str), cdata['energy_consumption'],
                         marker='o', label=country, color=palette[i], linewidth=2.5)
        axes[1].set_title("Top 5 Countries — Year-over-Year Energy Trend", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("Mean Energy Consumption (kWh)", fontsize=11)
        axes[1].legend(title="Country", fontsize=10)
        axes[1].grid(True, linestyle=':', alpha=0.5)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        self.embed_figure(fig, parent)

    # --- Tab 5: Feature Correlations ---
    def plot_feature_correlations(self, parent):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        for col in ['Year', 'Month']:
            if col in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[col])

        corr = numeric_df.corr()['energy_consumption'].drop('energy_consumption').sort_values()
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in corr.values]

        bars = axes[0].barh(corr.index, corr.values, color=colors, alpha=0.85)
        axes[0].axvline(x=0, color='black', linewidth=1)
        axes[0].set_title("Feature Correlation with Energy Consumption\n(Pearson Coefficient)",
                           fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Correlation Coefficient", fontsize=11)
        for bar, val in zip(bars, corr.values):
            axes[0].text(val + (0.008 if val >= 0 else -0.008), bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va='center',
                         ha='left' if val >= 0 else 'right', fontsize=9)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        top_features = numeric_df.corr()['energy_consumption'].abs().nlargest(7).index.tolist()
        sns.heatmap(numeric_df[top_features].corr(), annot=True, cmap="RdBu_r",
                    center=0, fmt=".2f", ax=axes[1], square=True, linewidths=0.5)
        axes[1].set_title("Correlation Matrix — Top 7 Features", fontsize=13, fontweight='bold')
        plt.setp(axes[1].get_xticklabels(), rotation=30, ha='right')

        plt.tight_layout()
        self.embed_figure(fig, parent)

    # --- Tab 6: Distribution Analysis ---
    def plot_distribution(self, parent):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].hist(self.df['energy_consumption'], bins=50, color='#27ae60',
                        alpha=0.7, edgecolor='white', density=True)
        self.df['energy_consumption'].plot.kde(ax=axes[0, 0], color='#2c3e50', linewidth=2)
        mean_v = self.df['energy_consumption'].mean()
        median_v = self.df['energy_consumption'].median()
        axes[0, 0].axvline(mean_v, color='#e74c3c', linestyle='--', linewidth=1.5,
                           label=f"Mean: {mean_v:,.0f}")
        axes[0, 0].axvline(median_v, color='#3498db', linestyle='--', linewidth=1.5,
                           label=f"Median: {median_v:,.0f}")
        axes[0, 0].set_title("Energy Consumption Distribution", fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel("Energy Consumption (kWh)", fontsize=11)
        axes[0, 0].set_ylabel("Density", fontsize=11)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)

        stats = self.df['energy_consumption'].describe()
        table_data = [
            ['Count', f"{stats['count']:,.0f}"],
            ['Mean', f"{stats['mean']:,.2f} kWh"],
            ['Std Dev', f"{stats['std']:,.2f} kWh"],
            ['Min', f"{stats['min']:,.2f} kWh"],
            ['25th %ile', f"{stats['25%']:,.2f} kWh"],
            ['Median', f"{stats['50%']:,.2f} kWh"],
            ['75th %ile', f"{stats['75%']:,.2f} kWh"],
            ['Max', f"{stats['max']:,.2f} kWh"],
        ]
        axes[0, 1].axis('off')
        table = axes[0, 1].table(cellText=table_data, colLabels=['Statistic', 'Value'],
                                  cellLoc='center', loc='center', bbox=[0.05, 0.05, 0.9, 0.9])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(color='white', fontweight='bold')
            elif row % 2 == 0:
                cell.set_facecolor('#f8f9fa')
        axes[0, 1].set_title("Descriptive Statistics", fontsize=13, fontweight='bold', pad=15)

        top8 = self.df.groupby('country')['energy_consumption'].mean().nlargest(8).index
        sns.violinplot(data=self.df[self.df['country'].isin(top8)],
                       x='country', y='energy_consumption',
                       palette='Set2', ax=axes[1, 0], inner='quartile')
        axes[1, 0].set_title("Energy Distribution — Top 8 Countries (Violin Plot)",
                              fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel("")
        axes[1, 0].set_ylabel("Energy Consumption (kWh)", fontsize=11)
        plt.setp(axes[1, 0].get_xticklabels(), rotation=30, ha='right', fontsize=9)
        axes[1, 0].spines['top'].set_visible(False)
        axes[1, 0].spines['right'].set_visible(False)

        pivot = self.df.groupby(['country', 'Month'])['energy_consumption'].mean().unstack()
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        sns.heatmap(pivot, cmap="YlOrRd", annot=False, ax=axes[1, 1],
                    cbar_kws={'label': 'Avg Energy (kWh)'}, linewidths=0.3)
        axes[1, 1].set_title("Average Energy — Country × Month Heatmap",
                              fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel("Month", fontsize=11)
        axes[1, 1].set_ylabel("")
        plt.setp(axes[1, 1].get_yticklabels(), fontsize=8)

        plt.tight_layout()
        self.embed_figure(fig, parent)

    def on_closing(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedEDAApp(root)
    root.mainloop()