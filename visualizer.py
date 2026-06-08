import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class AutoVisualizer:
    def __init__(self, df, target_col=None):
        self.df = df.copy()
        self.target_col = target_col
        feature_cols = [c for c in df.columns if c != target_col]
        self.num_cols = df[feature_cols].select_dtypes(include='number').columns.tolist()
        self.cat_cols = df[feature_cols].select_dtypes(include='object').columns.tolist()

    def run(self):
        if self.target_col:
            self.plot_target()
        if self.num_cols:
            self.plot_numeric_distributions()
            self.plot_correlation()
            self.plot_outliers()
        if self.cat_cols:
            self.plot_categorical()
        if self.target_col and self.num_cols:
            self.plot_target_vs_features()

    # ── 1. Target Distribution ─────────────────────────────
    def plot_target(self):
        col = self.target_col
        s = self.df[col]
        vc = s.value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Target Distribution: "{col}"', fontsize=13, fontweight='bold')

        # Bar chart
        colors = ['#e74c3c', '#2ecc71', '#f39c12']
        vc.plot(kind='bar', ax=axes[0], color=colors[:len(vc)],
                edgecolor='white', width=0.6)
        axes[0].set_title('Count')
        axes[0].set_xticklabels(vc.index, rotation=0)
        for i, v in enumerate(vc.values):
            axes[0].text(i, v + 100, f'{v:,}', ha='center', fontsize=10, fontweight='bold')

        # Pie chart
        vc.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                colors=colors[:len(vc)], startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        axes[1].set_title('Proportion')
        axes[1].set_ylabel('')

        plt.tight_layout()
        plt.show()

    # ── 2. Numeric Distributions ──────────────────────────
    def plot_numeric_distributions(self):
        cols = self.num_cols
        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
        fig.suptitle('Numeric Distributions (Histogram + KDE)', fontsize=13, fontweight='bold')
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(cols):
            s = self.df[col].dropna()
            axes[i].hist(s, bins=30, color='steelblue', edgecolor='white', alpha=0.7, density=True)
            s.plot.kde(ax=axes[i], color='red', linewidth=2)
            axes[i].axvline(s.mean(), color='orange', linestyle='--',
                            linewidth=1.5, label=f'mean={s.mean():.1f}')
            axes[i].set_title(f'{col}  (skew={s.skew():.2f})', fontsize=10)
            axes[i].legend(fontsize=8)

        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    # ── 3. Correlation Heatmap ─────────────────────────────
    def plot_correlation(self):
        if len(self.num_cols) < 2:
            return
        corr = self.df[self.num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True, linewidths=0.5)
        plt.title('Correlation Heatmap', fontweight='bold')
        plt.tight_layout()
        plt.show()

    # ── 4. Outlier Boxplots ────────────────────────────────
    def plot_outliers(self):
        cols = self.num_cols
        n = len(cols)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        fig.suptitle('Outlier Detection (Boxplots)', fontsize=13, fontweight='bold')
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(cols):
            s = self.df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            n_out = ((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum()
            color = '#e74c3c' if n_out/len(s) > 0.05 else '#2ecc71'
            axes[i].boxplot(s, patch_artist=True,
                            boxprops=dict(facecolor=color, alpha=0.5),
                            medianprops=dict(color='black', linewidth=2))
            title = f'{col}\n {n_out} outliers' if n_out > 0 else f'{col}\n✅ OK'
            axes[i].set_title(title, fontsize=9)
            axes[i].set_xticks([])

        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()

    # ── 5. Categorical Bar Charts ──────────────────────────
    def plot_categorical(self):
        cols = [c for c in self.cat_cols if self.df[c].nunique() <= 30]
        if not cols:
            return
        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
        fig.suptitle('Categorical Feature Distributions', fontsize=13, fontweight='bold')
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(cols):
            vc = self.df[col].value_counts().head(15)
            colors = plt.cm.tab20(np.linspace(0, 1, len(vc)))
            vc.plot(kind='barh', ax=axes[i], color=colors, edgecolor='white')
            axes[i].set_title(col, fontsize=10)
            axes[i].invert_yaxis()

        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()

    # ── 6. Target vs Features (Boxplot per class) ─────────
    def plot_target_vs_features(self):
        target = self.df[self.target_col]
        if target.dtype == 'object' or target.nunique() <= 10:
            cols = self.num_cols[:6]
            n = len(cols)
            ncols = min(3, n)
            nrows = (n + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
            fig.suptitle(f'Numeric Features by Target "{self.target_col}"',
                         fontsize=13, fontweight='bold')
            axes = np.array(axes).flatten() if n > 1 else [axes]

            for i, col in enumerate(cols):
                groups = [self.df[col][target == cls].dropna()
                          for cls in target.unique()]
                axes[i].boxplot(groups, labels=target.unique(), patch_artist=True)
                axes[i].set_title(col, fontsize=10)
                axes[i].tick_params(axis='x', rotation=15)

            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            plt.show()
