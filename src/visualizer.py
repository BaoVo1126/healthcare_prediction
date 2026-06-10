"""
AutoVisualizer — tự động tạo toàn bộ biểu đồ EDA cho một DataFrame.

Sử dụng:
    viz = AutoVisualizer(df, target_col='Test Results')
    viz.run()          # chạy tất cả plots

    # Hoặc chọn lọc từng phần:
    viz.plot_target()
    viz.plot_numeric_distributions()
    viz.plot_correlation()
    viz.plot_outliers()
    viz.plot_categorical()
    viz.plot_target_vs_features()
    viz.plot_seasonality()    # phân tích tính mùa vụ theo thời gian (mới)
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class AutoVisualizer:
    """
    Tạo tự động tất cả biểu đồ EDA cơ bản.

    Parameters
    ----------
    df         : DataFrame gốc (trước hoặc sau preprocess đều được)
    target_col : tên cột nhãn mục tiêu
    """

    def __init__(self, df: pd.DataFrame, target_col: str = None):
        self.df         = df.copy()
        self.target_col = target_col

        feature_cols    = [c for c in df.columns if c != target_col]
        self.num_cols   = df[feature_cols].select_dtypes(include='number').columns.tolist()
        self.cat_cols   = df[feature_cols].select_dtypes(include='object').columns.tolist()
        self.date_cols  = df[feature_cols].select_dtypes(include='datetime').columns.tolist()

        for col in feature_cols:
            if 'date' in col.lower() and col not in self.date_cols:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.date_cols.append(col)
                except Exception:
                    pass

    def run(self):
        """Chạy toàn bộ pipeline visualisation."""
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
        if self.date_cols and self.target_col:
            self.plot_seasonality()


    def plot_target(self):
        """Bar + Pie chart phân phối nhãn target."""
        col = self.target_col
        s   = self.df[col]
        vc  = s.value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Target Distribution: "{col}"', fontsize=13, fontweight='bold')

        colors = ['#e74c3c', '#2ecc71', '#f39c12', '#3498db', '#9b59b6'][:len(vc)]

        # Bar
        vc.plot(kind='bar', ax=axes[0], color=colors, edgecolor='white', width=0.6)
        axes[0].set_title('Count per class')
        axes[0].set_xticklabels(vc.index, rotation=0)
        axes[0].set_xlabel('')
        for i, v in enumerate(vc.values):
            axes[0].text(i, v + len(s) * 0.005, f'{v:,}\n({v/len(s)*100:.1f}%)',
                         ha='center', fontsize=9, fontweight='bold')
        axes[0].grid(True, axis='y', alpha=0.3)

        # Pie
        vc.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                colors=colors, startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        axes[1].set_title('Proportion')
        axes[1].set_ylabel('')

        plt.tight_layout()
        plt.show()

    def plot_numeric_distributions(self):
        """Histogram + KDE + mean line cho từng cột numeric."""
        cols  = self.num_cols
        n     = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        fig.suptitle('Numeric Distributions (Histogram + KDE)', fontsize=13, fontweight='bold')
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(cols):
            s = self.df[col].dropna()
            axes_flat[i].hist(s, bins=30, color='steelblue', edgecolor='white',
                              alpha=0.7, density=True)
            try:
                s.plot.kde(ax=axes_flat[i], color='red', linewidth=2)
            except Exception:
                pass
            axes_flat[i].axvline(s.mean(), color='orange', linestyle='--',
                                 linewidth=1.5, label=f'mean={s.mean():.1f}')
            axes_flat[i].axvline(s.median(), color='green', linestyle=':',
                                 linewidth=1.5, label=f'median={s.median():.1f}')
            axes_flat[i].set_title(f'{col}  (skew={s.skew():.2f})', fontsize=10)
            axes_flat[i].legend(fontsize=8)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_correlation(self):
        """Correlation heatmap cho các cột numeric (tam giác dưới)."""
        if len(self.num_cols) < 2:
            logger.info("Cần ít nhất 2 cột numeric để vẽ correlation.")
            return

        corr = self.df[self.num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(max(8, len(self.num_cols) * 1.5), max(6, len(self.num_cols) * 1.2)))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, square=True,
            linewidths=0.5, vmin=-1, vmax=1,
        )
        plt.title('Correlation Heatmap (numeric features)', fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_outliers(self):
        """Boxplot cho từng cột numeric — highlight outlier rate > 5%."""
        cols  = self.num_cols
        n     = len(cols)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        fig.suptitle('Outlier Detection (Boxplots — IQR method)', fontsize=13, fontweight='bold')
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(cols):
            s      = self.df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr    = q3 - q1
            n_out  = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            pct    = n_out / len(s)
            color  = '#e74c3c' if pct > 0.05 else '#2ecc71'

            axes_flat[i].boxplot(
                s, patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.5),
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markersize=3, alpha=0.4),
            )
            title = (
                f'{col}\n  {n_out} outliers ({pct*100:.1f}%)'
                if n_out > 0 else f'{col}\n OK'
            )
            axes_flat[i].set_title(title, fontsize=9)
            axes_flat[i].set_xticks([])

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.show()


    def plot_categorical(self, max_categories: int = 15):
        """
        Bar chart cho từng cột categorical.
        Bỏ qua cột có cardinality quá cao (> max_categories unique values).
        """
        plot_cols = [
            c for c in self.cat_cols
            if self.df[c].nunique() <= max_categories
        ]
        skipped = [c for c in self.cat_cols if c not in plot_cols]
        if skipped:
            logger.info("Bỏ qua các cột high-cardinality: %s", skipped)

        if not plot_cols:
            print("Không có cột categorical nào phù hợp để vẽ.")
            return

        n     = len(plot_cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        fig.suptitle('Categorical Distributions', fontsize=13, fontweight='bold')
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(plot_cols):
            vc = self.df[col].value_counts()
            vc.plot(kind='bar', ax=axes_flat[i],
                    color='steelblue', edgecolor='white', width=0.7)
            axes_flat[i].set_title(f'{col}  ({self.df[col].nunique()} unique)', fontsize=10)
            axes_flat[i].set_xticklabels(vc.index, rotation=30, ha='right')
            axes_flat[i].grid(True, axis='y', alpha=0.3)
            for j, v in enumerate(vc.values):
                axes_flat[i].text(j, v + len(self.df) * 0.003, f'{v/len(self.df)*100:.1f}%',
                                  ha='center', fontsize=7)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.show()


    def plot_target_vs_features(self, max_features: int = 6):
        """
        Violin plot: phân phối của từng numeric feature, breakdown theo target class.
        Nếu feature thực sự ảnh hưởng đến target → các violin sẽ có hình dạng khác nhau.
        Nếu giống nhau → feature không có tín hiệu (Pure Noise confirmation).
        """
        plot_cols = self.num_cols[:max_features]
        n     = len(plot_cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        fig.suptitle(
            f'Feature Distribution by Target Class: "{self.target_col}"\n'
            'Các violin giống nhau → feature không phân biệt được target class (Pure Noise)',
            fontsize=12, fontweight='bold'
        )
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(plot_cols):
            try:
                sns.violinplot(
                    data=self.df, x=self.target_col, y=col,
                    ax=axes_flat[i], palette='Set2',
                    inner='quartile', cut=0,
                )
            except Exception:
                sns.boxplot(data=self.df, x=self.target_col, y=col,
                            ax=axes_flat[i], palette='Set2')
            axes_flat[i].set_title(col, fontsize=10)
            axes_flat[i].set_xlabel('')

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_seasonality(self, date_col: str = None):
        """
        Phân tích tính mùa vụ: số lượng records theo tháng, phân theo target class.

        Nếu dataset có tính mùa vụ (ví dụ: mùa cúm) → đường sẽ có đỉnh rõ ràng.
        Nếu phẳng lỳ → dataset không có tính mùa vụ (synthetic data confirmation).

        Parameters
        ----------
        date_col : tên cột datetime. Nếu None → dùng cột đầu tiên tìm được.
        """
        col = date_col or (self.date_cols[0] if self.date_cols else None)
        if not col:
            print("Không tìm thấy cột datetime để vẽ seasonality.")
            return

        df_work = self.df.copy()
        df_work[col] = pd.to_datetime(df_work[col], errors='coerce')
        df_work['_month'] = df_work[col].dt.to_period('M')

        if self.target_col:
            monthly = (
                df_work.groupby(['_month', self.target_col])
                .size()
                .unstack(fill_value=0)
            )
        else:
            monthly = df_work.groupby('_month').size().rename('Count').to_frame()

        fig, ax = plt.subplots(figsize=(14, 4))
        monthly.plot(ax=ax, linewidth=2, marker='o', markersize=3)
        ax.set_title(
            f'Monthly Record Count by "{self.target_col or "Total"}"\n'
            'Đường phẳng = không có tính mùa vụ | Có đỉnh rõ = có seasonality',
            fontweight='bold'
        )
        ax.set_xlabel('Tháng')
        ax.set_ylabel('Số lượng records')
        ax.grid(True, alpha=0.3)
        ax.legend(title=self.target_col, fontsize=9)


        peak_month = monthly.sum(axis=1).idxmax()
        peak_val   = monthly.sum(axis=1).max()
        ax.axvline(x=monthly.index.get_loc(peak_month), color='red',
                   linestyle='--', alpha=0.5, label=f'Peak: {peak_month}')

        plt.tight_layout()
        plt.show()