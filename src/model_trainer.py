"""
Tính năng:
    - Train đa mô hình với thống nhất interface
    - Cross-validation (Stratified K-Fold) trên train set
    - Bootstrap learning curves với mean ± std (stability analysis)
    - Confusion matrix đầy đủ — đã fix bug cũ (method bị bỏ trống)
    - Summary table so sánh Test Acc / Val Acc / F1 / CV / Time
    - CV vs Test gap report để phát hiện overfitting ẩn

Lưu ý về Bootstrap Curve:
    Nếu có val set → dùng val để vẽ learning curve (hoàn toàn sạch).
    Nếu không có   → dùng test (acceptable, nhưng nên có val khi có thể).
    Mỗi điểm = mean ± std của n_iters lần bootstrap — vùng bóng hẹp = model ổn định.
"""

import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import clone

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Huấn luyện và so sánh nhiều mô hình ML.

    Ví dụ sử dụng:
        trainer = ModelTrainer()
        trainer.fit(X_train, y_train, X_test, y_test, X_val, y_val)

        trainer.summary()
        trainer.plot_comparison()
        trainer.plot_confusion_matrices(y_test, class_names=['Normal', 'Abnormal', 'Inconclusive'])
        trainer.plot_bootstrap_curves()
        trainer.print_cv_gap_report()   # kiểm tra overfitting ẩn
    """

    DEFAULT_MODELS = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest':       RandomForestClassifier(
                                   n_estimators=100, max_depth=10,
                                   n_jobs=-1, random_state=42
                               ),
        'Gradient Boosting':   GradientBoostingClassifier(
                                   n_estimators=100, max_depth=4, random_state=42
                               ),
    }

    MODEL_COLORS = {
        'Logistic Regression': '#3498db',
        'Decision Tree':       '#e67e22',
        'Random Forest':       '#2ecc71',
        'Gradient Boosting':   '#9b59b6',
    }

    def __init__(self, models: dict = None):
        """
        Parameters
        ----------
        models : dict tùy chỉnh {tên: sklearn_estimator}.
                 Nếu None → dùng DEFAULT_MODELS.
        """
        # Clone tất cả model để tránh state pollution giữa các lần fit
        raw = models or self.DEFAULT_MODELS
        self.models = {name: clone(m) for name, m in raw.items()}

        self.results          = {}   # {model_name: metrics_dict}
        self.bootstrap_curves = {}   # {model_name: list of per-epoch dicts}
        self._eval_label      = 'test'
        self._y_test          = None  # lưu lại để plot_confusion_matrices dùng
        self.best_name        = None

    def fit(
        self,
        X_train, y_train,
        X_test,  y_test,
        X_val=None, y_val=None,
        run_bootstrap: bool = True,
        n_points: int = 12,
        n_iters:  int = 3,
        run_cv:   bool = True,
    ):
        """
        Train tất cả models, đánh giá, và vẽ learning curves.

        Parameters
        ----------
        X_train, y_train : training data (numpy array hoặc DataFrame)
        X_test,  y_test  : test data — dùng để báo cáo kết quả cuối
        X_val,   y_val   : validation data (optional nhưng khuyến khích)
                           Nếu có → bootstrap dùng val (hoàn toàn sạch)
        run_bootstrap    : có chạy bootstrap learning curves không
        n_points         : số điểm trên trục % data (5% → 100%)
        n_iters          : số lần bootstrap tại mỗi điểm → tính mean ± std
        run_cv           : có chạy 5-Fold Stratified CV không

        Returns
        -------
        self — để chain: trainer.fit(...).summary()
        """
        def _to_np(a):
            return a.values if hasattr(a, 'values') else np.asarray(a)

        X_train, y_train = _to_np(X_train), _to_np(y_train)
        X_test,  y_test  = _to_np(X_test),  _to_np(y_test)
        self._y_test = y_test   # lưu lại để dùng trong plot_confusion_matrices

        if X_val is not None:
            X_val, y_val     = _to_np(X_val), _to_np(y_val)
            X_eval, y_eval   = X_val, y_val
            self._eval_label = 'val'
        else:
            X_eval, y_eval   = X_test, y_test
            self._eval_label = 'test'

    
        header = f"Train: {X_train.shape}"
        if X_val is not None:
            header += f" | Val: {X_val.shape}"
        header += f" | Test: {X_test.shape}"
        print("\n" + "=" * 60)
        print(f"  {header}")
        print("=" * 60)

        for name, model in self.models.items():
            print(f"  Training {name}...", end=" ", flush=True)
            t0 = time.time()


            model.fit(X_train, y_train)

            y_pred_test = model.predict(X_test)
            acc_test = accuracy_score(y_test, y_pred_test)
            f1_test  = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

        
            acc_val, f1_val = None, None
            if X_val is not None:
                y_pred_val = model.predict(X_val)
                acc_val = round(accuracy_score(y_val, y_pred_val), 4)
                f1_val  = round(f1_score(y_val, y_pred_val, average='macro', zero_division=0), 4)

          
            cv_mean, cv_std = None, None
            if run_cv:
                cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(
                    clone(model), X_train, y_train,
                    cv=cv, scoring='accuracy', n_jobs=-1
                )
                cv_mean = round(float(cv_scores.mean()), 4)
                cv_std  = round(float(cv_scores.std()),  4)

            elapsed = round(time.time() - t0, 1)
            self.results[name] = {
                'model':    model,
                'y_pred':   y_pred_test,
                'accuracy': round(acc_test, 4),
                'f1_macro': round(f1_test, 4),
                'acc_val':  acc_val,
                'f1_val':   f1_val,
                'cv_mean':  cv_mean,
                'cv_std':   cv_std,
                'time':     elapsed,
            }

            # Bootstrap learning curves — dùng clone() nhất quán
            if run_bootstrap:
                self.bootstrap_curves[name] = self._bootstrap_curve(
                    model, X_train, y_train, X_eval, y_eval, n_points, n_iters
                )

            val_str = f"  val={acc_val:.4f}" if acc_val is not None else ""
            cv_str  = f"  cv={cv_mean:.4f}±{cv_std:.4f}" if cv_mean is not None else ""
            print(f"test={acc_test:.4f}  f1={f1_test:.4f}{val_str}{cv_str}  ({elapsed}s)")

        self.best_name = max(self.results, key=lambda n: self.results[n]['accuracy'])
        print(f"\n  Best: {self.best_name}  "
              f"(test_acc={self.results[self.best_name]['accuracy']:.4f})")
        print("=" * 60 + "\n")

        logger.info("Training hoàn tất. Best model: %s", self.best_name)
        return self


    def plot_confusion_matrices(self, y_test=None, class_names: list = None):
        """
        Vẽ confusion matrix cho tất cả các model.

        Parameters
        ----------
        y_test       : ground truth labels (numpy array).
                       Nếu None → dùng y_test đã lưu từ lần fit() gần nhất.
        class_names  : danh sách tên class (ví dụ ['Normal', 'Abnormal', 'Inconclusive']).
                       Nếu None → dùng số nguyên.

        Lý do cần y_test:
            ModelTrainer không lưu toàn bộ y_test vì khi dataset lớn sẽ tốn RAM.
            Cách dùng: trainer.plot_confusion_matrices(y_test, class_names=prep.get_target_classes())
        """
        _y_test = y_test if y_test is not None else self._y_test
        if _y_test is None:
            raise ValueError(
                "Cần truyền y_test vào hàm này. "
                "Ví dụ: trainer.plot_confusion_matrices(y_test, class_names=...)"
            )

        n = len(self.results)
        ncols = min(2, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
        fig.suptitle(
            'Confusion Matrices (test set)\n'
            'Đường chéo = predicted đúng | Ngoài đường chéo = predicted sai',
            fontsize=12, fontweight='bold'
        )
        axes_flat = np.array(axes).flatten()

        for i, (name, res) in enumerate(self.results.items()):
            cm = confusion_matrix(_y_test, res['y_pred'])
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

            sns.heatmap(
                cm_norm, annot=True, fmt='.2f', ax=axes_flat[i],
                cmap='Blues', vmin=0, vmax=1,
                xticklabels=class_names or range(cm.shape[1]),
                yticklabels=class_names or range(cm.shape[0]),
                linewidths=0.5,
            )
            acc = res['accuracy']
            f1  = res['f1_macro']
            axes_flat[i].set_title(
                f"{name}\nAcc={acc:.4f}  F1={f1:.4f}",
                fontsize=10, fontweight='bold'
            )
            axes_flat[i].set_xlabel('Predicted label')
            axes_flat[i].set_ylabel('True label')

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_bootstrap_curves(self):
        """
        Vẽ bootstrap learning curves: Accuracy và F1 theo % data training.
        Vùng bóng = mean ± std qua n_iters lần bootstrap.
        Đường đỏ đứt = random baseline (33.3% cho 3-class).
        """
        if not self.bootstrap_curves:
            print("Chưa có bootstrap curves. Gọi fit() với run_bootstrap=True.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f'Bootstrap Learning Curves  (eval on {self._eval_label} set)\n'
            'Vùng bóng = mean ± std  |  Hẹp = ổn định  |  Rộng = high variance',
            fontsize=12, fontweight='bold'
        )

        for name, curves in self.bootstrap_curves.items():
            color     = self.MODEL_COLORS.get(name, 'gray')
            x         = [r['pct']      for r in curves]
            acc_means = np.array([r['acc_mean'] for r in curves])
            acc_stds  = np.array([r['acc_std']  for r in curves])
            f1_means  = np.array([r['f1_mean']  for r in curves])
            f1_stds   = np.array([r['f1_std']   for r in curves])

            ax1.plot(x, acc_means, label=name, color=color, linewidth=2.5, marker='o', markersize=4)
            ax1.fill_between(x, acc_means - acc_stds, acc_means + acc_stds, alpha=0.15, color=color)

            ax2.plot(x, f1_means, label=name, color=color, linewidth=2.5, marker='s', markersize=4)
            ax2.fill_between(x, f1_means - f1_stds, f1_means + f1_stds, alpha=0.15, color=color)

        for ax in [ax1, ax2]:
            ax.axhline(y=0.333, color='red', linestyle='--', alpha=0.6, label='Random baseline (33.3%)')
            ax.set_xlabel('% Training Data Used')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        ax1.set_title(f'Accuracy ({self._eval_label} set)')
        ax1.set_ylabel('Accuracy')
        ax2.set_title(f'F1 Macro ({self._eval_label} set)')
        ax2.set_ylabel('F1 Macro')

        plt.tight_layout()
        plt.show()

    def plot_comparison(self):
        """
        Vẽ bar chart so sánh Accuracy (test), F1 Macro (test), và Accuracy (val nếu có).
        Đường đỏ đứt = random baseline.
        """
        if not self.results:
            print("Chưa có kết quả. Hãy gọi fit() trước.")
            return

        names    = list(self.results.keys())
        accs     = [self.results[n]['accuracy'] * 100 for n in names]
        f1s      = [self.results[n]['f1_macro']  * 100 for n in names]
        val_accs = [
            self.results[n]['acc_val'] * 100
            if self.results[n]['acc_val'] is not None else None
            for n in names
        ]
        colors   = [self.MODEL_COLORS.get(n, 'gray') for n in names]
        x        = np.arange(len(names))
        has_val  = any(v is not None for v in val_accs)

        ncols = 3 if has_val else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
        fig.suptitle('Model Comparison', fontsize=13, fontweight='bold')

        def _bar(ax, values, title, ylabel):
            bars = ax.bar(x, values, color=colors, edgecolor='white', alpha=0.85, width=0.6)
            ax.axhline(y=33.33, color='red', linestyle='--', linewidth=1.5,
                       label='Random baseline (33.3%)')
            ax.set_xticks(x)
            ax.set_xticklabels([n.replace(' ', '\n') for n in names])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(max(0, min(values) - 3), min(100, max(values) + 5))
            ax.legend(fontsize=9)
            ax.grid(True, axis='y', alpha=0.3)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold'
                )

        _bar(axes[0], accs, 'Accuracy (test)', 'Accuracy (%)')
        _bar(axes[1], f1s,  'F1 Macro (test)', 'F1 Macro (%)')
        if has_val:
            clean_val = [v if v is not None else 0 for v in val_accs]
            _bar(axes[2], clean_val, 'Accuracy (val)', 'Accuracy (%)')

        plt.tight_layout()
        plt.show()

    def summary(self) -> pd.DataFrame:
        """
        In và trả về bảng tóm tắt tất cả metrics.

        Columns: Test Acc, Test F1, Val Acc, Val F1, CV Mean, CV Std, CV-Test Gap, Time(s)

        CV-Test Gap:
            Gap lớn (> 0.03) = model ổn định trên CV nhưng kém hơn trên test → có thể
            có distribution shift hoặc train set không đại diện tốt cho test set.
        """
        rows = []
        for name, res in self.results.items():
            cv_test_gap = None
            if res['cv_mean'] is not None:
                cv_test_gap = round(res['cv_mean'] - res['accuracy'], 4)

            rows.append({
                'Model':        name,
                'Test Acc':     res['accuracy'],
                'Test F1':      res['f1_macro'],
                'Val Acc':      res['acc_val'],
                'Val F1':       res['f1_val'],
                'CV Mean':      res['cv_mean'],
                'CV Std':       res['cv_std'],
                'CV-Test Gap':  cv_test_gap,
                'Time (s)':     res['time'],
            })

        df = (
            pd.DataFrame(rows)
            .set_index('Model')
            .sort_values('Test Acc', ascending=False)
        )
        print("\n" + df.to_string() + "\n")
        return df

    def print_classification_reports(self, y_test=None, class_names: list = None):
        """
        In classification report chi tiết (precision, recall, F1 per class) cho mỗi model.

        Quan trọng trên noisy data: nếu model bị bias về 1 class
        (predict toàn 'Normal'), macro F1 sẽ thấp dù accuracy có vẻ ổn.
        """
        _y_test = y_test if y_test is not None else self._y_test
        if _y_test is None:
            raise ValueError("Cần truyền y_test vào hàm này.")

        for name, res in self.results.items():
            print(f"\n{'='*50}")
            print(f"  {name}")
            print('=' * 50)
            print(classification_report(
                _y_test, res['y_pred'],
                target_names=class_names,
                zero_division=0
            ))

    def print_cv_gap_report(self):
        """
        In báo cáo CV vs Test gap để phát hiện overfitting ẩn.

        Nếu CV mean >> Test accuracy: model overfits trên train folds.
        Nếu CV mean ≈ Test accuracy: model generalizes tốt.
        """
        if not self.results:
            print("Chưa có kết quả.")
            return

        print("\n" + "=" * 60)
        print("  CV vs Test Gap Report")
        print("  (Gap lớn > 0.03 → cần xem xét overfitting hoặc distribution shift)")
        print("=" * 60)
        for name, res in self.results.items():
            if res['cv_mean'] is not None:
                gap = res['cv_mean'] - res['accuracy']
                flag = "⚠️  Xem lại" if abs(gap) > 0.03 else "✅ OK"
                print(
                    f"  {name:<22}  "
                    f"CV={res['cv_mean']:.4f}±{res['cv_std']:.4f}  "
                    f"Test={res['accuracy']:.4f}  "
                    f"Gap={gap:+.4f}  {flag}"
                )
        print("=" * 60 + "\n")

    def get_best_model(self):
        """Trả về sklearn estimator của best model (theo test accuracy)."""
        if not self.best_name:
            raise RuntimeError("Chưa train. Hãy gọi fit() trước.")
        return self.results[self.best_name]['model']

    def get_model(self, name: str):
        """Trả về sklearn estimator theo tên."""
        if name not in self.results:
            raise KeyError(f"Model '{name}' không tồn tại. Các model hiện có: {list(self.results.keys())}")
        return self.results[name]['model']


    def _bootstrap_curve(self, model, X_tr, y_tr, X_eval, y_eval, n_points, n_iters):
        """
        Tạo bootstrap learning curve.

        Tại mỗi fraction (5% → 100% data), train n_iters lần với bootstrap sampling
        → tính mean và std của accuracy và F1.

        Vùng bóng trong plot:
            Hẹp → model ổn định, ít nhạy cảm với sampling noise
            Rộng → model có high variance, cần thêm data hoặc regularization

        Dùng clone(model) mỗi lần để đảm bảo independent training.
        """
        fracs  = np.linspace(0.05, 1.0, n_points)
        eval_n = min(2000, len(X_eval))
        curve  = []

        for frac in fracs:
            n = max(50, int(len(X_tr) * frac))
            accs, f1s = [], []

            for _ in range(n_iters):
                idx = np.random.choice(len(X_tr), size=min(n, 5000), replace=True)
                m   = clone(model)
                m.fit(X_tr[idx], y_tr[idx])

                yp = m.predict(X_eval[:eval_n])
                accs.append(accuracy_score(y_eval[:eval_n], yp))
                f1s.append(f1_score(y_eval[:eval_n], yp, average='macro', zero_division=0))

            curve.append({
                'pct':      round(frac * 100),
                'acc_mean': round(float(np.mean(accs)), 4),
                'acc_std':  round(float(np.std(accs)),  4),
                'f1_mean':  round(float(np.mean(f1s)),  4),
                'f1_std':   round(float(np.std(f1s)),   4),
            })

        return curve