import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    DEFAULT_MODELS = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=10,
                                                       n_jobs=-1, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100,
                                                           max_depth=4, random_state=42),
    }

    MODEL_COLORS = {
        'Logistic Regression': '#3498db',
        'Decision Tree':       '#e67e22',
        'Random Forest':       '#2ecc71',
        'Gradient Boosting':   '#9b59b6',
    }

    def __init__(self, models=None):
        self.models = models or self.DEFAULT_MODELS
        self.results = {}
        self.bootstrap_curves = {}
        self._eval_label = 'test'   # 'val' hoặc 'test' — dùng để label trục chart

    # ── PUBLIC: train tất cả ──────────────────────────────
    def fit(self, X_train, y_train, X_test, y_test,
            X_val=None, y_val=None,
            run_bootstrap=True, n_points=12, n_iters=3, run_cv=True):
        """
        Train tất cả models, tính metrics, chạy bootstrap và cross-validation.

        Parameters
        ----------
        X_train, y_train : training data
        X_test,  y_test  : test data — dùng để đánh giá cuối cùng (báo cáo)
        X_val,   y_val   : validation data (optional)
                           Nếu có → bootstrap curves dùng val để track learning
                           Nếu không → dùng test (chấp nhận được khi không có val)
        run_bootstrap    : chạy bootstrap learning curve hay không
        n_points         : số epoch trong learning curve (5% → 100% data)
        n_iters          : số bootstrap runs tại mỗi epoch → tính mean ± std
        run_cv           : chạy 5-fold cross-validation hay không
        """
        # Convert về numpy
        def to_np(a): return a.values if hasattr(a, 'values') else np.array(a)
        X_train, y_train = to_np(X_train), to_np(y_train)
        X_test,  y_test  = to_np(X_test),  to_np(y_test)

        # Val set: dùng để theo dõi learning (tránh leakage vào test)
        if X_val is not None:
            X_val, y_val = to_np(X_val), to_np(y_val)
            X_eval, y_eval = X_val, y_val
            self._eval_label = 'val'
        else:
            X_eval, y_eval = X_test, y_test
            self._eval_label = 'test'

        # In tóm tắt
        line = f"Train: {X_train.shape}"
        if X_val is not None: line += f" | Val: {X_val.shape}"
        line += f" | Test: {X_test.shape}"
        print(line + "\n")

        for name, model in self.models.items():
            print(f"Training {name}...", end=' ')
            t0 = time.time()

            # Train đầy đủ trên toàn bộ X_train
            model.fit(X_train, y_train)

            # Evaluate trên TEST (kết quả cuối cùng)
            y_pred_test = model.predict(X_test)
            acc_test = accuracy_score(y_test, y_pred_test)
            f1_test  = f1_score(y_test, y_pred_test, average='macro')

            # Evaluate trên VAL (nếu có)
            acc_val, f1_val = None, None
            if X_val is not None:
                y_pred_val = model.predict(X_val)
                acc_val = round(accuracy_score(y_val, y_pred_val), 4)
                f1_val  = round(f1_score(y_val, y_pred_val, average='macro'), 4)

            # Cross-validation (trên train)
            cv_mean, cv_std = None, None
            if run_cv:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_s = cross_val_score(clone(model), X_train, y_train,
                                       cv=cv, scoring='accuracy', n_jobs=-1)
                cv_mean = round(cv_s.mean(), 4)
                cv_std  = round(cv_s.std(),  4)

            self.results[name] = {
                'model':    model,
                'y_pred':   y_pred_test,
                'accuracy': round(acc_test, 4),
                'f1_macro': round(f1_test, 4),
                'acc_val':  acc_val,
                'f1_val':   f1_val,
                'cv_mean':  cv_mean,
                'cv_std':   cv_std,
                'time':     round(time.time() - t0, 1),
            }

            # Bootstrap Learning Curve — dùng X_eval (val nếu có, test nếu không)
            if run_bootstrap:
                self.bootstrap_curves[name] = self._bootstrap_curve(
                    model, X_train, y_train, X_eval, y_eval, n_points, n_iters
                )

            val_str = f"  val_acc={acc_val:.4f}" if acc_val else ""
            cv_str  = f"  cv={cv_mean:.4f}±{cv_std:.4f}" if cv_mean else ""
            print(f"test_acc={acc_test:.4f}{val_str}  f1={f1_test:.4f}{cv_str}  ({time.time()-t0:.1f}s)")

        self.best_name = max(self.results, key=lambda n: self.results[n]['accuracy'])
        print(f"\n Best: {self.best_name} (test_acc={self.results[self.best_name]['accuracy']:.4f})")
        return self

    # ── Bootstrap Learning Curve ──────────────────────────
    def _bootstrap_curve(self, model, X_tr, y_tr, X_eval, y_eval, n_points, n_iters):
        """
        Tại mỗi 'epoch' (% data tăng dần), chạy n_iters lần với bootstrap sampling.
        Trả về mean ± std của accuracy và F1 tại mỗi epoch.

        Nếu có val set → X_eval = X_val (sạch, chưa model thấy)
        Nếu không      → X_eval = X_test (chấp nhận được)

        Ý nghĩa vùng bóng (±std):
        - Hẹp = model ổn định, không nhạy với sampling
        - Rộng = model phụ thuộc nhiều vào data (high variance)
        """
        fracs = np.linspace(0.05, 1.0, n_points)
        results = []
        eval_n = min(2000, len(X_eval))

        for frac in fracs:
            n = max(50, int(len(X_tr) * frac))
            accs, f1s = [], []
            for _ in range(n_iters):
                idx = np.random.choice(len(X_tr), size=min(n, 5000), replace=True)
                m = clone(model)
                m.fit(X_tr[idx], y_tr[idx])
                yp = m.predict(X_eval[:eval_n])
                accs.append(accuracy_score(y_eval[:eval_n], yp))
                f1s.append(f1_score(y_eval[:eval_n], yp, average='macro'))
            results.append({
                'pct':      round(frac * 100),
                'acc_mean': round(np.mean(accs), 4),
                'acc_std':  round(np.std(accs), 4),
                'f1_mean':  round(np.mean(f1s), 4),
                'f1_std':   round(np.std(f1s), 4),
            })
        return results

    # ── PLOT: Bootstrap Curves ────────────────────────────
    def plot_bootstrap_curves(self):
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f'Bootstrap Learning Curves  (eval on {self._eval_label} set)\n'
            'Vùng bóng = mean ± std  |  Hẹp = ổn định  |  Rộng = high variance',
            fontsize=12, fontweight='bold'
        )

        for name, curves in self.bootstrap_curves.items():
            c = self.MODEL_COLORS.get(name, 'gray')
            x         = [r['pct'] for r in curves]
            acc_means = np.array([r['acc_mean'] for r in curves])
            acc_stds  = np.array([r['acc_std']  for r in curves])
            f1_means  = np.array([r['f1_mean']  for r in curves])
            f1_stds   = np.array([r['f1_std']   for r in curves])

            ax1.plot(x, acc_means, label=name, color=c, linewidth=2.5, marker='o', markersize=4)
            ax1.fill_between(x, acc_means - acc_stds, acc_means + acc_stds, alpha=0.15, color=c)

            ax2.plot(x, f1_means, label=name, color=c, linewidth=2.5, marker='s', markersize=4)
            ax2.fill_between(x, f1_means - f1_stds, f1_means + f1_stds, alpha=0.15, color=c)

        for ax in [ax1, ax2]:
            ax.axhline(y=0.333, color='red', linestyle='--', alpha=0.5, label='Random baseline')
            ax.set_xlabel('% Training Data Used')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        ax1.set_title(f'Accuracy ({self._eval_label})')
        ax1.set_ylabel('Accuracy')
        ax2.set_title(f'F1 Macro ({self._eval_label})')
        ax2.set_ylabel('F1 Macro')
        plt.tight_layout()
        plt.show()

    # ── PLOT: Model Comparison ────────────────────────────
    def plot_comparison(self):
        import matplotlib.pyplot as plt

        names  = list(self.results.keys())
        accs   = [self.results[n]['accuracy'] * 100 for n in names]
        f1s    = [self.results[n]['f1_macro']  * 100 for n in names]
        # Val acc nếu có
        val_accs = [self.results[n]['acc_val'] * 100
                    if self.results[n]['acc_val'] else None for n in names]
        colors = [self.MODEL_COLORS.get(n, 'gray') for n in names]
        x = np.arange(len(names))
        has_val = any(v is not None for v in val_accs)

        ncols = 3 if has_val else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))
        fig.suptitle('Model Comparison', fontsize=13, fontweight='bold')

        def _bar(ax, values, title, ylabel):
            bars = ax.bar(x, values, color=colors, edgecolor='white', alpha=0.85, width=0.6)
            ax.axhline(y=33.33, color='red', linestyle='--', linewidth=1.5, label='Random baseline')
            ax.set_xticks(x)
            ax.set_xticklabels([n.replace(' ', '\n') for n in names])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(min(values)-2, max(values)+3)
            ax.legend(); ax.grid(True, axis='y', alpha=0.3)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')

        _bar(axes[0], accs, 'Accuracy (test)', 'Accuracy (%)')
        _bar(axes[1], f1s,  'F1 Macro (test)', 'F1 Macro (%)')
        if has_val:
            _bar(axes[2], [v for v in val_accs if v],
                 'Accuracy (val)', 'Accuracy (%)')

        plt.tight_layout()
        plt.show()

    # ── PLOT: Confusion Matrices ──────────────────────────
    def plot_confusion_matrices(self, target_classes=None):
        import matplotlib.pyplot as plt
        import seaborn as sns

        n = len(self.results)
        ncols = min(2, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows))
        fig.suptitle('Confusion Matrices (test set)', fontsize=13, fontweight='bold')
        axes = np.array(axes).flatten()

        # Cần y_test từ bên ngoài → truyền vào khi gọi hàm này
        # Nếu không có, in warning
        print(" Gọi trực tiếp từ notebook với y_test — xem cell bên dưới")

    # ── Summary Table ─────────────────────────────────────
    def summary(self):
        rows = []
        for name, res in self.results.items():
            row = {'Model': name,
                   'Test Acc':   res['accuracy'],
                   'Test F1':    res['f1_macro'],
                   'Val Acc':    res['acc_val'],
                   'Val F1':     res['f1_val'],
                   'CV Mean':    res['cv_mean'],
                   'CV Std':     res['cv_std'],
                   'Time (s)':   res['time']}
            rows.append(row)
        df = pd.DataFrame(rows).set_index('Model').sort_values('Test Acc', ascending=False)
        print(df.to_string())
        return df

    def get_best_model(self):
        return self.results[self.best_name]['model']

    def get_model(self, name):
        return self.results[name]['model']
