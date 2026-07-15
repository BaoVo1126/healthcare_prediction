import pandas as pd
import numpy as np


class DataInspector:
    def __init__(self, df, target_col=None):
        self.df = df.copy()
        self.target_col = target_col

    def run(self):
        self._check_shape()
        self._check_missing()
        self._check_duplicates()
        self._check_dtypes()
        self._check_numeric()
        self._check_categorical()
        if self.target_col:
            self._check_target()

  
    def _check_shape(self):
        rows, cols = self.df.shape
        mem = self.df.memory_usage(deep=True).sum() / 1024**2
        print("SHAPE & MEMORY")
        print(f"  Rows   : {rows:,}")
        print(f"  Columns: {cols}")
        print(f"  Memory : {mem:.2f} MB")

  
    def _check_missing(self):
        miss = self.df.isnull().sum()
        miss_pct = (self.df.isnull().mean() * 100).round(2)

        print("MISSING VALUES")
        has_miss = miss[miss > 0]
        if has_miss.empty:
            print("Không có missing values!")
        else:
            for col in has_miss.index:
                pct = miss_pct[col]
                tag = " DROP?" if pct > 50 else " Cần xử lý" if pct > 10 else " Impute được"
                print(f"  {col:<25} {miss[col]:>6,}  ({pct:.1f}%)  {tag}")

  
    def _check_duplicates(self):
        dup = self.df.duplicated().sum()
        print("DUPLICATE ROWS")
        if dup == 0:
            print("   Không có duplicate rows!")
        else:
            print(f"{dup:,} rows bị trùng → dùng df.drop_duplicates()")

  
    def _check_dtypes(self):
        print("COLUMN TYPES")
        for col, dtype in self.df.dtypes.items():
            n_unique = self.df[col].nunique()
            print(f"  {col:<25} {str(dtype):<10}  ({n_unique:,} unique)")


    def _check_numeric(self):
        num_cols = self.df.select_dtypes(include='number').columns.tolist()
        if not num_cols:
            return
        print("NUMERIC COLUMNS — THỐNG KÊ")
        stats = self.df[num_cols].describe().round(2)
        print(stats.to_string())

        print("\n  Outliers (IQR method):")
        for col in num_cols:
            s = self.df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            n_out = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
            n_neg = (s < 0).sum()
            flag = f"    {n_out} outliers ({n_out/len(s)*100:.1f}%)" if n_out > 0 else "OK"
            neg_flag = f"   {n_neg} âm!" if n_neg > 0 else ""
            print(f"  {col:<25}{flag}{neg_flag}")

  
    def _check_categorical(self):
        cat_cols = self.df.select_dtypes(include='object').columns.tolist()
        if not cat_cols:
            return
        print("CATEGORICAL COLUMNS")
        print(f"  {'Cột':<25} {'Unique':>7}  {'Top value':<20}  Gợi ý encode")
        print("  " + "-" * 65)
        for col in cat_cols:
            n = self.df[col].nunique()
            top = str(self.df[col].value_counts().index[0])[:18]
            enc = "LabelEncoder" if n <= 2 else "OneHotEncoder" if n <= 10 else "High cardinality"
            print(f"  {col:<25} {n:>7}  {top:<20}  {enc}")


    def _check_target(self):
        col = self.target_col
        if col not in self.df.columns:
            return
        s = self.df[col]
        print(f"TARGET: '{col}'")
        vc = s.value_counts()
        vc_pct = s.value_counts(normalize=True) * 100
        ratio = vc_pct.max() / vc_pct.min()
        for cls, cnt, pct in zip(vc.index, vc.values, vc_pct.values):
            bar = '█' * int(pct / 2)
            print(f"  {str(cls):<18} {cnt:>7,}  ({pct:.1f}%)  {bar}")
        if ratio > 3:
            print(f"\n Imbalance ratio: {ratio:.1f}x → cân nhắc class_weight='balanced'")
        else:
            print(f"\n Classes tương đối cân bằng (ratio={ratio:.1f}x)")
