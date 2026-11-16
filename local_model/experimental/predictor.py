"""
预测模型模块 v4.0 - 集成学习预测股票涨跌
（网格搜索 + 三段切分校准 + 两步特征选择 + 加权集成 + Regime过滤 + 模型持久化）

v4.0 升级（8 项）：
  1. 超参数网格搜索（GridSearchCV + TimeSeriesSplit，rf/xgb/lgb）
  2. 三段切分校准（60/20/20 train/calib/test，校准改用 calib 集）
  3. 特征选择升级为两步法（RF importance top-30 + RFE 递归消除）
  4. 集成投票加权（按子模型在 test 集准确率作权重）
  5. 市场 Regime 过滤（熊市下调看多概率至70%，牛市/震荡贴标签）
  6. 模型持久化（joblib 缓存 + 加载，支持 max_age_days）
  7. 概率校准改用 calib 集（基于三段切分，避免 test 泄露）
  8. _auto_select_model / _walk_forward_validate 使用 default 模型快速比较

保留 v3.0/v2.0 全部功能：集成模型/特征标准化/滚动验证/概率校准/相似日检索
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.isotonic import IsotonicRegression
    ISOTONIC_AVAILABLE = True
except ImportError:
    ISOTONIC_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# v4.0: joblib 用于模型持久化
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    try:
        from sklearn.externals import joblib
        JOBLIB_AVAILABLE = True
    except ImportError:
        JOBLIB_AVAILABLE = False

# v2.0: 扩展后的特征列（归一化版本优先）
FEATURE_COLS = [
    # 均线偏离（归一化）
    "ma5_bias", "ma10_bias", "ma20_bias", "ma60_bias",
    "ma_bullish", "ma_cross", "ma5_slope", "ma20_slope",
    # RSI
    "rsi", "rsi_oversold", "rsi_overbought", "rsi_change",
    # MACD（归一化）
    "macd_dif_norm", "macd_hist_norm", "macd_cross",
    # KDJ
    "kdj_k", "kdj_d", "kdj_j", "kdj_cross",
    # 布林带
    "boll_pct", "boll_width",
    # 成交量
    "vol_ratio", "vol_change", "vol_surge", "amount_ratio",
    # 价格变化
    "price_change_1d", "price_change_5d", "price_change_10d",
    "high_low_ratio", "volatility_5d", "volatility_20d",
    # v2.0 新增：K线形态
    "upper_shadow", "lower_shadow", "body_ratio",
    "consecutive_up", "consecutive_down",
    # v2.0 新增：OBV
    "obv_norm", "obv_trend",
    # v2.0 新增：Williams %R
    "wr", "wr_oversold", "wr_overbought",
    # v2.0 新增：CCI
    "cci",
    # v2.0 新增：ATR
    "atr_pct",
    # v2.0 新增：换手率
    "turnover_proxy", "turnover_change",
    # v2.0 新增：交叉特征
    "vol_price_up", "vol_price_diverge", "rsi_price_diverge",
    # v2.0 新增：时间特征
    "day_of_week", "is_monday", "is_friday", "is_month_end", "is_month_start",
    # 原始涨跌幅
    "pct_change",
    # v2.0: 市场特征（如果有）
    "market_pct_change", "market_change_5d", "relative_strength",
    # v4.1 新增: chanlun_features.py 缠论适配特征（TD序列/背驰/二买二卖+中枢位置）
    "td9_buy", "td9_sell", "td13_buy", "td13_sell",
    "macd_bull_div", "macd_bull_div_strength", "macd_bear_div", "macd_bear_div_strength",
    "chanlun_second_buy", "chanlun_second_sell", "zhongshu_position_pct",
    # v4.1 新增: Vespa314 缠论引擎特征（买卖点/中枢/笔背驰），无网络时自动为 0
    "bsp_1buy", "bsp_2buy", "bsp_3buy", "bsp_1sell", "bsp_2sell", "bsp_3sell",
    "zhongshu_high", "zhongshu_low", "zhongshu_position_raw", "zhongshu_leave_dir",
    "vespa_macd_bull_div", "vespa_macd_bear_div",
]


class StockPredictor:
    """股票涨跌预测器 v4.0，支持集成学习、特征标准化、概率校准、网格搜索"""

    def __init__(self, forward_days: int = 5, model_type: str = "auto"):
        """
        参数:
            forward_days: 预测未来多少天的涨跌
            model_type: "auto" | "ensemble" | "rf" | "xgb" | "lgb"
                        auto:     自动选择最优单模型
                        ensemble: RF+XGB+LGB 集成投票
                        rf/xgb/lgb: 指定单模型
        """
        self.forward_days = forward_days
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = [c for c in FEATURE_COLS]
        self.accuracy = 0.0
        self.feature_importance = None
        self.trained_model_type = None
        self.model_comparison = {}
        self._scaler_fitted = False
        # v3.0: 概率校准相关
        self.calibrator = None
        self.is_calibrated = False
        self.calib_method = "none"
        self.calibration_brier_before = 0.0
        self.calibration_brier_after = 0.0
        self.calibration_warning = False

    # ------------------------------------------------------------------
    # v4.0: 默认模型（固定合理参数，用于快速比较 / walk-forward，不做网格搜索）
    # ------------------------------------------------------------------
    def _create_default_model(self, model_type: str):
        """创建默认参数模型（无网格搜索，未拟合）"""
        if model_type == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
        elif model_type == "xgb":
            if not XGBOOST_AVAILABLE:
                return self._create_default_model("rf")
            return xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                scale_pos_weight=1.0,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            )
        elif model_type == "lgb":
            if not LIGHTGBM_AVAILABLE:
                print("LightGBM 未安装，回退到随机森林")
                return self._create_default_model("rf")
            return lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
        elif model_type == "ensemble":
            return self._create_ensemble()
        else:
            return self._create_default_model("rf")

    # ------------------------------------------------------------------
    # v4.0: 模型创建（带可选网格搜索）
    # ------------------------------------------------------------------
    def _create_model(self, model_type: str, X_train=None, y_train=None,
                      X_val=None, y_val=None):
        """
        v4.0: 创建指定类型模型。
        - 提供 X_train/y_train：用 GridSearchCV + TimeSeriesSplit 做超参搜索，
          返回 best_estimator_（已拟合）。
        - 未提供：返回 _create_default_model（固定参数，未拟合）。
        """
        if X_train is None or y_train is None:
            return self._create_default_model(model_type)

        if model_type == "ensemble":
            est = self._create_ensemble(X_train=X_train, y_train=y_train,
                                        X_test=X_val, y_val=y_val)
            est.fit(X_train, y_train)
            return est

        tscv = TimeSeriesSplit(n_splits=3)

        if model_type == "rf":
            base = RandomForestClassifier(
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
            param_grid = {
                "n_estimators": [100, 200, 500],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            }
            gs = GridSearchCV(base, param_grid, cv=tscv,
                              scoring="accuracy", n_jobs=-1)
            gs.fit(X_train, y_train)
            print(f"    [GridSearchCV rf] best_params={gs.best_params_} "
                  f"best_acc={gs.best_score_:.4f}")
            return gs.best_estimator_

        if model_type == "xgb":
            if not XGBOOST_AVAILABLE:
                print("XGBoost 未安装，回退到随机森林")
                return self._create_model("rf", X_train=X_train, y_train=y_train,
                                          X_val=X_val, y_val=y_val)
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            }
            base = None
            try:
                base = xgb.XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    early_stopping_rounds=20,
                )
                if X_val is not None and y_val is not None:
                    base.set_params(eval_set=[(X_val, y_val)])
            except (TypeError, Exception):
                base = xgb.XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                )
            gs = GridSearchCV(base, param_grid, cv=tscv,
                              scoring="accuracy", n_jobs=-1)
            gs.fit(X_train, y_train)
            print(f"    [GridSearchCV xgb] best_params={gs.best_params_} "
                  f"best_acc={gs.best_score_:.4f}")
            return gs.best_estimator_

        if model_type == "lgb":
            if not LIGHTGBM_AVAILABLE:
                print("LightGBM 未安装，回退到随机森林")
                return self._create_model("rf", X_train=X_train, y_train=y_train,
                                          X_val=X_val, y_val=y_val)
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "num_leaves": [31, 63],
            }
            base = None
            try:
                base = lgb.LGBMClassifier(
                    verbose=-1,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                    early_stopping_rounds=20,
                )
                if X_val is not None and y_val is not None:
                    base.set_params(eval_set=[(X_val, y_val)])
            except (TypeError, Exception):
                base = lgb.LGBMClassifier(
                    verbose=-1,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                )
            gs = GridSearchCV(base, param_grid, cv=tscv,
                              scoring="accuracy", n_jobs=-1)
            gs.fit(X_train, y_train)
            print(f"    [GridSearchCV lgb] best_params={gs.best_params_} "
                  f"best_acc={gs.best_score_:.4f}")
            return gs.best_estimator_

        # 默认回退到 rf
        return self._create_model("rf", X_train=X_train, y_train=y_train,
                                  X_val=X_val, y_val=y_val)

    # ------------------------------------------------------------------
    # v4.0: 加权集成（按子模型准确率）
    # ------------------------------------------------------------------
    def _create_ensemble(self, X_train=None, y_train=None, X_test=None,
                         y_test=None, weights=None):
        """
        v4.0: 创建集成模型（软投票）。
        - 提供 weights：直接用作投票权重，跳过子模型重训（避免双重训练）。
        - 提供 X_train/y_train/X_test/y_test 但无 weights：训练各子模型，
          用其在 X_test（应为 calib 集）上的准确率作权重，跳过 0.0 准确率模型。
          【注意】调用方应传入 calib 集而非 test 集，避免 test 泄露。
        - 都不提供：等权集成（旧逻辑）。
        """
        names = ["rf"]
        if XGBOOST_AVAILABLE:
            names.append("xgb")
        if LIGHTGBM_AVAILABLE:
            names.append("lgb")

        if weights is None:
            if (X_train is not None and y_train is not None
                    and X_test is not None and y_test is not None):
                weights = []
                for name in names:
                    try:
                        m = self._create_default_model(name)
                        m.fit(X_train, y_train)
                        acc = accuracy_score(y_test, m.predict(X_test))
                    except Exception:
                        acc = 0.0
                    weights.append(float(acc))
                    print(f"    [ensemble weight] {name}: acc={acc:.4f}")
            else:
                weights = [1.0] * len(names)

        estimators = []
        final_weights = []
        for name, w in zip(names, weights):
            if float(w) > 0.0:
                estimators.append((name, self._create_default_model(name)))
                final_weights.append(float(w))

        if not estimators:
            estimators.append(("rf", self._create_default_model("rf")))
            final_weights = [1.0]

        print(f"    [ensemble weights] "
              f"{[e[0] for e in estimators]} = "
              f"{[round(w, 4) for w in final_weights]}")

        return VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=final_weights,
            n_jobs=-1,
        )

    def _get_available_features(self, df: pd.DataFrame) -> list:
        """获取数据中实际存在的特征列"""
        return [c for c in self.feature_cols if c in df.columns]

    # ------------------------------------------------------------------
    # v4.0: 三段切分 60/20/20
    # ------------------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """v4.0: 准备训练数据，三段切分 60/20/20（train/calib/test）"""
        available_features = self._get_available_features(df)
        self.feature_cols = available_features

        data = df.dropna(subset=available_features + ["label"]).copy()

        if len(data) < 100:
            raise ValueError(f"有效数据太少（{len(data)} 条），至少需要 100 条用于训练。")

        X = data[available_features]
        y = data["label"]

        # v4.0: 60/20/20 三段切分
        train_idx = int(len(data) * 0.6)
        calib_idx = int(len(data) * 0.8)

        X_train = X.iloc[:train_idx]
        X_calib = X.iloc[train_idx:calib_idx]
        X_test = X.iloc[calib_idx:]
        y_train = y.iloc[:train_idx]
        y_calib = y.iloc[train_idx:calib_idx]
        y_test = y.iloc[calib_idx:]

        # v2.0: 特征标准化（仅用训练集均值/方差）
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_calib_scaled = self.scaler.transform(X_calib)
        X_test_scaled = self.scaler.transform(X_test)
        self._scaler_fitted = True

        return X_train_scaled, X_calib_scaled, X_test_scaled, y_train, y_calib, y_test

    # ------------------------------------------------------------------
    # v4.0: 两步特征选择（RF importance + RFE）
    # ------------------------------------------------------------------
    def _select_and_rescale(self, X_train, X_calib, X_test, y_train,
                            top_k: int = 30):
        """
        v4.0: 两步特征选择
          Step 1: RF importance top-30（防过拟合）
          Step 2: RFE 递归消除至 min(15, max(5, len//20))
        选择后重新标准化，并更新 self.feature_cols / self.scaler，保证预测时一致。
        """
        n_features = len(self.feature_cols)

        # Step 1: RF importance top-k
        if n_features > top_k:
            selector = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42,
                class_weight="balanced", n_jobs=-1,
            )
            selector.fit(X_train, y_train)
            importances = selector.feature_importances_
            top_idx = np.sort(np.argsort(importances)[::-1][:top_k])
            dropped = n_features - top_k
            self.feature_cols = [self.feature_cols[i] for i in top_idx]
            X_train = X_train[:, top_idx]
            X_calib = X_calib[:, top_idx]
            X_test = X_test[:, top_idx]
            print(f"    特征选择 Step1: {n_features} → {top_k} 个"
                  f"（剔除 {dropped} 个低重要性特征）")
        else:
            print(f"    特征选择 Step1: 跳过（{n_features} ≤ {top_k}）")

        # Step 2: RFE 递归消除（固定保留 top-10，避免 n//20 魔法除数）
        n_current = len(self.feature_cols)
        target_n = min(10, n_current)
        if n_current > target_n:
            try:
                base_rfe = RandomForestClassifier(
                    n_estimators=50, max_depth=6,
                    random_state=42, n_jobs=-1,
                )
                rfe = RFE(base_rfe, n_features_to_select=target_n)
                rfe.fit(X_train, y_train)
                mask = rfe.support_
                eliminated = n_current - target_n
                self.feature_cols = [self.feature_cols[i]
                                     for i in range(len(self.feature_cols))
                                     if mask[i]]
                X_train = X_train[:, mask]
                X_calib = X_calib[:, mask]
                X_test = X_test[:, mask]
                print(f"    特征选择 Step2: {n_current} → {target_n} 个"
                      f"（RFE消除 {eliminated} 个）")
            except Exception as e:
                print(f"    特征选择 Step2: RFE 失败 ({type(e).__name__})，"
                      f"保留 Step1 结果")

        # 重新标准化（仅用选中特征）
        self.scaler = StandardScaler()
        X_train_sel = self.scaler.fit_transform(X_train)
        X_calib_sel = self.scaler.transform(X_calib)
        X_test_sel = self.scaler.transform(X_test)
        return X_train_sel, X_calib_sel, X_test_sel

    # ------------------------------------------------------------------
    # v4.0: 训练（集成网格搜索 + 三段切分 + 两步特征选择 + 加权集成 + 持久化）
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame, stock_code: str = None) -> dict:
        """训练模型并返回评估结果"""
        X_train, X_calib, X_test, y_train, y_calib, y_test = self.prepare_data(df)

        # v4.0: 两步特征选择
        X_train, X_calib, X_test = self._select_and_rescale(
            X_train, X_calib, X_test, y_train
        )

        # v4.1: 模型选择 / 网格搜索（用 CV/calib 选模型，避免 test 泄露）
        if self.model_type == "auto":
            best_model, best_type, comparison = self._auto_select_model(
                X_train, X_calib, y_train, y_calib
            )
            self.model_comparison = comparison
            if best_type == "ensemble":
                # 复用比较阶段各子模型的 CV 准确率作权重（无 test 泄露）
                names = ["rf"]
                if XGBOOST_AVAILABLE:
                    names.append("xgb")
                if LIGHTGBM_AVAILABLE:
                    names.append("lgb")
                w = [float(comparison.get(n, 0.0)) for n in names]
                self.model = self._create_ensemble(
                    X_train=X_train, y_train=y_train,
                    X_test=X_calib, y_test=y_calib, weights=w,
                )
                self.model.fit(X_train, y_train)
                self.trained_model_type = "ensemble"
            else:
                self.model = best_model
                self.trained_model_type = best_type
        elif self.model_type == "ensemble":
            # 用 calib 集算权重（无 test 泄露）
            self.model = self._create_ensemble(
                X_train=X_train, y_train=y_train,
                X_test=X_calib, y_test=y_calib,
            )
            self.model.fit(X_train, y_train)
            self.trained_model_type = "ensemble"
        else:
            # v4.0: 网格搜索（用 calib 集作 early_stopping 的 eval_set）
            self.model = self._create_model(
                self.model_type,
                X_train=X_train, y_train=y_train,
                X_val=X_calib, y_val=y_calib,
            )
            self.trained_model_type = self.model_type

        # 评估（在 test 集上）
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        self.accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        self.feature_importance = self._get_feature_importance()

        # v4.0: 概率校准（改用 calib 集，避免 test 泄露）
        y_prob_raw_calib = self.model.predict_proba(X_calib)[:, 1]
        self._calibrate_proba(X_calib, y_calib, y_prob_raw_calib)

        # v2.1: Purged 滚动验证准确率（消除标签泄露）
        walk_forward_acc = self._walk_forward_validate(df)

        result = {
            "accuracy": self.accuracy,
            "report": report,
            "confusion_matrix": cm,
            "train_size": len(X_train),
            "calibration_size": len(X_calib),
            "test_size": len(X_test),
            "test_date_range": f"{df['date'].iloc[int(len(df) * 0.8)]} ~ {df['date'].iloc[-1]}",
            "model_type": self.trained_model_type,
            "walk_forward_accuracy": walk_forward_acc,
            "feature_count": len(self.feature_cols),
            "calib_method": self.calib_method,
            "is_calibrated": self.is_calibrated,
            "calibration_brier_before": self.calibration_brier_before,
            "calibration_brier_after": self.calibration_brier_after,
            "calibration_warning": self.calibration_warning,
        }

        if self.model_comparison:
            result["model_comparison"] = self.model_comparison

        # v4.0: 模型持久化
        if stock_code is not None:
            self.save_model(stock_code)

        return result

    def _auto_select_model(self, X_train, X_calib, y_train, y_calib):
        """v4.1: 自动选择最优模型（用 TimeSeriesSplit CV 在训练集上选，避免 test 泄露）

        用 X_train 的时序交叉验证准确率选模型，comparison 存储 CV 准确率。
        不再使用 X_test/y_test 参与模型选择。
        """
        candidates = ["rf"]
        if XGBOOST_AVAILABLE:
            candidates.append("xgb")
        if LIGHTGBM_AVAILABLE:
            candidates.append("lgb")
        candidates.append("ensemble")

        best_model = None
        best_type = None
        best_acc = 0
        comparison = {}

        tscv = TimeSeriesSplit(n_splits=min(3, max(2, len(X_train) // 100)))

        for mtype in candidates:
            try:
                if mtype == "ensemble":
                    # ensemble 用 calib 集准确率作为权重基准（无泄露）
                    ens = self._create_ensemble(
                        X_train=X_train, y_train=y_train,
                        X_test=X_calib, y_test=y_calib,
                    )
                    ens.fit(X_train, y_train)
                    acc = accuracy_score(y_calib, ens.predict(X_calib))
                    model = ens
                else:
                    # 单模型用 TimeSeriesSplit CV 准确率选
                    model = self._create_default_model(mtype)
                    cv_accs = []
                    for tr_idx, val_idx in tscv.split(X_train):
                        m = self._create_default_model(mtype)
                        m.fit(X_train[tr_idx], y_train[tr_idx])
                        cv_accs.append(accuracy_score(y_train[val_idx], m.predict(X_train[val_idx])))
                    acc = float(np.mean(cv_accs)) if cv_accs else 0.0
                    model.fit(X_train, y_train)
                comparison[mtype] = acc
                print(f"    {mtype.upper():>10s} 模型准确率(CV): {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_type = mtype
            except Exception as e:
                print(f"    {mtype.upper():>10s} 模型失败: {type(e).__name__}")
                comparison[mtype] = 0

        print(f"    → 选择 {best_type.upper()} 模型（CV准确率 {best_acc:.4f}）")
        return best_model, best_type, comparison

    def _walk_forward_validate(self, df: pd.DataFrame, n_splits: int = 5) -> float:
        """v2.1: Purged 滚动验证（walk-forward），消除标签泄露，更贴近真实交易"""
        available_features = [c for c in self.feature_cols if c in df.columns]
        data = df.dropna(subset=available_features + ["label"]).copy()

        if len(data) < 200:
            return 0.0

        X = data[available_features].values
        y = data["label"].values

        # v2.1: gap = forward_days，剔除训练集尾部标签窗口与测试集重叠的样本
        gap = min(self.forward_days, 10)
        try:
            tscv = TimeSeriesSplit(n_splits=min(n_splits, 5), gap=gap)
        except TypeError:
            tscv = TimeSeriesSplit(n_splits=min(n_splits, 5))

        accuracies = []
        for train_idx, test_idx in tscv.split(X):
            if len(test_idx) < 20 or len(train_idx) < 100:
                continue
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # 标准化
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # v4.0: 用 default 模型（无网格搜索），与最终模型同类型
            model = self._create_default_model(self.trained_model_type or "rf")
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)
            accuracies.append(accuracy_score(y_te, y_pred))

        return np.mean(accuracies) if accuracies else 0.0

    def _get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性（兼容单模型与集成模型）"""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "estimators_"):
            # 集成模型：取各子模型平均
            importances = np.zeros(len(self.feature_cols))
            count = 0
            for item in self.model.estimators_:
                est = item[1] if isinstance(item, tuple) else item
                if hasattr(est, "feature_importances_"):
                    importances += est.feature_importances_
                    count += 1
            if count > 0:
                importances /= count
            else:
                return pd.DataFrame({"feature": self.feature_cols, "importance": 0})
        else:
            return pd.DataFrame({"feature": self.feature_cols, "importance": 0})

        return pd.DataFrame(
            {"feature": self.feature_cols, "importance": importances}
        ).sort_values("importance", ascending=False)

    def _calibrate_proba(self, X_calib, y_calib, y_prob_raw_calib=None):
        """
        v4.0: 概率校准（Platt scaling / Isotonic Regression）
        - 基于 calib 集（与 test 集独立，避免泄露）
        - calib 样本>=200: 优先 IsotonicRegression
        - calib 样本<200:  用 Platt (LogisticRegression 拟合1维概率)
        - 两者失败则标记 calibration_warning=True
        """
        try:
            n = len(y_calib)
            if n < 20:
                self.calibration_warning = True
                print("    ⚠️ 校准样本不足，跳过概率校准")
                return

            if y_prob_raw_calib is None:
                y_prob_raw_calib = self.model.predict_proba(X_calib)[:, 1]

            y_prob_raw = np.asarray(y_prob_raw_calib, dtype=float).flatten()
            y_true = np.asarray(y_calib, dtype=int).flatten()

            brier_before = float(np.mean((y_prob_raw - y_true) ** 2))
            self.calibration_brier_before = brier_before

            calibrator = None
            calib_method_name = None

            if ISOTONIC_AVAILABLE and n >= 200:
                try:
                    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                    ir.fit(y_prob_raw, y_true)
                    calibrator = ir
                    calib_method_name = "isotonic"
                except Exception:
                    calibrator = None

            if calibrator is None:
                try:
                    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
                    lr.fit(y_prob_raw.reshape(-1, 1), y_true)
                    calibrator = lr
                    calib_method_name = "platt"
                except Exception:
                    calibrator = None

            if calibrator is None:
                self.calibration_warning = True
                self.calib_method = "none"
                self.is_calibrated = False
                self.calibration_brier_after = brier_before
                print("    ⚠️ 概率校准失败（isotonic & platt 均不可用）")
                return

            if calib_method_name == "isotonic":
                prob_cal = calibrator.transform(y_prob_raw)
            else:
                prob_cal = calibrator.predict_proba(y_prob_raw.reshape(-1, 1))[:, 1]

            prob_cal = np.clip(prob_cal, 0.0, 1.0)
            brier_after = float(np.mean((prob_cal - y_true) ** 2))

            self.calibrator = calibrator
            self.is_calibrated = True
            self.calib_method = calib_method_name
            self.calibration_brier_after = brier_after
            self.calibration_warning = False
            print(f"    ✅ 概率校准完成 [{calib_method_name}] "
                  f"Brier: {brier_before:.5f} → {brier_after:.5f}")

        except Exception as e:
            self.calibration_warning = True
            print(f"    ⚠️ 概率校准异常: {type(e).__name__}")

    def _apply_calibration(self, up_prob: float) -> float:
        """应用校准器到单个概率值"""
        if not self.is_calibrated or self.calibrator is None:
            return float(up_prob)
        try:
            if self.calib_method == "isotonic":
                p = self.calibrator.transform(np.array([float(up_prob)]))[0]
            else:
                p = self.calibrator.predict_proba(np.array([[float(up_prob)]]))[0, 1]
            return float(np.clip(p, 0.0, 1.0))
        except Exception:
            return float(up_prob)

    def find_similar_days(self, df_labeled, feature_cols=None, scaler=None,
                          target_idx=-1, top_k=5, train_end_idx=None):
        """
        v3.0: 相似历史日检索
        - 用 scaler 变换后的特征计算欧氏距离
        - train_end_idx 默认取 60% 长度（v4.0：与三段切分的训练窗口一致，防泄露）
        - 返回 Top-K 近邻及真实 next_1d/3d 涨跌幅（从实际 close 列计算）
        """
        if feature_cols is None:
            feature_cols = [c for c in self.feature_cols if c in df_labeled.columns]
        if scaler is None:
            scaler = self.scaler

        df = df_labeled.reset_index(drop=True)
        available = [c for c in feature_cols if c in df.columns]
        df_clean = df.dropna(subset=available + ["close"]).reset_index(drop=True)

        if len(df_clean) < top_k + 5:
            return []

        if train_end_idx is None:
            # v4.0: 0.8 → 0.6，与三段切分训练窗口一致，避免泄露
            train_end_idx = int(len(df_clean) * 0.6)
        train_end_idx = min(max(int(train_end_idx), top_k + 2), len(df_clean) - 1)

        if target_idx < 0:
            target_abs_idx = len(df_clean) - 1
        else:
            target_abs_idx = min(int(target_idx), len(df_clean) - 1)

        try:
            X_all = scaler.transform(df_clean[available].values)
        except Exception:
            scaler_local = StandardScaler()
            X_all = scaler_local.fit_transform(df_clean[available].values)

        x_target = X_all[target_abs_idx]
        search_end = train_end_idx
        X_train_pool = X_all[:search_end]

        if len(X_train_pool) == 0:
            return []

        diff = X_train_pool - x_target
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        top_idx = np.argsort(dists)[:top_k]

        results = []
        close_arr = df_clean["close"].values
        date_arr = df_clean["date"].values

        for rank, idx in enumerate(top_idx):
            idx = int(idx)
            sim_score = float(1.0 / (1.0 + dists[idx]))

            n1 = idx + 1
            n3 = idx + 3
            ret1 = float("nan")
            ret3 = float("nan")
            if n1 < len(close_arr) and close_arr[idx] > 0:
                ret1 = (close_arr[n1] / close_arr[idx] - 1) * 100.0
            if n3 < len(close_arr) and close_arr[idx] > 0:
                ret3 = (close_arr[n3] / close_arr[idx] - 1) * 100.0

            dt = date_arr[idx]
            if hasattr(dt, "strftime"):
                date_str = dt.strftime("%Y-%m-%d")
            else:
                date_str = str(dt)[:10]

            results.append(dict(
                date=date_str,
                sim_score=round(sim_score, 4),
                next_1d_ret_pct=round(ret1, 2) if not np.isnan(ret1) else None,
                next_3d_ret_pct=round(ret3, 2) if not np.isnan(ret3) else None,
            ))
        return results

    # ------------------------------------------------------------------
    # v4.0: 预测（含 Regime 过滤）
    # ------------------------------------------------------------------
    def predict_latest(self, df: pd.DataFrame) -> dict:
        """对最新一条数据进行预测（v4.0: 含校准概率 + Regime过滤）"""
        latest = df.iloc[-1:]
        available_features = [c for c in self.feature_cols if c in df.columns]
        self.feature_cols = available_features

        if latest[available_features].isna().any().any():
            for i in range(len(df) - 1, -1, -1):
                row = df.iloc[[i]]
                if not row[available_features].isna().any().any():
                    latest = row
                    break

        X = latest[available_features]
        if self._scaler_fitted:
            X = self.scaler.transform(X)

        prob = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]

        up_prob_raw = float(prob[1])
        calibrated_up_prob = self._apply_calibration(up_prob_raw)

        # v4.0: 市场 Regime 过滤
        calibrated_up_prob_raw = float(calibrated_up_prob)
        regime = "unknown"
        regime_label = "未知"
        regime_adjustment = None
        if "regime" in df.columns:
            try:
                regime_val = str(df.iloc[-1]["regime"]).lower().strip()
            except Exception:
                regime_val = ""
            if regime_val == "bear":
                regime = "bear"
                regime_label = "🔴熊市"
                # v4.1: 移除硬编码 *0.7 系数（无统计依据）。
                # 正确做法：将 regime 作为特征输入模型训练，让模型自动学习不同 regime 下的概率偏移。
                # 当前保持概率不变，仅在输出中标注 regime 状态供参考。
                calibrated_up_prob = float(calibrated_up_prob_raw)
                regime_adjustment = "⚠️当前为熊市regime，请注意风险（概率未硬编码调整，建议将regime纳入特征训练）"
            elif regime_val == "bull":
                regime = "bull"
                regime_label = "🟢牛市"
                regime_adjustment = "当前为牛市regime，看多概率保持不变"
            elif regime_val == "sideways":
                regime = "sideways"
                regime_label = "🟡震荡市"
                regime_adjustment = "当前为震荡市regime，看多概率保持不变"

        # v2.0: 置信度评估（用校准+Regime调整后的概率）
        confidence = abs(calibrated_up_prob - 0.5) * 2
        if confidence > 0.4:
            confidence_level = "高"
        elif confidence > 0.2:
            confidence_level = "中"
        else:
            confidence_level = "低"

        result_dict = {
            "prediction": "涨" if pred == 1 else "跌",
            "up_probability": up_prob_raw,
            "down_probability": float(prob[0]),
            "target_date": latest["date"].iloc[0],
            "close_price": latest["close"].iloc[0],
            "confidence": confidence,
            "confidence_level": confidence_level,
            "calibrated_up_prob": calibrated_up_prob,
            "calibrated_up_prob_raw": calibrated_up_prob_raw,
            "is_calibrated": bool(self.is_calibrated),
            "calibration_warning": bool(self.calibration_warning),
            "calib_method": self.calib_method,
            "regime": regime,
            "regime_label": regime_label,
            "regime_adjustment": regime_adjustment,
        }
        # v4.1: chanlun features latest row for stock-advisor display
        extra = {}
        row = df.iloc[-1]
        for c in ("td9_buy","td9_sell","td13_buy","td13_sell",
                  "macd_bull_div","macd_bull_div_strength","macd_bear_div","macd_bear_div_strength",
                  "chanlun_second_buy","chanlun_second_sell","zhongshu_position_pct"):
            if c in df.columns:
                try:
                    val = float(row[c]) if not pd.isna(row[c]) else 0.0
                except (ValueError, TypeError):
                    val = 0.0
                extra[c] = val
        result_dict["extra_chanlun_signals"] = extra
        return result_dict

    def predict(self, features_dict: dict) -> dict:
        """直接用特征字典进行预测"""
        row = pd.DataFrame([features_dict])
        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0

        X = row[self.feature_cols]
        if self._scaler_fitted:
            X = self.scaler.transform(X)

        prob = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]

        return {
            "prediction": "涨" if pred == 1 else "跌",
            "up_probability": prob[1],
            "down_probability": prob[0],
        }

    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """获取最重要的 N 个特征"""
        return self.feature_importance.head(n)

    # ------------------------------------------------------------------
    # v4.0: 模型持久化
    # ------------------------------------------------------------------
    def save_model(self, stock_code: str):
        """v4.0: 模型持久化（joblib 缓存：model+scaler+feature_cols+calibrator）"""
        if not JOBLIB_AVAILABLE:
            print("⚠️ joblib 不可用，跳过模型缓存")
            return
        try:
            project_root = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(project_root, "04_输出结果", "03模型缓存")
            os.makedirs(cache_dir, exist_ok=True)
            date_str = pd.Timestamp.now().strftime("%Y%m%d")
            fname = f"model_{stock_code}_{self.forward_days}d_{date_str}.joblib"
            path = os.path.join(cache_dir, fname)
            bundle = {
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "calibrator": self.calibrator,
                "is_calibrated": self.is_calibrated,
                "calib_method": self.calib_method,
                "trained_model_type": self.trained_model_type,
                "forward_days": self.forward_days,
                "accuracy": self.accuracy,
            }
            joblib.dump(bundle, path)
            print(f"✅ 模型已缓存: {path}")
        except Exception as e:
            print(f"⚠️ 模型缓存失败: {type(e).__name__}: {e}")

    def load_model(self, stock_code: str, max_age_days: int = 3) -> bool:
        """
        v4.0: 加载最近的模型缓存。
        - 在缓存目录查找 model_{stock_code}_{forward_days}d_*.joblib
        - 取 max_age_days 天内最新的一份，恢复 model/scaler/feature_cols/calibrator
        - 找到并成功加载返回 True，否则 False
        """
        if not JOBLIB_AVAILABLE:
            return False
        try:
            project_root = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(project_root, "04_输出结果", "03模型缓存")
            if not os.path.isdir(cache_dir):
                return False
            prefix = f"model_{stock_code}_{self.forward_days}d_"
            candidates = [f for f in os.listdir(cache_dir)
                          if f.startswith(prefix) and f.endswith(".joblib")]
            if not candidates:
                return False

            now = pd.Timestamp.now()
            best_path = None
            best_mtime = -1.0
            for f in candidates:
                full = os.path.join(cache_dir, f)
                mtime = os.path.getmtime(full)
                age_days = (now - pd.Timestamp(mtime, unit="s")).days
                if age_days <= max_age_days and mtime > best_mtime:
                    best_mtime = mtime
                    best_path = full

            if best_path is None:
                return False

            bundle = joblib.load(best_path)
            self.model = bundle.get("model")
            self.scaler = bundle.get("scaler", self.scaler)
            # v4.1: cache invalidation guard — feature set mismatch means cache stale, re-train
            # 1) 维度校验：scaler 训练时的特征数 != 当前 FEATURE_COLS 长度
            cached_feat_cols = bundle.get("feature_cols", [])
            if hasattr(self.scaler, "n_features_in_"):
                if self.scaler.n_features_in_ != len(self.feature_cols):
                    print(f"[WARN] 模型缓存特征维度不匹配（缓存{self.scaler.n_features_in_} vs 当前{len(self.feature_cols)}），重新训练")
                    return False
            # 2) 列名校验：缓存特征必须全部属于当前 FEATURE_COLS（防止旧列残留或列被替换）
            current_set = set(FEATURE_COLS)
            stale = [c for c in cached_feat_cols if c not in current_set]
            if stale:
                print(f"[WARN] 模型缓存含过期特征列 {stale[:3]}{'...' if len(stale)>3 else ''}，重新训练")
                return False
            self.feature_cols = cached_feat_cols if cached_feat_cols else self.feature_cols
            self.calibrator = bundle.get("calibrator")
            self.is_calibrated = bool(bundle.get("is_calibrated", False))
            self.calib_method = bundle.get("calib_method", "none")
            self.trained_model_type = bundle.get("trained_model_type")
            self.accuracy = float(bundle.get("accuracy", 0.0))
            self._scaler_fitted = self.scaler is not None
            print(f"✅ 已加载模型缓存: {best_path}")
            return True
        except Exception as e:
            print(f"⚠️ 模型加载失败: {type(e).__name__}: {e}")
            return False
