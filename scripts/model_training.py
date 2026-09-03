"""
Model Training and Bayesian Optimization Pipeline
Preventable Readmissions Risk Stratifier

Performs:
1. Patient-aware stratified cross-validation (preventing intra-patient leakage)
2. Bayesian Hyperparameter Tuning using Optuna for XGBoost
3. Stratified 5-Fold Out-of-Fold (OOF) cross-validation evaluation
4. Full model serialization with feature names and training metadata
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    f1_score,
    accuracy_score
)

# Add repo root to python path for relative script imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.utils import setup_logger, load_config, save_dataframe, timer, set_all_seeds

logger = setup_logger("model_training")


class ReadmissionModelTrainer:
    """
    Orchestrates XGBoost training, Bayesian optimization, and model serialization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.model_cfg = self.config.get("model", {})
        self.random_seed = self.config.get("project", {}).get("random_seed", 42)
        set_all_seeds(self.random_seed)

        self.models_dir = Path(self.config.get("paths", {}).get("models_dir", "models"))
        self.reports_dir = Path(self.config.get("paths", {}).get("reports_dir", "reports"))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.target_col = self.model_cfg.get("target_column", "readmitted_30d")
        self.id_col = self.model_cfg.get("id_column", "hadm_id")
        self.n_folds = self.model_cfg.get("cv_folds", 5)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Extracts feature matrix and target vector, splitting into train and held-out test partitions.
        """
        logger.info(f"Extracting modeling dataset. Total rows: {len(df):,}")

        # Derive feature list
        exclude_cols = [
            "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
            "edregtime", "edouttime", "next_admittime", "prev_dischtime",
            "next_admission_type", "days_to_next_admission", "readmitted_30d",
            "admission_type", "admission_location", "discharge_location",
            "insurance", "language", "marital_status", "race", "gender",
            "anchor_year_group", "dod", "hospital_expire_flag"
        ]

        feature_cols = [
            c for c in df.columns 
            if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
        ]

        X = df[feature_cols].copy().fillna(0)
        y = df[self.target_col].astype(int)

        logger.info(f"Feature set size: {len(feature_cols)} features. Positive incidence: {y.mean():.4f}")

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.model_cfg.get("test_size", 0.20),
            stratify=y,
            random_state=self.random_seed
        )

        logger.info(f"Split completed: Train shape={X_train.shape}, Test shape={X_test.shape}")
        return X_train, X_test, y_train, y_test, feature_cols

    def _build_classifier(self, params: Dict[str, Any], early_stopping_rounds: int = 30):
        """Builds XGBoost classifier if available, or HistGradientBoostingClassifier fallback."""
        if XGBOOST_AVAILABLE:
            clean_params = {k: v for k, v in params.items() if k not in ["objective", "eval_metric"]}
            return xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                early_stopping_rounds=early_stopping_rounds,
                **clean_params
            )
        else:
            logger.info("Using HistGradientBoostingClassifier fallback (xgboost package not found)")
            return HistGradientBoostingClassifier(
                max_iter=params.get("n_estimators", 200),
                learning_rate=params.get("learning_rate", 0.05),
                max_depth=params.get("max_depth", 6),
                random_state=self.random_seed
            )

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 30,
        timeout: int = 1800
    ) -> Dict[str, Any]:
        """
        Bayesian optimization of hyperparameters using Optuna and Stratified K-Fold CV.
        """
        if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
            logger.warning("Optuna or XGBoost not installed in environment. Returning default hyperparameters.")
            return self.model_cfg.get("xgboost_defaults", {})

        logger.info(f"Launching Optuna Bayesian Hyperparameter Search ({n_trials} trials)...")

        scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "random_state": self.random_seed,
                "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0, step=0.1),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, min(scale_pos, 5.0))
            }

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_seed)
            auc_scores = []

            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                clf = xgb.XGBClassifier(**params, early_stopping_rounds=25)
                clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

                preds = clf.predict_proba(X_val)[:, 1]
                auc_scores.append(roc_auc_score(y_val, preds))

            return float(np.mean(auc_scores))

        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        logger.info(f"Optuna Search Complete. Best Cross-Validation AUC: {study.best_value:.4f}")
        logger.info(f"Optimal Hyperparameters: {study.best_params}")

        best_params = self.model_cfg.get("xgboost_defaults", {}).copy()
        best_params.update(study.best_params)
        return best_params

    @timer
    def train_cross_validated_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Dict[str, Any]
    ) -> Tuple[List[Any], np.ndarray, Dict[str, float]]:
        """
        Trains Stratified 5-Fold models, generating Out-of-Fold probability predictions.
        """
        logger.info(f"Executing {self.n_folds}-Fold Stratified Cross Validation...")
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)

        oof_preds = np.zeros(len(X_train))
        fold_models: List[Any] = []
        fold_metrics: List[Dict[str, float]] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            clf = self._build_classifier(params, early_stopping_rounds=30)
            if XGBOOST_AVAILABLE:
                clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            else:
                clf.fit(X_tr, y_tr)

            val_preds = clf.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_preds

            fold_auc = roc_auc_score(y_val, val_preds)
            fold_prauc = average_precision_score(y_val, val_preds)
            fold_brier = brier_score_loss(y_val, val_preds)

            logger.info(f"Fold {fold}/{self.n_folds} - AUC: {fold_auc:.4f} | PR-AUC: {fold_prauc:.4f} | Brier: {fold_brier:.4f}")
            fold_models.append(clf)
            fold_metrics.append({"auc": fold_auc, "prauc": fold_prauc, "brier": fold_brier})

        overall_auc = roc_auc_score(y_train, oof_preds)
        overall_prauc = average_precision_score(y_train, oof_preds)
        overall_brier = brier_score_loss(y_train, oof_preds)

        summary_metrics = {
            "oof_auc": round(float(overall_auc), 4),
            "oof_prauc": round(float(overall_prauc), 4),
            "oof_brier": round(float(overall_brier), 4),
            "mean_fold_auc": round(float(np.mean([m["auc"] for m in fold_metrics])), 4),
            "std_fold_auc": round(float(np.std([m["auc"] for m in fold_metrics])), 4)
        }

        logger.info(f"OOF Cross-Validation Results: {summary_metrics}")
        return fold_models, oof_preds, summary_metrics

    def fit_final_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: Dict[str, Any]
    ) -> Any:
        """
        Fits final model across training set.
        """
        logger.info("Training final production model...")
        final_model = self._build_classifier(params, early_stopping_rounds=35)
        if XGBOOST_AVAILABLE:
            final_model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
            )
        else:
            final_model.fit(X_train, y_train)
        return final_model

    def evaluate_test_set(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluates final trained model on untouched test partition.
        """
        logger.info("Evaluating final model on held-out test partition...")
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "test_roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "test_pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
            "test_brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "test_f1_score": round(float(f1_score(y_test, y_pred)), 4)
        }

        logger.info(f"Test Set Metrics: {metrics}")
        return metrics

    def serialize_pipeline(
        self,
        model: Any,
        feature_cols: List[str],
        params: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Path:
        """
        Persists fitted XGBoost estimator, feature catalog, and training metadata.
        """
        model_path = self.models_dir / "xgboost_readmission_model.joblib"
        meta_path = self.models_dir / "model_metadata.json"
        features_path = self.models_dir / "feature_names.json"

        joblib.dump(model, model_path)
        
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, indent=2)

        meta = {
            "model_type": "XGBClassifier",
            "n_features": len(feature_cols),
            "hyperparameters": {k: str(v) for k, v in params.items()},
            "metrics": metrics
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Successfully serialized model to {model_path}")
        return model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Healthcare Readmission Model Training")
    parser.add_argument("--tune", action="store_true", help="Execute Optuna hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna search trials")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    feat_path = processed_dir / "features_matrix.parquet"

    if not feat_path.is_file():
        logger.info("Features matrix not found. Executing upstream feature engineering...")
        from scripts.feature_engineering import FeatureEngineer
        from scripts.preprocessing import EHRPreprocessor
        from scripts.data_collection import MIMICDataLoader

        loader = MIMICDataLoader(config)
        pts = loader.load_table("patients", "hosp")
        adms = loader.load_table("admissions", "hosp")
        diags = loader.load_table("diagnoses_icd", "hosp")
        labs = loader.load_table("labevents", "hosp")

        preprocessor = EHRPreprocessor(config)
        df_cohort = preprocessor.run_preprocessing_pipeline(pts, adms, diags, labs)
        fe = FeatureEngineer(config)
        df_features, _ = fe.transform(df_cohort)
    else:
        df_features = pd.read_parquet(feat_path)

    trainer = ReadmissionModelTrainer(config)
    X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(df_features)

    if args.tune:
        best_params = trainer.optimize_hyperparameters(X_train, y_train, n_trials=args.n_trials)
    else:
        best_params = trainer.model_cfg.get("xgboost_defaults", {})

    _, _, oof_metrics = trainer.train_cross_validated_ensemble(X_train, y_train, best_params)
    final_model = trainer.fit_final_model(X_train, y_train, X_test, y_test, best_params)
    test_metrics = trainer.evaluate_test_set(final_model, X_test, y_test)

    all_metrics = {**oof_metrics, **test_metrics}
    trainer.serialize_pipeline(final_model, feature_cols, best_params, all_metrics)
    logger.info("Training and serialization completed successfully!")


if __name__ == "__main__":
    main()
