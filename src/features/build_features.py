"""
Feature Engineering and Building Module
========================================

This module handles comprehensive feature engineering including:
- Advanced feature selection and correlation analysis
- Data preprocessing pipelines
- Train-test splitting with proper data leakage prevention
- Custom transformers for missing values and outliers

Classes:
--------
SmartImputer: Custom transformer for intelligent missing value handling
Winsorizer: Custom transformer for outlier treatment using winsorization
AdvancedFeatureSelector: Advanced feature selection based on correlation and importance
FlexibleColumnTransformer: Adaptive transformer for numerical and categorical features
FeatureBuilder: Main class orchestrating all feature engineering operations

Functions:
----------
create_preprocessing_pipeline: Factory function for preprocessing pipeline
"""

import logging
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from category_encoders import CatBoostEncoder
from scipy.stats import mstats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class SmartImputer(BaseEstimator, TransformerMixin):
    """
    Intelligent imputation strategy that handles missing values based on column types
    and missing value patterns. Drops columns with excessive missing values.
    """

    def __init__(self, drop_threshold: float = 0.8):
        """
        Initialize SmartImputer.

        Parameters:
        -----------
        drop_threshold : float, default=0.8
            Threshold for dropping columns with excessive missing values
        """
        self.drop_threshold = drop_threshold
        self.columns_to_drop = []
        self.numerical_imputer = SimpleImputer(strategy="median")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the imputer on training data."""
        # Identify columns to drop (>threshold% missing)
        missing_ratios = X.isnull().sum() / len(X)
        self.columns_to_drop = missing_ratios[missing_ratios > self.drop_threshold].index.tolist()

        if self.columns_to_drop:
            logger.info(
                f"   Dropping {len(self.columns_to_drop)} columns with >{self.drop_threshold*100}% missing values"
            )

        # Fit imputers on remaining columns
        X_clean = X.drop(self.columns_to_drop, axis=1)

        num_cols = X_clean.select_dtypes(include=[np.number]).columns
        cat_cols = X_clean.select_dtypes(include=["object"]).columns

        if len(num_cols) > 0:
            self.numerical_imputer.fit(X_clean[num_cols])
        if len(cat_cols) > 0:
            self.categorical_imputer.fit(X_clean[cat_cols])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values."""
        X_clean = X.drop(self.columns_to_drop, axis=1, errors="ignore").copy()

        num_cols = X_clean.select_dtypes(include=[np.number]).columns
        cat_cols = X_clean.select_dtypes(include=["object"]).columns

        if len(num_cols) > 0:
            X_clean[num_cols] = self.numerical_imputer.transform(X_clean[num_cols])
        if len(cat_cols) > 0:
            X_clean[cat_cols] = self.categorical_imputer.transform(X_clean[cat_cols])

        return X_clean


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Winsorization transformer for handling outliers by clipping extreme values
    to specified quantiles.
    """

    def __init__(self, limits: Tuple[float, float] = (0.05, 0.05)):
        """
        Initialize Winsorizer.

        Parameters:
        -----------
        limits : tuple, default=(0.05, 0.05)
            Lower and upper quantiles for clipping
        """
        self.limits = limits
        self.bounds = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Fit winsorizer by calculating quantile bounds."""
        for col in X.select_dtypes(include=[np.number]).columns:
            self.bounds[col] = mstats.mquantiles(
                X[col].dropna(), prob=[self.limits[0], 1 - self.limits[1]]
            )

        if self.bounds:
            logger.info(f"   Fitted winsorizer for {len(self.bounds)} numerical columns")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by winsorizing outliers."""
        X_winsorized = X.copy()
        for col, (lower, upper) in self.bounds.items():
            if col in X_winsorized.columns:
                X_winsorized[col] = np.clip(X_winsorized[col], lower, upper)
        return X_winsorized


class AdvancedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Advanced feature selection using correlation analysis and feature importance.
    Removes highly correlated features and low-importance features.
    """

    def __init__(self, correlation_threshold: float = 0.95, importance_threshold: float = 0.01):
        """
        Initialize AdvancedFeatureSelector.

        Parameters:
        -----------
        correlation_threshold : float, default=0.95
            Threshold for removing highly correlated features
        importance_threshold : float, default=0.01
            Threshold for removing low-importance features
        """
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        self.selected_features = None
        self.dropped_features = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit feature selector using correlation and importance analysis."""
        X_temp = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.dropped_features = []

        logger.info(f"   Starting feature selection with {len(X_temp.columns)} features")

        # Step 1: Drop highly correlated features
        numeric_cols = X_temp.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            try:
                # Fill NaN values temporarily for correlation calculation
                X_numeric = X_temp[numeric_cols].fillna(X_temp[numeric_cols].median())
                corr_matrix = X_numeric.corr().abs()

                # Get upper triangle of correlation matrix
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )

                # Find features to drop based on correlation
                high_corr_pairs = []
                for column in upper_triangle.columns:
                    correlated_features = upper_triangle.index[
                        upper_triangle[column] > self.correlation_threshold
                    ].tolist()
                    if correlated_features:
                        for corr_feature in correlated_features:
                            high_corr_pairs.append((column, corr_feature))

                # For each correlated pair, keep the one with higher target correlation
                if y is not None and len(high_corr_pairs) > 0:
                    features_to_drop = set()
                    for feat1, feat2 in high_corr_pairs:
                        try:
                            y_clean = pd.Series(y).fillna(0)
                            corr1 = abs(np.corrcoef(X_numeric[feat1].fillna(0), y_clean)[0, 1])
                            corr2 = abs(np.corrcoef(X_numeric[feat2].fillna(0), y_clean)[0, 1])

                            # Drop the feature with lower target correlation
                            if corr1 > corr2:
                                features_to_drop.add(feat2)
                            else:
                                features_to_drop.add(feat1)
                        except Exception as e:
                            print(e)
                            features_to_drop.add(feat2)

                    self.dropped_features.extend(list(features_to_drop))
                    logger.info(f"   Dropped {len(features_to_drop)} highly correlated features")

            except Exception as e:
                logger.warning(f"   Correlation analysis failed: {str(e)[:50]}...")

        # Step 2: Feature importance based selection
        X_after_corr = X_temp.drop(columns=list(set(self.dropped_features)), errors="ignore")

        if y is not None and len(X_after_corr.columns) > 5:  # Only if we have enough features
            try:
                # Prepare data for XGBoost
                X_for_importance = X_after_corr.copy()

                # Handle categorical columns
                categorical_cols = X_for_importance.select_dtypes(include=["object"]).columns
                for col in categorical_cols:
                    le = LabelEncoder()
                    X_for_importance[col] = le.fit_transform(X_for_importance[col].astype(str))

                # Fill NaN values
                X_for_importance = X_for_importance.fillna(0)

                # Train simple model for importance
                temp_model = xgb.XGBClassifier(
                    n_estimators=50, random_state=42, eval_metric="logloss", verbosity=0
                )
                temp_model.fit(X_for_importance, y)
                importances = temp_model.feature_importances_

                # Find low importance features
                low_importance_features = [
                    X_after_corr.columns[i]
                    for i, imp in enumerate(importances)
                    if imp < self.importance_threshold
                ]

                # Don't drop too many features - keep at least 5
                features_remaining = len(X_after_corr.columns) - len(low_importance_features)
                if features_remaining < 5:
                    # Keep top 5 most important features
                    importance_df = pd.DataFrame(
                        {"feature": X_after_corr.columns, "importance": importances}
                    ).sort_values("importance", ascending=False)

                    features_to_keep = importance_df.head(5)["feature"].tolist()
                    low_importance_features = [
                        f for f in X_after_corr.columns if f not in features_to_keep
                    ]

                self.dropped_features.extend(low_importance_features)
                logger.info(f"   Dropped {len(low_importance_features)} low importance features")

            except Exception as e:
                logger.warning(f"   Importance analysis failed: {str(e)[:50]}...")

        # Final selected features
        self.selected_features = [col for col in X_temp.columns if col not in self.dropped_features]

        # Ensure we have at least some features
        if len(self.selected_features) == 0:
            logger.warning("   No features selected, keeping all original features")
            self.selected_features = X_temp.columns.tolist()
        elif len(self.selected_features) < 3:
            logger.warning(
                f"   Only {len(self.selected_features)} features selected, keeping top features"
            )
            # If too few features, keep top numerical features
            numeric_cols = X_temp.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                self.selected_features = numeric_cols[:5]  # Keep top 5 numeric
            else:
                self.selected_features = X_temp.columns.tolist()[:5]  # Keep first 5

        logger.info(
            f"   Final selection: {len(self.selected_features)} features out of {len(X_temp.columns)}"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting important features."""
        X_temp = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # Make sure selected features exist in the data
        available_features = [f for f in self.selected_features if f in X_temp.columns]
        if len(available_features) != len(self.selected_features):
            logger.warning(
                f"   {len(self.selected_features) - len(available_features)} selected features not found in data"
            )

        return X_temp[available_features] if available_features else X_temp


class FlexibleColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Flexible transformer that adapts to available columns after preprocessing.
    Handles both numerical and categorical features with appropriate transformations.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize FlexibleColumnTransformer.

        Parameters:
        -----------
        random_state : int, default=42
            Random state for categorical encoder
        """
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = CatBoostEncoder(random_state=random_state)
        self.final_numerical_cols = None
        self.final_categorical_cols = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformers on available column types."""
        # Identify column types after preprocessing
        if hasattr(X, "columns"):
            self.final_numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self.final_categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        else:
            # If X is numpy array, assume all are numerical for now
            self.final_numerical_cols = list(range(X.shape[1]))
            self.final_categorical_cols = []

        logger.info(f"   Final numerical columns: {len(self.final_numerical_cols)}")
        logger.info(f"   Final categorical columns: {len(self.final_categorical_cols)}")

        # Fit transformers on available columns
        if self.final_numerical_cols and hasattr(X, "columns"):
            self.numerical_scaler.fit(X[self.final_numerical_cols])
        elif not hasattr(X, "columns"):  # numpy array case
            self.numerical_scaler.fit(X)

        if self.final_categorical_cols and hasattr(X, "columns"):
            self.categorical_encoder.fit(X[self.final_categorical_cols], y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scalers and encoders."""
        # X_transformed = X.copy() if hasattr(X, 'copy') else X

        if hasattr(X, "columns"):
            # DataFrame case
            transformed_parts = []

            # Transform numerical columns
            if self.final_numerical_cols:
                X_num_scaled = self.numerical_scaler.transform(X[self.final_numerical_cols])
                num_df = pd.DataFrame(
                    X_num_scaled,
                    columns=[f"num_{col}" for col in self.final_numerical_cols],
                    index=X.index,
                )
                transformed_parts.append(num_df)

            # Transform categorical columns
            if self.final_categorical_cols:
                X_cat_encoded = self.categorical_encoder.transform(X[self.final_categorical_cols])
                if hasattr(X_cat_encoded, "columns"):
                    cat_df = X_cat_encoded.copy()
                    cat_df.columns = [f"cat_{col}" for col in cat_df.columns]
                else:
                    cat_df = pd.DataFrame(
                        X_cat_encoded,
                        columns=[f"cat_{col}" for col in self.final_categorical_cols],
                        index=X.index,
                    )
                transformed_parts.append(cat_df)

            # Combine all parts
            if transformed_parts:
                X_final = pd.concat(transformed_parts, axis=1)
            else:
                X_final = pd.DataFrame(index=X.index)

        else:
            # Numpy array case
            X_final = self.numerical_scaler.transform(X)

        return X_final


class FeatureBuilder:
    """
    Main class for orchestrating all feature engineering operations.

    This class combines data preprocessing, feature selection, and transformation
    into a cohesive pipeline that prevents data leakage and ensures reproducibility.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize FeatureBuilder.

        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducible operations
        """
        self.random_state = random_state
        self.preprocessing_pipeline = None
        self.feature_names = None
        self.is_fitted = False

    def create_preprocessing_pipeline(self) -> Pipeline:
        """
        Create the complete preprocessing pipeline.

        Returns:
        --------
        Pipeline
            Scikit-learn pipeline with all preprocessing steps
        """
        logger.info("🏭 Creating preprocessing pipeline...")

        self.preprocessing_pipeline = Pipeline(
            [
                ("imputation", SmartImputer(drop_threshold=0.8)),
                ("winsorizing", Winsorizer(limits=(0.05, 0.05))),
                (
                    "feature_selection",
                    AdvancedFeatureSelector(correlation_threshold=0.95, importance_threshold=0.01),
                ),
                ("encoding_scaling", FlexibleColumnTransformer(random_state=self.random_state)),
                ("pca", PCA(n_components=0.90, random_state=self.random_state)),
                ("final_feature_selection", SelectKBest(f_classif, k=30)),
            ]
        )

        logger.info("✅ Preprocessing pipeline created!")
        return self.preprocessing_pipeline

    def split_data(
        self, data: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets with stratification.

        Parameters:
        -----------
        data : pd.DataFrame
            Complete dataset with target column
        test_size : float, default=0.2
            Proportion of data to use for testing

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        logger.info(f"✂️ Splitting data (test_size={test_size})...")

        if "target" not in data.columns:
            raise ValueError("Target column not found in dataset")

        X = data.drop("target", axis=1)
        y = data["target"]

        # Check target distribution for stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        except ValueError:
            # Fallback if stratification fails
            logger.warning("⚠️ Stratification failed, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

        logger.info("✅ Data split completed!")
        logger.info(f"   Training set: {X_train.shape[0]} samples")
        logger.info(f"   Test set: {X_test.shape[0]} samples")
        logger.info(f"   Training target distribution: {y_train.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def fit_transform_features(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """
        Fit the preprocessing pipeline and transform training data.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target

        Returns:
        --------
        pd.DataFrame
            Transformed training features
        """
        if self.preprocessing_pipeline is None:
            self.create_preprocessing_pipeline()

        logger.info("🔧 Fitting and transforming training features...")

        # Fit and transform training data
        X_train_transformed = self.preprocessing_pipeline.fit_transform(X_train, y_train)

        # Store feature names if possible
        if hasattr(X_train_transformed, "columns"):
            self.feature_names = X_train_transformed.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]

        self.is_fitted = True

        logger.info("✅ Training features transformed!")
        logger.info(f"   Input shape: {X_train.shape}")
        logger.info(f"   Output shape: {X_train_transformed.shape}")
        logger.info(f"   Features reduced by: {X_train.shape[1] - X_train_transformed.shape[1]}")

        return X_train_transformed

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted preprocessing pipeline.

        Parameters:
        -----------
        X : pd.DataFrame
            Features to transform

        Returns:
        --------
        pd.DataFrame
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform_features first.")

        logger.info("🔄 Transforming features...")
        X_transformed = self.preprocessing_pipeline.transform(X)

        logger.info("✅ Features transformed!")
        logger.info(f"   Input shape: {X.shape}")
        logger.info(f"   Output shape: {X_transformed.shape}")

        return X_transformed

    def build_features(
        self, data: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Complete feature building process from raw data to ML-ready features.

        Parameters:
        -----------
        data : pd.DataFrame
            Raw dataset with target column
        test_size : float, default=0.2
            Proportion of data to use for testing

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train_transformed, X_test_transformed, y_train, y_test
        """
        logger.info("🏗️ Starting complete feature building process...")

        # Step 1: Split data first to prevent leakage
        X_train, X_test, y_train, y_test = self.split_data(data, test_size)

        # Step 2: Fit and transform training features
        X_train_transformed = self.fit_transform_features(X_train, y_train)

        # Step 3: Transform test features
        X_test_transformed = self.transform_features(X_test)

        logger.info("🎉 Feature building completed successfully!")
        logger.info(f"   Final training shape: {X_train_transformed.shape}")
        logger.info(f"   Final test shape: {X_test_transformed.shape}")

        return X_train_transformed, X_test_transformed, y_train, y_test

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted feature transformation pipeline.

        Returns:
        --------
        Dict[str, Any]
            Information about features and transformations
        """
        if not self.is_fitted:
            return {"status": "error", "message": "Pipeline not fitted"}

        info = {
            "status": "success",
            "total_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "pipeline_steps": [step[0] for step in self.preprocessing_pipeline.steps],
            "is_fitted": self.is_fitted,
        }

        # Get information from each step if available
        for step_name, transformer in self.preprocessing_pipeline.steps:
            if hasattr(transformer, "selected_features"):
                info[f"{step_name}_selected_features"] = len(transformer.selected_features)
            if hasattr(transformer, "dropped_features"):
                info[f"{step_name}_dropped_features"] = len(transformer.dropped_features)

        return info


def create_preprocessing_pipeline(random_state: int = 42) -> Pipeline:
    """
    Factory function to create a preprocessing pipeline.

    Parameters:
    -----------
    random_state : int, default=42
        Random state for reproducible operations

    Returns:
    --------
    Pipeline
        Complete preprocessing pipeline
    """
    return Pipeline(
        [
            ("imputation", SmartImputer(drop_threshold=0.8)),
            ("winsorizing", Winsorizer(limits=(0.05, 0.05))),
            (
                "feature_selection",
                AdvancedFeatureSelector(correlation_threshold=0.95, importance_threshold=0.01),
            ),
            ("encoding_scaling", FlexibleColumnTransformer(random_state=random_state)),
            ("pca", PCA(n_components=0.90, random_state=random_state)),
            ("final_feature_selection", SelectKBest(f_classif, k=30)),
        ]
    )


def main():
    """Main function for testing the feature builder."""
    print("🚀 Multisim Feature Builder")
    print("=" * 40)

    # Example usage
    try:
        # Initialize feature builder
        feature_builder = FeatureBuilder(random_state=42)

        # This would normally load from the data module
        print("📂 Feature builder initialized")
        print("💡 To use with actual data:")
        print("   1. Load data using: from src.data.make_dataset import DatasetProcessor")
        print("   2. Process data: processor = DatasetProcessor()")
        print(
            "   3. Load dataset: data = processor.load_and_preprocess('multisim_dataset.parquet')"
        )
        print(
            "   4. Build features: X_train, X_test, y_train, y_test = feature_builder.build_features(data)"
        )

        # Show pipeline structure
        pipeline = feature_builder.create_preprocessing_pipeline()
        print("\n🔧 Preprocessing Pipeline Steps:")
        for i, (step_name, transformer) in enumerate(pipeline.steps):
            print(f"   {i+1}. {step_name}: {transformer.__class__.__name__}")

    except Exception as e:
        logger.error(f"❌ Error in feature builder: {str(e)}")
        print("💡 Please ensure all required packages are installed:")
        print("   pip install pandas numpy scikit-learn xgboost category-encoders scipy")


if __name__ == "__main__":
    main()
