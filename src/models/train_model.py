"""
Model Training and Hyperparameter Optimization Module
======================================================

This module handles the complete model training workflow including:
- Baseline model training with XGBoost
- Hyperparameter optimization using Optuna
- Model evaluation and performance metrics
- Model serialization and saving

Classes:
--------
ModelTrainer: Main class for training ML models with hyperparameter optimization
ModelEvaluator: Class for comprehensive model evaluation and metrics calculation
ModelSerializer: Class for saving and loading trained models

Functions:
----------
calculate_metrics: Calculate comprehensive classification metrics
optimize_hyperparameters: Hyperparameter optimization with Optuna
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import warnings
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# Import custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from features.build_features import FeatureBuilder, create_preprocessing_pipeline
    from data.make_dataset import DatasetProcessor
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Could not import custom modules. Using standalone mode.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation with detailed metrics calculation.
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : np.ndarray, optional
            Predicted probabilities for positive class
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing all calculated metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Add AUC-ROC if probabilities are provided
        if y_pred_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['auc_roc'] = 0.5  # Default for single class
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset") -> None:
        """
        Print metrics in a formatted way.
        
        Parameters:
        -----------
        metrics : Dict[str, float]
            Metrics dictionary
        dataset_name : str, default="Dataset"
            Name of the dataset for display
        """
        logger.info(f"   {dataset_name.upper()} METRICS:")
        for metric, value in metrics.items():
            logger.info(f"     {metric.upper()}: {value:.4f}")
    
    @staticmethod
    def evaluate_model_performance(model, X_train: pd.DataFrame, y_train: pd.Series, 
                                 X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive model evaluation on both training and test sets.
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained model
        X_train, y_train : pd.DataFrame, pd.Series
            Training data and labels
        X_test, y_test : pd.DataFrame, pd.Series
            Test data and labels
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Nested dictionary with train and test metrics
        """
        # Training predictions
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Test predictions
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        train_metrics = ModelEvaluator.calculate_metrics(y_train, y_train_pred, y_train_pred_proba)
        test_metrics = ModelEvaluator.calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
        
        return {
            'train': train_metrics,
            'test': test_metrics
        }


class ModelSerializer:
    """
    Handles model serialization and deserialization operations.
    """
    
    @staticmethod
    def save_model(model, filepath: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Save trained model with metadata to disk.
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained model to save
        filepath : str
            Path where to save the model
        metadata : Dict[str, Any], optional
            Additional metadata to save with model
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare model package
            model_package = {
                'model': model,
                'metadata': metadata or {},
                'saved_at': datetime.now().isoformat(),
                'model_type': type(model).__name__
            }
            
            # Save using joblib for sklearn models
            joblib.dump(model_package, filepath)
            
            logger.info(f"✅ Model saved successfully to: {filepath}")
            logger.info(f"   Model type: {type(model).__name__}")
            logger.info(f"   File size: {Path(filepath).stat().st_size / 1024**2:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {str(e)}")
            return False
    
    @staticmethod
    def load_model(filepath: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load trained model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        Tuple[Any, Dict[str, Any]]
            Loaded model and its metadata
        """
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            # Load model package
            model_package = joblib.load(filepath)
            
            model = model_package['model']
            metadata = model_package.get('metadata', {})
            
            logger.info(f"✅ Model loaded successfully from: {filepath}")
            logger.info(f"   Model type: {model_package.get('model_type', 'Unknown')}")
            logger.info(f"   Saved at: {model_package.get('saved_at', 'Unknown')}")
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            return None, {}


class ModelTrainer:
    """
    Main class for training ML models with hyperparameter optimization.
    
    This class provides methods for baseline training, hyperparameter tuning
    using Optuna, and comprehensive model evaluation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducible operations
        """
        self.random_state = random_state
        self.baseline_model = None
        self.best_model = None
        self.feature_builder = None
        self.best_params = None
        self.training_history = []
        
    def create_baseline_model(self) -> Pipeline:
        """
        Create a baseline XGBoost model with preprocessing pipeline.
        
        Returns:
        --------
        Pipeline
            Complete pipeline with preprocessing and XGBoost model
        """
        logger.info("🏗️ Creating baseline model pipeline...")
        
        # Create preprocessing pipeline
        preprocessing_pipeline = create_preprocessing_pipeline(self.random_state)
        
        # Create full pipeline with XGBoost
        pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('model', xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=2,
                objective='binary:logistic',
                verbosity=0
            ))
        ])
        
        logger.info("✅ Baseline model pipeline created!")
        return pipeline
    
    def train_baseline(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Train baseline XGBoost model and evaluate performance.
        
        Parameters:
        -----------
        X_train, y_train : pd.DataFrame, pd.Series
            Training data and labels
        X_test, y_test : pd.DataFrame, pd.Series
            Test data and labels
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Training and test metrics
        """
        logger.info("🎯 Training baseline XGBoost model...")
        
        # Create and train baseline model
        self.baseline_model = self.create_baseline_model()
        self.baseline_model.fit(X_train, y_train)
        
        # Evaluate performance
        metrics = ModelEvaluator.evaluate_model_performance(
            self.baseline_model, X_train, y_train, X_test, y_test
        )
        
        logger.info("✅ Baseline Model Performance:")
        ModelEvaluator.print_metrics(metrics['train'], "train")
        ModelEvaluator.print_metrics(metrics['test'], "test")
        
        # Store in training history
        self.training_history.append({
            'model_type': 'baseline',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        return metrics
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series, 
                                n_trials: int = 100) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
        """
        Hyperparameter optimization using Optuna.
        
        Parameters:
        -----------
        X_train, y_train : pd.DataFrame, pd.Series
            Training data and labels
        X_test, y_test : pd.DataFrame, pd.Series
            Test data and labels (for validation during optimization)
        n_trials : int, default=100
            Number of optimization trials
            
        Returns:
        --------
        Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]
            Best model metrics and best parameters
        """
        logger.info(f"🔬 Starting hyperparameter optimization with Optuna ({n_trials} trials)...")
        
        def objective(trial):
            """Optuna objective function for hyperparameter optimization."""
            # Create a new pipeline for each trial
            temp_pipeline = self.create_baseline_model()
            
            # Suggest hyperparameters for better performance
            params = {
                'model__n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'model__max_depth': trial.suggest_int('max_depth', 4, 12),
                'model__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'model__subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'model__colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'model__reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'model__reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'model__min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'model__gamma': trial.suggest_float('gamma', 0, 2),
                'preprocessing__pca__n_components': trial.suggest_float('pca_components', 0.85, 0.99),
                'preprocessing__final_feature_selection__k': trial.suggest_int('feature_k', 15, 100)
            }
            
            # Set parameters
            temp_pipeline.set_params(**params)
            
            # Fit and predict
            temp_pipeline.fit(X_train, y_train)
            y_pred_proba = temp_pipeline.predict_proba(X_test)[:, 1]
            
            return roc_auc_score(y_test, y_pred_proba)
        
        # Run optimization with progress tracking
        study = optuna.create_study(direction='maximize')
        
        # Add callback for progress tracking
        def callback(study, trial):
            if trial.number % 10 == 0:
                logger.info(f"   Trial {trial.number}: Best AUC = {study.best_value:.4f}")
        
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        
        # Get best parameters
        self.best_params = study.best_params
        logger.info(f"✅ Optimization completed! Best AUC: {study.best_value:.4f}")
        logger.info(f"📋 Best parameters:")
        for param, value in self.best_params.items():
            logger.info(f"   {param}: {value}")
        
        # Train best model
        self.best_model = self.create_baseline_model()
        
        # Format parameters for pipeline
        formatted_params = {}
        for key, value in self.best_params.items():
            if key == 'pca_components':
                formatted_params['preprocessing__pca__n_components'] = value
            elif key == 'feature_k':
                formatted_params['preprocessing__final_feature_selection__k'] = value
            else:
                formatted_params[f'model__{key}'] = value
        
        # Set best parameters and train
        self.best_model.set_params(**formatted_params)
        self.best_model.fit(X_train, y_train)
        
        # Evaluate best model
        best_metrics = ModelEvaluator.evaluate_model_performance(
            self.best_model, X_train, y_train, X_test, y_test
        )
        
        logger.info("✅ Best Model Performance:")
        ModelEvaluator.print_metrics(best_metrics['train'], "train")
        ModelEvaluator.print_metrics(best_metrics['test'], "test")
        
        # Store in training history
        self.training_history.append({
            'model_type': 'optimized',
            'timestamp': datetime.now().isoformat(),
            'metrics': best_metrics,
            'parameters': self.best_params,
            'n_trials': n_trials
        })
        
        return best_metrics, self.best_params
    
    def train_complete_pipeline(self, data: pd.DataFrame, test_size: float = 0.2, 
                               n_trials: int = 50) -> Dict[str, Any]:
        """
        Train complete ML pipeline from raw data to optimized model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw dataset with target column
        test_size : float, default=0.2
            Proportion of data for testing
        n_trials : int, default=50
            Number of optimization trials
            
        Returns:
        --------
        Dict[str, Any]
            Complete training results including metrics and models
        """
        logger.info("🚀 Starting complete model training pipeline...")
        logger.info("="*60)
        
        # Step 1: Feature engineering
        self.feature_builder = FeatureBuilder(random_state=self.random_state)
        X_train, X_test, y_train, y_test = self.feature_builder.build_features(data, test_size)
        
        # Step 2: Train baseline model
        logger.info("\n📊 Phase 1: Baseline Model Training")
        baseline_metrics = self.train_baseline(X_train, y_train, X_test, y_test)
        
        # Step 3: Hyperparameter optimization
        logger.info(f"\n🔬 Phase 2: Hyperparameter Optimization")
        best_metrics, best_params = self.optimize_hyperparameters(
            X_train, y_train, X_test, y_test, n_trials
        )
        
        # Step 4: Performance comparison
        logger.info(f"\n📈 Phase 3: Performance Analysis")
        self._analyze_performance(baseline_metrics, best_metrics)
        
        # Compile results
        results = {
            'baseline_metrics': baseline_metrics,
            'best_metrics': best_metrics,
            'best_params': best_params,
            'baseline_model': self.baseline_model,
            'best_model': self.best_model,
            'feature_builder': self.feature_builder,
            'training_history': self.training_history,
            'data_shapes': {
                'train': X_train.shape,
                'test': X_test.shape
            }
        }
        
        logger.info("🎉 Complete training pipeline finished!")
        return results
    
    def _analyze_performance(self, baseline_metrics: Dict[str, Dict[str, float]], 
                           best_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Analyze and compare baseline vs optimized model performance.
        
        Parameters:
        -----------
        baseline_metrics : Dict[str, Dict[str, float]]
            Baseline model metrics
        best_metrics : Dict[str, Dict[str, float]]
            Optimized model metrics
        """
        logger.info("📊 Model Performance Comparison:")
        logger.info("   BASELINE MODEL:")
        logger.info(f"     Train AUC: {baseline_metrics['train']['auc_roc']:.4f}")
        logger.info(f"     Test AUC:  {baseline_metrics['test']['auc_roc']:.4f}")
        logger.info("   OPTIMIZED MODEL:")
        logger.info(f"     Train AUC: {best_metrics['train']['auc_roc']:.4f}")
        logger.info(f"     Test AUC:  {best_metrics['test']['auc_roc']:.4f}")
        
        # Calculate improvement
        improvement = best_metrics['test']['auc_roc'] - baseline_metrics['test']['auc_roc']
        logger.info("   IMPROVEMENT:")
        logger.info(f"     Test AUC Improvement: {improvement:.4f}")
        
        # Check for overfitting
        train_test_diff = best_metrics['train']['auc_roc'] - best_metrics['test']['auc_roc']
        if train_test_diff > 0.1:
            logger.warning(f"   ⚠️ Potential overfitting detected (Train-Test AUC diff: {train_test_diff:.4f})")
        else:
            logger.info(f"   ✅ Good generalization (Train-Test AUC diff: {train_test_diff:.4f})")
    
    def save_trained_models(self, base_path: str = "models/") -> Dict[str, bool]:
        """
        Save all trained models to disk.
        
        Parameters:
        -----------
        base_path : str, default="models/"
            Base directory for saving models
            
        Returns:
        --------
        Dict[str, bool]
            Success status for each model saved
        """
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare metadata
        metadata = {
            'training_timestamp': timestamp,
            'random_state': self.random_state,
            'training_history': self.training_history,
            'best_params': self.best_params
        }
        
        # Save baseline model
        if self.baseline_model is not None:
            baseline_path = f"{base_path}baseline_model_{timestamp}.pkl"
            results['baseline'] = ModelSerializer.save_model(
                self.baseline_model, baseline_path, 
                {**metadata, 'model_type': 'baseline'}
            )
        
        # Save best model
        if self.best_model is not None:
            best_path = f"{base_path}best_model_{timestamp}.pkl"
            results['best'] = ModelSerializer.save_model(
                self.best_model, best_path,
                {**metadata, 'model_type': 'optimized'}
            )
            
            # Also save as 'latest' for easy loading
            latest_path = f"{base_path}latest_model.pkl"
            results['latest'] = ModelSerializer.save_model(
                self.best_model, latest_path,
                {**metadata, 'model_type': 'latest'}
            )
        
        # Save feature builder
        if self.feature_builder is not None:
            feature_path = f"{base_path}feature_builder_{timestamp}.pkl"
            results['feature_builder'] = ModelSerializer.save_model(
                self.feature_builder, feature_path,
                {**metadata, 'model_type': 'feature_builder'}
            )
        
        return results
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of training process.
        
        Returns:
        --------
        Dict[str, Any]
            Complete training summary
        """
        return {
            'models_trained': len(self.training_history),
            'best_model_available': self.best_model is not None,
            'baseline_model_available': self.baseline_model is not None,
            'feature_builder_available': self.feature_builder is not None,
            'best_parameters': self.best_params,
            'training_history': self.training_history,
            'random_state': self.random_state
        }


def train_model_from_file(file_path: str, test_size: float = 0.2, n_trials: int = 50) -> Dict[str, Any]:
    """
    Convenience function to train model directly from parquet file.
    
    Parameters:
    -----------
    file_path : str
        Path to the parquet dataset
    test_size : float, default=0.2
        Test set proportion
    n_trials : int, default=50
        Number of optimization trials
        
    Returns:
    --------
    Dict[str, Any]
        Complete training results
    """
    # Load and preprocess data
    processor = DatasetProcessor()
    data = processor.load_and_preprocess(file_path)
    
    if data is None:
        logger.error("❌ Failed to load data")
        return {}
    
    # Train model
    trainer = ModelTrainer()
    results = trainer.train_complete_pipeline(data, test_size, n_trials)
    
    # Save models
    save_results = trainer.save_trained_models()
    results['save_results'] = save_results
    
    return results


def load_trained_model(model_path: str = "models/latest_model.pkl") -> Tuple[Any, Dict[str, Any]]:
    """
    Load a previously trained model.
    
    Parameters:
    -----------
    model_path : str, default="models/latest_model.pkl"
        Path to the saved model
        
    Returns:
    --------
    Tuple[Any, Dict[str, Any]]
        Loaded model and metadata
    """
    return ModelSerializer.load_model(model_path)


def main():
    """Main function for training the multisim prediction model."""
    print("🚀 Multisim Model Trainer")
    print("="*40)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(random_state=42)
        
        # Example training workflow
        file_path = 'multisim_dataset.parquet'
        print(f"📂 Training model with dataset: {file_path}")
        
        # Check if data file exists
        if not Path(file_path).exists():
            print(f"❌ Dataset file not found: {file_path}")
            print("💡 Please ensure the dataset file exists in the current directory")
            print("📖 Usage example:")
            print("   1. Place 'multisim_dataset.parquet' in current directory")
            print("   2. Run: python train_model.py")
            print("   3. Models will be saved to 'models/' directory")
            return
        
        # Train complete pipeline
        print("🏁 Starting training process...")
        results = train_model_from_file(file_path, test_size=0.2, n_trials=30)
        
        if results:
            print("\n🎉 Training completed successfully!")
            print("📊 Final Results:")
            
            # Show best performance
            best_metrics = results.get('best_metrics', {})
            if best_metrics:
                test_auc = best_metrics['test']['auc_roc']
                test_f1 = best_metrics['test']['f1_score']
                print(f"   🎯 Best Test AUC: {test_auc:.4f}")
                print(f"   🎯 Best Test F1:  {test_f1:.4f}")
            
            # Show save results
            save_results = results.get('save_results', {})
            if save_results:
                print("💾 Models saved:")
                for model_type, success in save_results.items():
                    status = "✅" if success else "❌"
                    print(f"   {status} {model_type} model")
        else:
            print("❌ Training failed")
            
    except Exception as e:
        logger.error(f"❌ Error in training: {str(e)}")
        print(f"💡 Error details: {str(e)}")
        print("🔧 Please check your data and dependencies")


if __name__ == "__main__":
    main()