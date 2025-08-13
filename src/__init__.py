"""
Multisim ML Pipeline Package
============================

A comprehensive machine learning pipeline for predicting multisim usage behavior.

This package provides end-to-end functionality for:
- Data loading and preprocessing
- Feature engineering and selection
- Model training and hyperparameter optimization
- Model evaluation and visualization
- Model serialization and deployment

Modules:
--------
- data.make_dataset: Data loading and initial preprocessing
- features.build_features: Feature engineering and transformation
- models.train_model: Model training and hyperparameter tuning
- models.predict_model: Model prediction and inference
- visualization.visualize: Data visualization and model analysis

Usage:
------
```python
from src.data.make_dataset import DatasetProcessor
from src.features.build_features import FeatureBuilder
from src.models.train_model import ModelTrainer

# Load and preprocess data
processor = DatasetProcessor()
data = processor.load_and_preprocess('multisim_dataset.parquet')

# Build features
feature_builder = FeatureBuilder()
X_train, X_test, y_train, y_test = feature_builder.build_features(data)

# Train model
trainer = ModelTrainer()
model = trainer.train(X_train, y_train)
```

Author: Mirali Abdullayev
Version: 1.0.0
"""

__author__ = "Mirali Abdullayev"
__version__ = "1.0.0"

# Package metadata
__all__ = ["data", "features", "models", "visualization"]


def main():
    """Main entry point for the package"""
    print("🚀 Multisim ML Pipeline Package")
    print("=" * 40)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print("\nAvailable modules:")
    print("  📊 data.make_dataset - Data loading and preprocessing")
    print("  🔧 features.build_features - Feature engineering")
    print("  🤖 models.train_model - Model training")
    print("  📈 models.predict_model - Model prediction")
    print("  📉 visualization.visualize - Visualization tools")
    print("\nFor detailed usage, check individual module documentation.")


if __name__ == "__main__":
    main()
