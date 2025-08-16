"""
Dataset Creation and Preprocessing Module
==========================================

This module handles loading, cleaning, and initial preprocessing of the multisim dataset.
It performs data type conversions, missing value handling, and basic data validation.

Classes:
--------
DatasetProcessor: Main class for dataset processing operations

Functions:
----------
load_parquet_data: Load parquet files with error handling
validate_data: Validate dataset structure and content
"""

import pandas as pd
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class DatasetProcessor:
    """
    Handles loading and initial preprocessing of the multisim dataset.

    This class provides methods for loading parquet files, performing initial
    data type conversions, and basic preprocessing steps required before
    feature engineering.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the DatasetProcessor.

        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducible operations
        """
        self.random_state = random_state
        self.data = None
        self.original_shape = None
        self.processing_log = []

    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load the parquet dataset with comprehensive error handling.

        Parameters:
        -----------
        file_path : str
            Path to the parquet file

        Returns:
        --------
        pd.DataFrame or None
            Loaded dataset or None if loading fails
        """
        logger.info("📊 Loading dataset...")

        try:
            # Check if file exists
            if not Path(file_path).exists():
                logger.error(f"❌ File not found: {file_path}")
                return None

            # Load the dataset
            self.data = pd.read_parquet(file_path)
            self.original_shape = self.data.shape

            logger.info(f"✅ Dataset loaded successfully! Shape: {self.data.shape}")
            logger.info(
                f"   Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            )

            # Log basic info
            self.processing_log.append(f"Dataset loaded: {self.data.shape}")

            return self.data

        except Exception as e:
            logger.error(f"❌ Error loading dataset: {str(e)}")
            return None

    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the loaded dataset structure and content.

        Returns:
        --------
        Dict[str, Any]
            Validation results containing various checks
        """
        if self.data is None:
            return {"status": "error", "message": "No data loaded"}

        logger.info("🔍 Validating dataset...")

        validation_results = {
            "status": "success",
            "shape": self.data.shape,
            "columns": self.data.columns.tolist(),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "duplicate_rows": self.data.duplicated().sum(),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
        }

        # Check for target column
        if "target" not in self.data.columns:
            validation_results["warnings"] = validation_results.get("warnings", [])
            validation_results["warnings"].append("Target column not found")

        # Check for excessive missing values
        missing_pct = self.data.isnull().sum() / len(self.data) * 100
        high_missing_cols = missing_pct[missing_pct > 80].index.tolist()

        if high_missing_cols:
            validation_results["high_missing_columns"] = high_missing_cols

        logger.info("✅ Data validation completed")
        return validation_results

    def initial_preprocessing(self) -> pd.DataFrame:
        """
        Perform initial preprocessing as per pipeline requirements.

        This includes:
        - Converting telephone_number to index
        - Data type conversions
        - Creating new features
        - Dropping unnecessary columns

        Returns:
        --------
        pd.DataFrame
            Preprocessed dataset
        """
        if self.data is None:
            logger.error("❌ No data loaded. Please load data first.")
            return None

        logger.info("🔧 Performing initial preprocessing...")

        # Make a copy to avoid modifying original data
        processed_data = self.data.copy()

        # Convert telephone_number to index if exists
        if "telephone_number" in processed_data.columns:
            processed_data = processed_data.reset_index(drop=True)
            processed_data = processed_data.drop("telephone_number", axis=1)
            logger.info("✅ telephone_number converted to index and dropped")
            self.processing_log.append("Dropped telephone_number column")

        # Convert specified columns to int type with better error handling
        int_columns = ["age_dev", "dev_num", "is_dualsim", "is_featurephone", "is_smartphone"]
        for col in int_columns:
            if col in processed_data.columns:
                try:
                    # First convert to numeric, then to int
                    original_type = processed_data[col].dtype
                    processed_data[col] = (
                        pd.to_numeric(processed_data[col], errors="coerce").fillna(0).astype(int)
                    )
                    logger.info(f"✅ Converted {col} from {original_type} to int")
                    self.processing_log.append(f"Converted {col} to int")
                except Exception as e:
                    logger.warning(f"⚠️ Could not convert {col} to int: {str(e)}")

        # Convert target to float with better error handling
        if "target" in processed_data.columns:
            try:
                original_type = processed_data["target"].dtype
                processed_data["target"] = pd.to_numeric(
                    processed_data["target"], errors="coerce"
                ).astype(float)
                logger.info(f"✅ Target converted from {original_type} to float")

                # Check for any NaN values introduced
                nan_count = processed_data["target"].isna().sum()
                if nan_count > 0:
                    logger.warning(f"⚠️ {nan_count} NaN values introduced in target conversion")

                self.processing_log.append("Converted target to float")

            except Exception as e:
                logger.warning(f"⚠️ Could not convert target to float: {str(e)}")

        # Convert age column to numeric if it's not already
        if "age" in processed_data.columns:
            try:
                original_type = processed_data["age"].dtype
                if processed_data["age"].dtype == "object":
                    processed_data["age"] = pd.to_numeric(processed_data["age"], errors="coerce")
                    logger.info(f"✅ Age converted from {original_type} to numeric")
                    self.processing_log.append("Converted age to numeric")
            except Exception as e:
                logger.warning(f"⚠️ Could not convert age to numeric: {str(e)}")

        # Create new feature combining age and gender with error handling
        if "age" in processed_data.columns and "gndr" in processed_data.columns:
            try:
                # Handle potential NaN values
                age_filled = processed_data["age"].fillna("unknown")
                gndr_filled = processed_data["gndr"].fillna("U")
                processed_data["age_gender_combined"] = (
                    age_filled.astype(str) + "_" + gndr_filled.astype(str)
                )
                logger.info("✅ Created age_gender_combined feature")

                # Show some examples
                sample_values = processed_data["age_gender_combined"].head(3).tolist()
                logger.info(f"   Sample values: {sample_values}")
                self.processing_log.append("Created age_gender_combined feature")

            except Exception as e:
                logger.warning(f"⚠️ Could not create age_gender_combined: {str(e)}")

        # Drop all columns starting with 'val' AFTER creating new feature
        val_cols = [col for col in processed_data.columns if col.startswith("val")]
        if val_cols:
            processed_data = processed_data.drop(val_cols, axis=1)
            logger.info(f"✅ Dropped columns starting with 'val': {val_cols}")
            self.processing_log.append(f"Dropped {len(val_cols)} 'val' columns")

        # Drop all columns starting with other patterns
        cols_to_drop = [
            col for col in processed_data.columns if col.startswith(("temp_", "tmp_", "test_"))
        ]
        if cols_to_drop:
            processed_data = processed_data.drop(cols_to_drop, axis=1)
            logger.info(f"✅ Dropped columns: {cols_to_drop}")
            self.processing_log.append(f"Dropped {len(cols_to_drop)} temporary columns")

        # Update the stored data
        self.data = processed_data

        logger.info(f"✅ Initial preprocessing completed. New shape: {processed_data.shape}")
        logger.info(f"   Shape change: {self.original_shape} → {processed_data.shape}")

        # Display final data types
        logger.info("📋 Final Data Types:")
        for col in processed_data.columns:
            logger.info(f"   {col}: {processed_data[col].dtype}")

        self.processing_log.append(f"Final shape: {processed_data.shape}")

        return processed_data

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all processing steps performed.

        Returns:
        --------
        Dict[str, Any]
            Summary of processing steps and data changes
        """
        if self.data is None:
            return {"status": "error", "message": "No data processed"}

        return {
            "original_shape": self.original_shape,
            "final_shape": self.data.shape,
            "processing_steps": self.processing_log,
            "columns_dropped": self.original_shape[1] - self.data.shape[1],
            "final_columns": self.data.columns.tolist(),
            "target_available": "target" in self.data.columns,
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
        }

    def load_and_preprocess(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Convenience method to load and preprocess data in one call.

        Parameters:
        -----------
        file_path : str
            Path to the parquet file

        Returns:
        --------
        pd.DataFrame or None
            Fully processed dataset ready for feature engineering
        """
        # Load data
        if self.load_data(file_path) is None:
            return None

        # Validate data
        validation_results = self.validate_data()
        if validation_results["status"] == "error":
            logger.error("❌ Data validation failed")
            return None

        # Perform initial preprocessing
        processed_data = self.initial_preprocessing()

        if processed_data is not None:
            logger.info("🎉 Data loading and preprocessing completed successfully!")

        return processed_data


def load_parquet_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Standalone function to load parquet data with basic error handling.

    Parameters:
    -----------
    file_path : str
        Path to the parquet file

    Returns:
    --------
    pd.DataFrame or None
        Loaded dataset or None if loading fails
    """
    processor = DatasetProcessor()
    return processor.load_data(file_path)


def validate_data(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function to validate dataset structure and content.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to validate

    Returns:
    --------
    Dict[str, Any]
        Validation results
    """
    processor = DatasetProcessor()
    processor.data = data
    return processor.validate_data()


def main():
    """Main function for testing the dataset processor."""
    print("🚀 Multisim Dataset Processor")
    print("=" * 40)

    # Initialize processor
    processor = DatasetProcessor()

    # Example usage
    file_path = "multisim_dataset.csv"
    print(f"📂 Attempting to load: {file_path}")

    # Load and preprocess data
    processed_data = processor.load_and_preprocess(file_path)

    if processed_data is not None:
        print("\n📊 Processing Summary:")
        summary = processor.get_processing_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
    else:
        print("❌ Failed to process dataset")
        print("💡 Please ensure the dataset file exists and is accessible")


if __name__ == "__main__":
    main()
