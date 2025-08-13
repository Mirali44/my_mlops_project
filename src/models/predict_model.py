import pandas as pd
import numpy as np
import joblib
import warnings
import os
from pathlib import Path
import json
from datetime import datetime
from typing import Union, Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

class MultisimPredictor:
    """
    Model prediction class for Multisim dataset.
    Handles model loading, preprocessing, and prediction.
    """
    
    def __init__(self, model_path: str = 'best_multisim_model.pkl'):
        """
        Initialize the predictor with model path.
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.model_metadata = {}
        self.is_loaded = False
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the trained model from disk.
        
        Args:
            model_path (Optional[str]): Path to model file. If None, uses instance path.
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if model_path:
            self.model_path = model_path
            
        print(f"🔄 Loading model from: {self.model_path}")
        
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            # Load the model
            self.model = joblib.load(self.model_path)
            print(f"✅ Model loaded successfully!")
            
            # Try to load metadata if available
            metadata_path = self.model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                print(f"✅ Model metadata loaded!")
            
            self.is_loaded = True
            
            # Display model info
            self._display_model_info()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def _display_model_info(self):
        """Display information about the loaded model."""
        print(f"\n📋 Model Information:")
        print(f"   Model Type: {type(self.model).__name__}")
        
        if hasattr(self.model, 'named_steps'):
            print(f"   Pipeline Steps: {list(self.model.named_steps.keys())}")
        
        if self.model_metadata:
            print(f"   Training Date: {self.model_metadata.get('training_date', 'Unknown')}")
            print(f"   Best AUC Score: {self.model_metadata.get('best_auc', 'Unknown')}")
            print(f"   Feature Count: {self.model_metadata.get('feature_count', 'Unknown')}")
    
    def preprocess_input_data(self, data: Union[pd.DataFrame, dict, list]) -> pd.DataFrame:
        """
        Preprocess input data to match training format.
        
        Args:
            data: Input data in various formats
            
        Returns:
            pd.DataFrame: Preprocessed data ready for prediction
        """
        print(f"🔧 Preprocessing input data...")
        
        # Convert input to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be dict, list, or DataFrame")
        
        # Apply same preprocessing as training (basic transformations)
        print(f"   Input shape: {df.shape}")
        
        # Convert specified columns to int type (same as training)
        int_columns = ['age_dev', 'dev_num', 'is_dualsim', 'is_featurephone', 'is_smartphone']
        for col in int_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                except:
                    pass
        
        # Convert age to numeric if needed
        if 'age' in df.columns:
            try:
                if df['age'].dtype == 'object':
                    df['age'] = pd.to_numeric(df['age'], errors='coerce')
            except:
                pass
        
        # Create age_gender_combined feature if both columns exist
        if 'age' in df.columns and 'gndr' in df.columns:
            try:
                age_filled = df['age'].fillna('unknown')
                gndr_filled = df['gndr'].fillna('U')
                df['age_gender_combined'] = age_filled.astype(str) + '_' + gndr_filled.astype(str)
            except:
                pass
        
        # Drop columns that were dropped during training
        cols_to_drop = []
        
        # Drop val columns
        val_cols = [col for col in df.columns if col.startswith('val')]
        cols_to_drop.extend(val_cols)
        
        # Drop temp/tmp/test columns
        temp_cols = [col for col in df.columns if col.startswith(('temp_', 'tmp_', 'test_'))]
        cols_to_drop.extend(temp_cols)
        
        # Drop telephone_number if exists
        if 'telephone_number' in df.columns:
            cols_to_drop.append('telephone_number')
        
        # Drop target if exists (for prediction)
        if 'target' in df.columns:
            cols_to_drop.append('target')
        
        # Remove duplicates and drop existing columns
        cols_to_drop = list(set(cols_to_drop))
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        if existing_cols_to_drop:
            df = df.drop(existing_cols_to_drop, axis=1)
            print(f"   Dropped columns: {existing_cols_to_drop}")
        
        print(f"   Preprocessed shape: {df.shape}")
        return df
    
    def predict(self, data: Union[pd.DataFrame, dict, list]) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            data: Input data for prediction
            
        Returns:
            np.ndarray: Binary predictions (0 or 1)
        """
        if not self.is_loaded:
            print("❌ Model not loaded. Please load model first.")
            return None
        
        print(f"🔮 Making predictions...")
        
        try:
            # Preprocess the data
            processed_data = self.preprocess_input_data(data)
            
            # Make predictions
            predictions = self.model.predict(processed_data)
            
            print(f"✅ Predictions completed!")
            print(f"   Predicted {len(predictions)} samples")
            print(f"   Multisimmers predicted: {np.sum(predictions == 1)}")
            print(f"   Non-multisimmers predicted: {np.sum(predictions == 0)}")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            return None
    
    def predict_proba(self, data: Union[pd.DataFrame, dict, list]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            data: Input data for prediction
            
        Returns:
            np.ndarray: Prediction probabilities [prob_class_0, prob_class_1]
        """
        if not self.is_loaded:
            print("❌ Model not loaded. Please load model first.")
            return None
        
        print(f"🎯 Getting prediction probabilities...")
        
        try:
            # Preprocess the data
            processed_data = self.preprocess_input_data(data)
            
            # Get probabilities
            probabilities = self.model.predict_proba(processed_data)
            
            print(f"✅ Probability prediction completed!")
            print(f"   Average multisim probability: {probabilities[:, 1].mean():.4f}")
            
            return probabilities
            
        except Exception as e:
            print(f"❌ Error during probability prediction: {str(e)}")
            return None
    
    def predict_with_details(self, data: Union[pd.DataFrame, dict, list]) -> Dict:
        """
        Make predictions with detailed output including probabilities and confidence.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Dict: Detailed prediction results
        """
        if not self.is_loaded:
            print("❌ Model not loaded. Please load model first.")
            return None
        
        print(f"📊 Making detailed predictions...")
        
        try:
            # Get both predictions and probabilities
            predictions = self.predict(data)
            probabilities = self.predict_proba(data)
            
            if predictions is None or probabilities is None:
                return None
            
            # Calculate confidence scores
            confidence_scores = np.max(probabilities, axis=1)
            
            # Create detailed results
            results = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'confidence_scores': confidence_scores.tolist(),
                'multisim_probabilities': probabilities[:, 1].tolist(),
                'summary': {
                    'total_samples': len(predictions),
                    'predicted_multisimmers': int(np.sum(predictions == 1)),
                    'predicted_non_multisimmers': int(np.sum(predictions == 0)),
                    'average_multisim_probability': float(probabilities[:, 1].mean()),
                    'average_confidence': float(confidence_scores.mean()),
                    'high_confidence_predictions': int(np.sum(confidence_scores > 0.8))
                }
            }
            
            print(f"✅ Detailed predictions completed!")
            return results
            
        except Exception as e:
            print(f"❌ Error during detailed prediction: {str(e)}")
            return None
    
    def batch_predict(self, data_path: str, output_path: Optional[str] = None) -> bool:
        """
        Make predictions on a batch file and save results.
        
        Args:
            data_path (str): Path to input data file (CSV or Parquet)
            output_path (Optional[str]): Path to save predictions
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_loaded:
            print("❌ Model not loaded. Please load model first.")
            return False
        
        print(f"📁 Processing batch file: {data_path}")
        
        try:
            # Load data based on file extension
            if data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            elif data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            else:
                print("❌ Unsupported file format. Use CSV or Parquet.")
                return False
            
            print(f"   Loaded {len(data)} samples")
            
            # Make predictions
            detailed_results = self.predict_with_details(data)
            
            if detailed_results is None:
                return False
            
            # Prepare output DataFrame
            output_df = data.copy()
            output_df['multisim_prediction'] = detailed_results['predictions']
            output_df['multisim_probability'] = detailed_results['multisim_probabilities']
            output_df['prediction_confidence'] = detailed_results['confidence_scores']
            
            # Save results
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"predictions_{timestamp}.csv"
            
            output_df.to_csv(output_path, index=False)
            print(f"✅ Predictions saved to: {output_path}")
            
            # Print summary
            print(f"\n📋 Batch Prediction Summary:")
            for key, value in detailed_results['summary'].items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during batch prediction: {str(e)}")
            return False
    
    def save_predictions_report(self, predictions_data: Dict, report_path: str = None) -> bool:
        """
        Save a detailed prediction report.
        
        Args:
            predictions_data (Dict): Prediction results from predict_with_details
            report_path (str): Path to save the report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if report_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = f"prediction_report_{timestamp}.json"
            
            # Add metadata to report
            report_data = {
                'report_timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'model_metadata': self.model_metadata,
                'predictions': predictions_data
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"✅ Prediction report saved to: {report_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving report: {str(e)}")
            return False


def load_sample_data() -> pd.DataFrame:
    """
    Create sample data for testing predictions.
    
    Returns:
        pd.DataFrame: Sample data
    """
    sample_data = {
        'age': [25, 35, 45, 30, 28],
        'gndr': ['M', 'F', 'M', 'F', 'M'],
        'tenure': [365, 730, 1095, 500, 200],
        'age_dev': [12, 24, 36, 18, 6],
        'dev_num': [1, 2, 1, 3, 1],
        'is_dualsim': [0, 1, 0, 1, 0],
        'is_featurephone': [0, 0, 1, 0, 0],
        'is_smartphone': [1, 1, 0, 1, 1],
        'dev_man': ['Samsung', 'Apple', 'Nokia', 'Huawei', 'Samsung'],
        'device_os_name': ['Android', 'iOS', 'KaiOS', 'Android', 'Android'],
        'simcard_type': ['prepaid', 'postpaid', 'prepaid', 'postpaid', 'prepaid'],
        'region': ['urban', 'suburban', 'rural', 'urban', 'suburban']
    }
    
    return pd.DataFrame(sample_data)


def main():
    """Main function to demonstrate prediction functionality."""
    print("🔮 Multisim Prediction Model")
    print("=" * 50)
    
    # Initialize predictor
    predictor = MultisimPredictor()
    
    # Try to load model
    if not predictor.load_model():
        print("\n⚠️ No trained model found. Please train a model first using train_model.py")
        print("   Using sample model path: 'best_multisim_model.pkl'")
        return
    
    # Create sample data for prediction
    print(f"\n📊 Creating sample data for prediction...")
    sample_data = load_sample_data()
    print(f"   Sample data shape: {sample_data.shape}")
    print(f"   Sample data preview:")
    print(sample_data.head())
    
    # Make simple predictions
    print(f"\n🎯 Making simple predictions...")
    predictions = predictor.predict(sample_data)
    
    if predictions is not None:
        print(f"   Predictions: {predictions}")
    
    # Make detailed predictions
    print(f"\n📈 Making detailed predictions...")
    detailed_results = predictor.predict_with_details(sample_data)
    
    if detailed_results:
        print(f"   Detailed results summary:")
        for key, value in detailed_results['summary'].items():
            print(f"     {key}: {value}")
        
        # Save prediction report
        predictor.save_predictions_report(detailed_results)
    
    # Example of single prediction
    print(f"\n🔍 Single customer prediction example...")
    single_customer = {
        'age': 30,
        'gndr': 'M',
        'tenure': 500,
        'age_dev': 18,
        'dev_num': 2,
        'is_dualsim': 1,
        'is_featurephone': 0,
        'is_smartphone': 1,
        'dev_man': 'Samsung',
        'device_os_name': 'Android',
        'simcard_type': 'postpaid',
        'region': 'urban'
    }
    
    single_pred = predictor.predict_with_details(single_customer)
    if single_pred:
        pred = single_pred['predictions'][0]
        prob = single_pred['multisim_probabilities'][0]
        confidence = single_pred['confidence_scores'][0]
        
        print(f"   Customer Profile: {single_customer}")
        print(f"   Prediction: {'Multisimmer' if pred == 1 else 'Non-multisimmer'}")
        print(f"   Multisim Probability: {prob:.4f}")
        print(f"   Confidence: {confidence:.4f}")
    
    print(f"\n✅ Prediction demonstration completed!")


if __name__ == "__main__":
    main()