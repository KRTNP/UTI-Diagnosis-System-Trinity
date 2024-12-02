import pandas as pd
import numpy as np
import joblib
from typing import Dict, Union

class UTIPredictionModel:
    def __init__(self, model_path='./models'):
        """
        Initialize the UTI prediction model with flexible model loading
        
        Args:
            model_path (str): Path to directory containing model files
        """
        self.model_artifacts = self._load_models(model_path)
    
    def _load_models(self, model_path):
        """Load trained models and associated parameters"""
        try:
            return {
                'rf_model': joblib.load(f'{model_path}/rf_model.joblib'),
                'xgb_model': joblib.load(f'{model_path}/xgb_model.joblib'),
                'scaler': joblib.load(f'{model_path}/scaler.joblib'),
                'thresholds': joblib.load(f'{model_path}/optimal_thresholds.joblib')
            }
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return None
    
    def _validate_input(self, patient_data: Dict[str, Union[int, float, str]]) -> bool:
        """
        Validate patient input against required features
        
        Args:
            patient_data (dict): Patient data dictionary
        
        Returns:
            bool: Whether input is valid
        """
        required_features = [
            # Symptoms
            'frequent_urination', 'painful_urination', 'lower_abdominal_pain', 
            'cloudy_urine', 'blood_in_urine', 'fever', 'urgent_urination', 
            'foul_smelling_urine', 'nitrites', 'leukocyte_esterase',
            
            # Numerical data
            'age', 'urine_ph', 'wbc', 'rbc',
            
            # Additional categorical data
            'gender', 'diabetes', 'hypertension', 'bacteria'
        ]
        
        # Check if all required features are present
        missing_features = [feat for feat in required_features if feat not in patient_data]
        
        if missing_features:
            print(f"Missing features: {missing_features}")
            return False
        
        return True
    
    def predict(self, patient_data: Dict[str, Union[int, float, str]]) -> Dict:
        """
        Main prediction method with flexible input handling
        
        Args:
            patient_data (dict): Patient data dictionary
        
        Returns:
            dict: Prediction results
        """
        # Validate input
        if not self._validate_input(patient_data):
            return {"error": "Invalid input data"}
        
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Preprocess data
        processed_data = self._preprocess_data(patient_df)
        
        # Get predictions from models
        rf_proba = self.model_artifacts['rf_model'].predict_proba(processed_data)[:, 1]
        xgb_proba = self.model_artifacts['xgb_model'].predict_proba(processed_data)[:, 1]
        
        # Calculate ensemble probability
        ensemble_proba = (rf_proba + xgb_proba) / 2
        
        # Determine confidence
        confidence_score = np.maximum(ensemble_proba, 1 - ensemble_proba)
        
        return {
            "prediction": 1 if ensemble_proba >= 0.5 else 0,
            "probability": float(ensemble_proba[0]),
            "confidence": float(confidence_score[0]),
            "recommendation": self._get_recommendation(ensemble_proba[0], confidence_score[0])
        }
    
    def _preprocess_data(self, data):
        """Preprocess input data for prediction"""
        # Feature order must match training data
        feature_order = [
            'age', 'urine_ph', 'wbc', 'rbc',
            'frequent_urination', 'painful_urination', 'lower_abdominal_pain', 
            'cloudy_urine', 'blood_in_urine', 'fever', 'urgent_urination', 
            'foul_smelling_urine', 'nitrites', 'leukocyte_esterase',
            'gender', 'diabetes', 'hypertension', 'bacteria'
        ]
        
        # Ensure correct order and encoding
        processed_data = data[feature_order].copy()
        processed_data['gender'] = processed_data['gender'].map({'M': 0, 'F': 1})
        
        # Scale numerical features
        numerical_features = ['age', 'urine_ph', 'wbc', 'rbc']
        processed_data[numerical_features] = self.model_artifacts['scaler'].transform(processed_data[numerical_features])
        
        return processed_data.values
    
    def _get_recommendation(self, probability, confidence):
        """Generate recommendation based on prediction and confidence"""
        if probability >= 0.5:
            if confidence >= 0.8:
                return "High probability of UTI. Immediate clinical evaluation recommended."
            elif confidence >= 0.6:
                return "Moderate UTI risk. Clinical evaluation within 24 hours recommended."
            else:
                return "Possible UTI. Monitor symptoms and consider medical consultation."
        else:
            if confidence >= 0.8:
                return "Low UTI probability. Continue monitoring symptoms."
            else:
                return "UTI unlikely, but consult healthcare provider if symptoms persist."

def input_patient_data(mode='basic'):
    """
    Collect patient data with different input depths
    
    Args:
        mode (str): 'basic' or 'detailed' input mode
    
    Returns:
        dict: Patient data dictionary
    """
    patient_data = {}
    
    # Basic symptom questions
    basic_symptoms = [
        'frequent_urination', 'painful_urination', 'fever', 
        'urgent_urination', 'cloudy_urine'
    ]
    
    # Detailed additional symptoms and tests
    detailed_symptoms = [
        'lower_abdominal_pain', 'blood_in_urine', 
        'foul_smelling_urine', 'nitrites', 'leukocyte_esterase'
    ]
    
    # Core questions for both modes
    patient_data['age'] = int(input("Patient's age: "))
    patient_data['gender'] = input("Gender (M/F): ").upper()
    
    # Basic mode: core symptoms
    if mode == 'basic':
        for symptom in basic_symptoms:
            patient_data[symptom] = int(input(f"{symptom.replace('_', ' ').title()} (0/1): "))
        
        # Set default/neutral values for other features
        default_features = (detailed_symptoms + 
                            ['wbc', 'rbc', 'urine_ph', 
                             'diabetes', 'hypertension', 'bacteria'])
        for feature in default_features:
            patient_data[feature] = 0
    
    # Detailed mode: comprehensive assessment
    else:
        # All symptoms
        for symptom in (basic_symptoms + detailed_symptoms):
            patient_data[symptom] = int(input(f"{symptom.replace('_', ' ').title()} (0/1): "))
        
        # Additional medical details
        patient_data['urine_ph'] = float(input("Urine pH level: "))
        patient_data['wbc'] = float(input("White Blood Cell count: "))
        patient_data['rbc'] = float(input("Red Blood Cell count: "))
        
        # Chronic conditions
        patient_data['diabetes'] = int(input("Diabetes (0/1): "))
        patient_data['hypertension'] = int(input("Hypertension (0/1): "))
        patient_data['bacteria'] = int(input("Bacteria present (0/1): "))
    
    return patient_data

def main():
    # Initialize the UTI prediction model
    uti_model = UTIPredictionModel()
    
    # Choose input mode
    print("Select Input Mode:")
    print("1. Basic Symptom Assessment")
    print("2. Detailed Medical Evaluation")
    
    mode_choice = input("Enter choice (1/2): ")
    input_mode = 'basic' if mode_choice == '1' else 'detailed'
    
    # Collect patient data
    try:
        patient_data = input_patient_data(input_mode)
        
        # Predict UTI risk
        result = uti_model.predict(patient_data)
        
        # Display results
        print("\n--- UTI Risk Assessment ---")
        print(f"Prediction: {'Positive' if result['prediction'] == 1 else 'Negative'}")
        print(f"Probability: {result['probability']:.2f}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Recommendation: {result['recommendation']}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()