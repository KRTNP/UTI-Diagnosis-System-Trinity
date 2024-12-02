import streamlit as st
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
            st.error(f"Error loading models: {str(e)}")
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
            st.warning(f"Missing features: {missing_features}")
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

def main():
    # Set page configuration
    st.set_page_config(page_title="UTI Risk Prediction", page_icon="🩺", layout="wide")
    
    # Title and description
    st.title("Urinary Tract Infection (UTI) Risk Prediction")
    st.markdown("""
    ### Disclaimer
    This is a predictive tool and should not replace professional medical advice.
    Always consult with a healthcare provider for accurate diagnosis and treatment.
    """)
    
    # Sidebar for input mode selection
    st.sidebar.header("Assessment Mode")
    input_mode = st.sidebar.radio("Select Input Depth", 
                                   ["Basic Symptom Assessment", 
                                    "Detailed Medical Evaluation"])
    
    # Initialize the UTI prediction model
    uti_model = UTIPredictionModel()
    
    # Patient data input
    st.header("Patient Information")
    
    # Create columns for input layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Core patient details
        age = st.number_input("Patient's Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["M", "F"])
    
    # Symptom inputs
    st.subheader("Symptoms")
    
    # Basic symptoms (always shown)
    basic_symptoms = [
        'frequent_urination', 'painful_urination', 'fever', 
        'urgent_urination', 'cloudy_urine'
    ]
    
    # Detailed additional symptoms
    detailed_symptoms = [
        'lower_abdominal_pain', 'blood_in_urine', 
        'foul_smelling_urine', 'nitrites', 'leukocyte_esterase'
    ]
    
    # Create columns for symptoms
    col3, col4 = st.columns(2)
    
    # Symptom input
    patient_data = {
        'age': age,
        'gender': gender
    }
    
    with col3:
        for symptom in basic_symptoms:
            patient_data[symptom] = st.radio(
                symptom.replace('_', ' ').title(), 
                [0, 1], 
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
    
    # Detailed mode additional inputs
    if input_mode == "Detailed Medical Evaluation":
        with col4:
            for symptom in detailed_symptoms:
                patient_data[symptom] = st.radio(
                    symptom.replace('_', ' ').title(), 
                    [0, 1], 
                    format_func=lambda x: "Yes" if x == 1 else "No"
                )
            
            # Additional medical details
            patient_data['urine_ph'] = st.number_input("Urine pH Level", min_value=0.0, max_value=14.0, value=6.0, step=0.1)
            patient_data['wbc'] = st.number_input("White Blood Cell Count", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
            patient_data['rbc'] = st.number_input("Red Blood Cell Count", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
            
            # Chronic conditions
            patient_data['diabetes'] = st.radio("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            patient_data['hypertension'] = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            patient_data['bacteria'] = st.radio("Bacteria Present", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    else:
        # Set default/neutral values for detailed features in basic mode
        default_features = (detailed_symptoms + 
                            ['wbc', 'rbc', 'urine_ph', 
                             'diabetes', 'hypertension', 'bacteria'])
        for feature in default_features:
            patient_data[feature] = 0
    
    # Prediction button
    if st.button("Predict UTI Risk"):
        try:
            # Predict UTI risk
            result = uti_model.predict(patient_data)
            
            # Display results
            st.header("Risk Assessment Results")
            
            # Prediction visualization
            if result['prediction'] == 1:
                st.error("UTI Risk: Positive")
                risk_color = "red"
            else:
                st.success("UTI Risk: Negative")
                risk_color = "green"
            
            # Detailed results
            col5, col6, col7 = st.columns(3)
            
            with col5:
                st.metric("Probability", f"{result['probability']:.2f}", 
                          help="Likelihood of having a UTI")
            
            with col6:
                st.metric("Confidence", f"{result['confidence']:.2f}", 
                          help="Model's confidence in the prediction")
            
            with col7:
                st.markdown(f"""
                <div style="background-color:{risk_color}; color:white; padding:10px; border-radius:5px;">
                <strong>Recommendation:</strong><br>
                {result['recommendation']}
                </div>
                """, unsafe_allow_html=True)
            
            # Additional information
            st.info("""
            ### Important Notes:
            - This is a predictive tool, not a definitive diagnosis
            - Always consult a healthcare professional
            - Your personal medical history is crucial for accurate assessment
            """)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()