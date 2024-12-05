# UTI Diagnosis System

An AI-based system for preliminary diagnosis of Urinary Tract Infections (UTIs) based on user-reported symptoms and laboratory results. The system leverages machine learning and natural language processing (NLP) to provide early assessments, supporting healthcare providers and raising awareness.

## Project Overview

- **Exploratory Data Analysis:** Understanding relationships between UTI symptoms, lab results, and demographics.
- **Machine Learning Models:** Training predictive models using Random Forest and XGBoost.
- **Inference Pipeline:** Applying trained models to new data, generating predictions, and interpreting results.

## Features

- **Symptom Analysis:** Identifies key indicators and correlations from patient data.
- **Model Evaluation:** Comprehensive performance metrics, including ROC curves and feature importance.
- **User-Friendly Prediction:** Interactive input and confidence scoring for UTI likelihood.

## Setup and Installation

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Run the Jupyter notebooks in order:**
   - **`1_data_exploration.ipynb`:** Data exploration, symptom analysis, and key insights.
   - **`2_model_training.ipynb`:** Train and evaluate Random Forest and XGBoost models. Includes threshold optimization and feature analysis.
   - **`3_model_inference.ipynb`:** Load models, preprocess new data, and run predictions with confidence scoring.

4. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

## Usage

1. **Data Exploration:** Analyze UTI symptoms, lab results, and demographics to gain insights.
2. **Model Training:** Train models using provided scripts and evaluate performance.
3. **Inference:** Apply the trained model to user input or sample data for prediction.

## Important Notes

- This system is intended for preliminary screening only and should **not** replace professional medical advice.
- Consult a healthcare provider for accurate diagnosis and treatment.
- The model currently uses synthetic data. For production use, it must be retrained with real medical data.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure your code follows project standards.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
