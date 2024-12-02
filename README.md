# UTI Diagnosis System

An AI-based system for preliminary diagnosis of Urinary Tract Infections (UTIs) based on user-reported symptoms.

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Run the notebooks in order:
   - `1_data_exploration.ipynb`: Data analysis and preprocessing
   - `2_model_training.ipynb`: Model training and evaluation
   - `3_model_inference.ipynb`: Making predictions
```

## Important Notes

- This system is for preliminary screening only and should not replace professional medical advice.
- Always consult a healthcare provider for proper diagnosis and treatment.
- The model is trained on synthetic data and should be retrained with real medical data for production use.

## License

MIT License
