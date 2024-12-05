from setuptools import setup, find_packages

setup(
    name='uti_diagnosis_system',
    version='1.0.0',
    author='Your Name',
    description='An AI-based system for preliminary UTI diagnosis.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'xgboost',
        'matplotlib',
        'seaborn',
        'jupyter',
        'spacy',
        'openpyxl',
        'ipykernel'
    ],
    entry_points={
        'console_scripts': [
            # Add command-line scripts here if applicable
        ],
    },
)
