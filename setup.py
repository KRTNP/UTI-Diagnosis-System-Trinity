from setuptools import setup, find_packages

setup(
    name='uti_diagnosis_system',
    version='1.0.0',
    author='Nattaphon Honghin',
    author_email='nattaphon.honghin@gmail.com',
    description='An AI-based system for preliminary UTI diagnosis.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KRTNP/UTI-Prediction-Trinity',
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
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.10',
    include_package_data=True,
    keywords='UTI diagnosis, machine learning, healthcare, AI'
)
