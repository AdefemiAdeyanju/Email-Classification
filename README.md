# Email-Classification

Predicting e-mail messages genuinity using machine learning models.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Source and Preprocessing](#data-source-and-preprocessing)
- [Model Details](#model-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Installation
Install the required packages from the `requirements.txt` file:

## Project Structure
The project is organized as follows:

- `Spam.csv/`: Contains raw data files.
- `Email Classification.ipynb/`: Holds Jupyter notebooks used for data analysis and exploration.
- `gbc.pkl/`: Stores trained machine learning models for type prediction.
- `app.py`: A Python script to make predictions using trained models.
- `vectorizer.pkl`: Tfidfvectorizer model.
- `nltk.txt`: nltk libraries used.

## Data Source and Preprocessing
- The dataset is obtained from [Kaggle](https://www.kaggle.com/dataset).
- Preprocessed data by handling missing values and encoding categorical features.

## Model Details
- Trained a Random Forest, Gradient Boosting and MultinomialNB Classifier.

## Evaluation Metrics
- Evaluated models using Accuracy Score, Precision Score, and Classification report.

## Acknowledgments
- Used the `scikit-learn` library for machine learning models.

## Contact
For questions or feedback, contact me at adefemiadeyanju101@hotmail.com.
