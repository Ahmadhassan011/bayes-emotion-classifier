# Social Media Emotion Prediction using Bayesian Classification

## 1. Project Overview

This project implements a custom Naive Bayes classifier for the prediction of dominant human emotions from digital behavioral data. This project was completed as a core component of a Probability and Statistics course, serving as a comprehensive demonstration of applying the fundamental principles of Bayes' Theorem—specifically, the calculation of prior and conditional probabilities—to build a predictive machine learning model. The primary objective is to illustrate foundational statistical concepts underpinning Bayesian classification in a practical and accessible manner.

## 2. Project Structure and Key Components

The repository is structured to provide a clear and organized overview of the project:

* `data.csv`: The dataset used for training and evaluating the model.
* `probability_and_statistics_project.ipynb`: A Jupyter Notebook detailing the entire machine learning pipeline, from data loading and preprocessing to model training, evaluation, and prediction. This notebook is suitable for interactive exploration and understanding the step-by-step process.
* `Project_Report_PsT_.pdf`: A detailed scientific report documenting the project, including the problem statement, purpose, data analysis, Bayesian fundamentals, results, and conclusion.
* `my_model.joblib`: The trained Bayes classifier model saved for future use.
* `le_gender.joblib`, `le_platform.joblib`, `le_emotion.joblib`: Saved LabelEncoder objects used for transforming categorical features and target variable.

The core of the implementation is the `SimpleBayesClassifier` class, which custom-builds the Naive Bayes algorithm.

## 3. Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Ahmadhassan011/bayes-emotion-classifier.git
    cd bayes-emotion-classifier
    ```

2.  **Install Dependencies:**
    Ensure you have Python installed (preferably Python 3.x). Install the required libraries using pip:

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    ```

## 4. Dataset Details and Preprocessing Steps

The project utilizes a custom-collected dataset comprising over 1000 samples of user-centric digital interaction data. This dataset was meticulously prepared to ensure its quality and suitability for emotion prediction.

**Features:**
The dataset includes the following features:
* `Age`
* `Gender`
* `Platform`
* `Daily_Usage_Time(minutes)`
* `Posts_Per_Day`
* `Likes_Received_Per_Day`
* `Comments_Received_Per_Day`
* `Messages_Sent_Per_Day`

**Target Variable:**
The target variable is `Dominant_Emotion`.

**Preprocessing Steps:**
The `data.csv` file underwent the following preprocessing steps:
1.  **Missing Value Handling**: Completely empty rows were removed. Any remaining missing values were filled using the forward fill method (`data.ffill(inplace=True)`).
2.  **Gender Data Cleaning**: Only valid entries ('Male', 'Female', 'Non-binary') were retained.
3.  **Age Data Cleaning**: Age values were converted to a numeric type, with non-convertible entries coerced to `NaN`. Data was then filtered to include ages within a realistic range of 10 to 100.
4.  **Index Reset**: The DataFrame index was reset after cleaning operations.
5.  **Irrelevant Column Removal**: The `User_ID` column was dropped as it is not a predictive feature.
6.  **Label Encoding**: Categorical features (`Gender`, `Platform`) and the target variable (`Dominant_Emotion`) were label encoded using `LabelEncoder` from `sklearn.preprocessing`.
7.  **Train/Test Split**: The dataset was split into an 80% training set and a 20% testing set (`test_size=0.2`, `random_state=42`, `stratify=y`) to ensure proper model evaluation.

## 5. Usage

To use the trained machine learning model for predictions, you can run the `probability_and_statistics_project.ipynb` Jupyter Notebook. The model and encoders are saved as `.joblib` files, which can be loaded for making new predictions. Refer to the `probability_and_statistics_project.py` file for detailed examples of how to load the model, prepare new input data, and obtain predictions.

## 6. Model Performance and Key Metrics

The custom Naive Bayes classifier achieved a notable accuracy of **85.4%** on the test dataset.

The model's performance was further evaluated using standard classification metrics:

* **Accuracy**: Overall correctness of predictions.
* **Precision**: The proportion of true positive predictions among all positive predictions for each class.
* **Recall**: The proportion of true positive predictions among all actual positives for each class.
* **F1-Score**: The harmonic mean of precision and recall, providing a balanced measure for each class.

A detailed **classification report** and **confusion matrix** are generated during the model evaluation phase in the provided Jupyter notebook, offering deeper insights into the model's predictive capabilities across different emotion categories.

## 6. Contact

For any questions or inquiries, please contact:

* Ahmad Hassan
* Email: [ahmadhassan6531@gmail.com]
