# Breast Cancer Classification using Machine Learning

This project is a **binary classification** model that predicts whether a tumor is **malignant** or **benign** using the **Breast Cancer Wisconsin (Diagnostic) Dataset**. The goal of this project is to gain hands-on experience with the end-to-end machine learning pipeline — from loading data and preprocessing to training, evaluating, and improving models.

Deployment : -  https://breast-cancer-app-deployment.streamlit.app/ 

---

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Learnings](#learnings)
- [License](#license)


---

##  Motivation

This is a beginner-level project created to understand the basics of machine learning using a real-world medical dataset. Breast cancer diagnosis is a crucial application where machine learning can assist doctors in making faster and more accurate decisions.

Although it's a commonly used dataset, I focused on learning:
- How to preprocess real data
- How different classification models behave
- How to tune hyperparameters
- How to evaluate a model beyond just accuracy

---

## Dataset

- **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** Built-in dataset from `sklearn.datasets` → `load_breast_cancer()` OR CSV version from UCI Machine Learning Repository
- **Samples:** 569 patient records
- **Features:** 30 real-valued input features extracted from digitized images of fine needle aspirates (FNA) of breast masses
- **Target Variable (`diagnosis`):**
  - `'M'` → **Malignant** (cancerous)
  - `'B'` → **Benign** (non-cancerous)
- **Class Distribution:**
  - Malignant (`M`): 212 samples (~37.3%)
  - Benign (`B`): 357 samples (~62.7%)

---

### Feature Structure

Each record consists of:

- An `ID` (not used in ML model)
- A `diagnosis` label (M/B)
- 30 real-valued features grouped into 3 categories:

| Group           | Suffix       | Description                                  |
|----------------|--------------|----------------------------------------------|
| Mean           | `_mean`      | Mean value of each feature                   |
| Standard Error | `_se`        | Standard error (variation)                   |
| Worst Value    | `_worst`     | Worst/largest value observed                 |

---

### Key Feature Categories

| Category              | Description |
|-----------------------|-------------|
| **Radius**            | Distance from center to perimeter |
| **Texture**           | Standard deviation of gray-scale values |
| **Perimeter**         | Length of the perimeter boundary |
| **Area**              | Size of the nucleus |
| **Smoothness**        | Local variation in radius lengths |
| **Compactness**       | `(Perimeter² / Area) - 1.0` |
| **Concavity**         | Severity of concave portions |
| **Concave Points**    | Number of concave portions |
| **Symmetry**          | Symmetry of the cell shape |
| **Fractal Dimension** | Border complexity (self-similarity) |

---

### Dataset Format

- **Shape:** `(569 rows × 32 columns)` → includes `id`, `diagnosis`, and 30 features
- **Feature Data Type:** `float64`
- **Target Variable Type:** `object` (string: `M` or `B`)
- **No Missing Values:** Clean and ready-to-use


---

## Project Pipeline

1. **Data Loading**  
   - Dataset used: data.csv (Breast Cancer Wisconsin Diagnostic Dataset)
   - Loaded using pandas.read_csv()

2. **Exploratory Data Analysis (EDA)**  
   - Visualizations created using Seaborn and Matplotlib
   Explored:
   - Class distribution (Malignant vs Benign)
   - Feature correlations (Heatmap)
   - Feature importance and relationships

3. **Preprocessing**
   - Dropped irrelevant columns (id, Unnamed: 32)
   - Encoded target variable: diagnosis (M = 1, B = 0)
   - Applied Standard Scaling using StandardScaler
   - Performed Train/Test split (80% training, 20% testing)

4. **Model Training**
   - Logistic Regression  
   - K-Nearest Neighbors (KNN)  
   - Random Forest  
   - Support Vector Machine (SVM) with RBF Kernel

5. **Hyperparameter Tuning**  
   - Optimized models using GridSearchCV for best performance

6. **Model Evaluation**  
   - Accuracy Score
   - Precision & Recall
   - Confusion Matrix  
   - Classification Report (Precision, Recall, F1-score)  
   - ROC-AUC Curve
  

8. **Model Inference & Prediction**

- SVM (RBF Kernel) achieved the highest accuracy (98.25%) and perfect precision (100%), making it the most confident model in predicting malignant cases correctly without false positives.

- Logistic Regression closely followed with an impressive ROC-AUC score of 99.74%, indicating excellent ability to distinguish between classes.

- K-Nearest Neighbors, while slightly behind, still maintained strong overall performance and balance between precision and recall.

- Random Forest offered high generalization with a well-balanced F1 score and excellent ROC-AUC.

- Based on these results, SVM (RBF Kernel) was selected as the final deployed model for prediction, due to its superior accuracy, zero false positives, and robust performance across all metrics.


10. **Model Deployment using Steamlit**
    The final model was deployed using Streamlit, a lightweight Python framework for building interactive web apps.
    Deployment Highlights:
    - User can input feature values through a form.
    - Instant prediction on whether the tumor is Benign or Malignant.
    - Clean and responsive interface.

Live App: https://breast-cancer-app-deployment.streamlit.app/

---

## Models Used

| Model               | Accuracy   |
| ------------------- | ---------- | 
| Logistic Regression | **97.37%** | 
| K-Nearest Neighbors | **94.74%** |
| Random Forest       | **96.49%** | 
| SVM (RBF Kernel)    | **98.25%** | 


---
## Evaluation Metrics

- Accuracy
- Precision & Recall
- F1 Score
- Confusion Matrix
- ROC-AUC Score

| Model               | Accuracy   | Precision | Recall | F1 Score | ROC-AUC Score |
| ------------------- | ---------- | --------- | ------ | -------- | ------------- |
| Logistic Regression | **97.37%** | 97.62%    | 95.35% | 96.47%   | 99.74%        |
| K-Nearest Neighbors | **94.74%** | 93.02%    | 93.02% | 93.02%   | 98.20%        |
| Random Forest       | **96.49%** | 97.56%    | 93.02% | 95.24%   | 99.53%        |
| SVM (RBF Kernel)    | **98.25%** | 100.00%   | 95.35% | 97.62%   | 99.74%        |

These metrics ensure not just correct predictions but also focus on **minimizing false negatives**, which is important in medical applications.

---

## Installation

<pre> <code>
git clone https://github.com/your-username/breast-cancer-classification.git
  
cd breast-cancer-classification
  
pip install -r requirements.txt

</pre>  </code>

## How to Run
Clone the Repository

<pre> <code>
git clone https://github.com/yourusername/breast-cancer-classification.git

cd breast-cancer-classification
  
Install Dependencies

  </pre>  </code>

Make sure you have Python 3.7+ installed.
Then run:

<pre> <code>
pip install -r requirements.txt
Run the Streamlit App
</pre>  </code>

<pre> <code>
streamlit run app.py
</pre>  </code>
  
Interact with the App
A browser window will open where you can upload data and get predictions for whether a tumor is benign or malignant.

## Results
| Model               | Accuracy   | Precision | Recall | F1 Score   | ROC-AUC    |
| ------------------- | ---------- | --------- | ------ | ---------- | ---------- |
| Logistic Regression | 97.37%     | 97.62%    | 95.35% | 96.47%     | 0.9974     |
| K-Nearest Neighbors | 94.74%     | 93.02%    | 93.02% | 93.02%     | 0.9820     |
| Random Forest       | 96.49%     | 97.56%    | 93.02% | 95.24%     | 0.9953     |
| SVM (RBF Kernel)    | **98.25%** | **100%**  | 95.35% | **97.62%** | **0.9974** |


## Future Improvements

- Integrate a proper frontend UI with additional visual explanations (e.g., SHAP values).
- Add support for live data input from hospitals via API.
- Train ensemble or stacking models for even better accuracy.
- Implement email/Slack alerts for potential malignancy predictions.
- Extend dataset and include more patient demographics.

## Learnings

- Understood the end-to-end ML pipeline: loading, preprocessing, training, and evaluation.
- Learned the importance of choosing the right evaluation metric for medical datasets.
- Got hands-on experience with multiple ML models and their trade-offs.
- Learned how to deploy ML models using Streamlit for real-time interaction.
- Improved Python and scikit-learn proficiency, and streamlined model comparison.

## License
This project is licensed under the MIT License.
You're free to use, modify, and distribute it for both commercial and non-commercial purposes.
