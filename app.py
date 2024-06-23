import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

st.title('Medical Diagnostic Web App ⚕️')
st.subheader("Does the patient have diabetes?")

# Load dataset
df = pd.read_csv('dataset/diabetes.csv')

# Sidebar options
if st.sidebar.checkbox('View data', False):
    st.write(df)

if st.sidebar.checkbox('View Distribution', False):
    # Create separate subplots for each histogram
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = axs.flatten()
    for i, col in enumerate(df.columns[:-1]):  # Skip the 'Outcome' column
        sns.histplot(df[col], kde=True, ax=axs[i])
        axs[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

# Add heatmap option
if st.sidebar.checkbox('View Heatmap', False):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, ax=ax)
    st.pyplot(fig)

# Define the performance data dictionary
performance_data = {
    "Model": ["RandomForest", "DecisionTree", "GradientBoost", "NaiveBayes", "Logreg", "SVM"],
    "Accuracy": [0.865, 0.825, 0.800, 0.745, 0.730, 0.715],
    "Recall": [0.88, 0.81, 0.79, 0.69, 0.72, 0.70],
    "Precision": [0.854369, 0.835052, 0.806122, 0.775281, 0.734694, 0.721649],
    "F1Score": [0.866995, 0.822335, 0.797980, 0.730159, 0.727273, 0.710660],
    "ROC": [0.865, 0.825, 0.800, 0.745, 0.730, 0.715]
}

performance_df = pd.DataFrame(performance_data)

# Display model performance metrics table and hyperparameters
if st.sidebar.checkbox('Model Performance Metrics', False):
    st.subheader('Model Performance Metrics')
    st.write(performance_df)

    # Checkbox for hyperparameter optimization
if st.sidebar.checkbox('Hyperparameter Optimization', False):
    # Display hyperparameter optimization details
    param_dist = {
        'n_estimators': range(100, 1000, 100),
        'max_depth': range(10, 100, 5),
        'min_samples_leaf': range(1, 10, 1),
        'min_samples_split': range(2, 20, 2),
        'max_features': ["log2", 'sqrt'],
        'criterion': ['entropy', 'gini']
    }
    n_folds = 10

    st.subheader('Hyperparameter Optimization Details')
    st.markdown(f"""
    - **n_estimators**: {list(param_dist['n_estimators'])}
    - **max_depth**: {list(param_dist['max_depth'])}
    - **min_samples_leaf**: {list(param_dist['min_samples_leaf'])}
    - **min_samples_split**: {list(param_dist['min_samples_split'])}
    - **max_features**: {param_dist['max_features']}
    - **criterion**: {param_dist['criterion']}
    - **Number of folds for cross-validation**: {n_folds}
    """)

# Load the pre-trained model
with open('rfc.pickle', 'rb') as model_file:
    clf = pickle.load(model_file)

# Documented model performance metrics
documented_performance_data = {
    "Metric": ["Accuracy Score", "Recall", "Precision", "ROC Score", "F1 Score"],
    "Value": [0.855, 0.88, 0.838, 0.855, 0.859]
}
documented_performance_df = pd.DataFrame(documented_performance_data)

st.sidebar.subheader('Documented Model Performance Metrics')
st.sidebar.table(documented_performance_df)

# Display Confusion Matrix
st.sidebar.subheader("Confusion Matrix")
conf_matrix = np.array([[83, 17], [12, 88]])
conf_fig = ff.create_annotated_heatmap(
    conf_matrix,
    x=['Predicted Non-diabetic', 'Predicted Diabetic'],
    y=['Actual Non-diabetic', 'Actual Diabetic'],
    annotation_text=conf_matrix,
    colorscale='Viridis'
)
st.sidebar.plotly_chart(conf_fig)

# Display Classification Report without support column
st.sidebar.subheader("Classification Report")
report_data = {
    "precision": [0.87, 0.84, 0.86, 0.86],
    "recall": [0.83, 0.88, 0.85, 0.85],
    "f1-score": [0.85, 0.86, 0.85, 0.85]
}
report_index = ["Non-diabetic", "Diabetic", "macro avg", "weighted avg"]
report_df = pd.DataFrame(report_data, index=report_index)

st.sidebar.table(report_df)

# User input fields
pregs = st.number_input('Pregnancies', 0, 20, 0)
plas = st.slider('Glucose', 40, 200, 40)
pres = st.slider('BloodPressure', 20, 150, 20)
skin = st.slider('SkinThickness', 7, 99, 7)
insulin = st.slider('Insulin', 14, 850, 14)
bmi = st.slider('BMI', 18, 70, 18)
dpf = st.slider('DiabetesPedigreeFunction', 0.05, 2.50, 0.05)
age = st.slider('Age', 21, 90, 21)

# Prepare the input data for prediction
input_data = np.array([[pregs, plas, pres, skin, insulin, bmi, dpf, age]])

# Predict and display results
if st.button('Predict'):
    prediction = clf.predict(input_data)[0]
    if prediction == 0:
        st.subheader("Non diabetic")
    else:
        st.subheader("Diabetic")
