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

# Load the pre-trained model
with open('rfc.pickle', 'rb') as model_file:
    clf = pickle.load(model_file)

# Display documented model performance metrics
st.sidebar.subheader('Documented Model Performance Metrics')
st.sidebar.write("**Metrics for model RandomForestClassifier()**")
st.sidebar.write("**Accuracy Score**: 0.855")
st.sidebar.write("**Recall**: 0.88")
st.sidebar.write("**Precision**: 0.838")
st.sidebar.write("**ROC Score**: 0.855")
st.sidebar.write("**F1 Score**: 0.859")

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


