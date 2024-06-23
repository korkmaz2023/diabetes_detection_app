
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

import shap
import pickle
import streamlit as st
print('All libraries are imported')

data = pd.read_csv("dataset/diabetes.csv")
data.head()

data.info()

data.shape
data.describe()
data.duplicated().sum()

data['Outcome'].value_counts()

df = data.copy()

# Exploratory Data Analysis
# univariate
# bivariate
# Scatter plots
# Correlation/ heat map

# Univariate numerical
df.hist()
plt.tight_layout()
plt.show()

repeat = 1
for col in df.select_dtypes(exclude = 'O').columns:
    plt.subplot(3,3, repeat)
    sns.boxplot(df[col])
    plt.title(col)
    repeat +=1
plt.tight_layout()
plt.show()

sns.pairplot(df)
plt.show()

repeater = 1
for col in df.select_dtypes(exclude = 'O').columns:
    plt.subplot(3,3,repeater)
    df.groupby('Outcome')[col].mean().plot(kind = 'bar')
    plt.ylabel(col)
    repeater += 1
plt.tight_layout()
plt.show()

sns.pairplot(df, hue = "Outcome")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot = True)
plt.show()

X = df.drop('Outcome', axis = 1)
y = df["Outcome"]
# solve data imbalance

sm = SMOTE()
X, y = sm.fit_resample(X,y)
y.value_counts()

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                   test_size= 0.2,
                                                   random_state= 101,
                                                   stratify= y)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

def print_metrics(y_test, y_pred, model_name):
    print("Metrics for model", model_name)
    print("Accuracy Score :", accuracy_score(y_test, y_pred))
    print("Recall :", recall_score(y_test, y_pred))
    print("Precision :", precision_score(y_test, y_pred))
    print("ROC Score :", roc_auc_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
print_metrics(y_test, y_pred_knn, knn)

clfs = {'Logreg': LogisticRegression(),
        'NaiveBayes': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'GradientBoost': GradientBoostingClassifier(),
        'SVM': SVC()
        }

models_report = pd.DataFrame(columns=["Model", "Accuracy", "Recall", "Precision", "F1Score", "ROC"])

for clf_name, clf in clfs.items():
    clf.fit(X_train, y_train)
    print("Fitting the model....", clf_name)
    y_pred = clf.predict(X_test)
    t = pd.DataFrame([{
        "Model": clf_name, "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred), "Precision": precision_score(y_test, y_pred),
        "F1Score": f1_score(y_test, y_pred), "ROC": roc_auc_score(y_test, y_pred)
    }])
    models_report = pd.concat([models_report, t], ignore_index=True)

    models_report.sort_values(by='F1Score', ascending=False)

param_dist={'n_estimators':range(100,1000,100),
           'max_depth':range(10,100,5),
           'min_samples_leaf':range(1,10,1),
           'min_samples_split':range(2,20,2),
           'max_features':["log2",'sqrt'],
           'criterion':['entropy','gini']}
n_folds=10

rfc=RandomForestClassifier()

cv_rfc=RandomizedSearchCV(estimator=rfc,
                          param_distributions=param_dist,
                          n_jobs=-1,
                          verbose=3,
                          cv=n_folds,
                          scoring="f1",
                          n_iter=10)

cv_rfc.fit(X_train, y_train)

import pickle

def is_pickleable(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError):
        return False

print(is_pickleable(cv_rfc))




rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print_metrics(y_test, y_pred, rfc)


# Deploy the model in Streamlit
#pickle the best model

model = open('rfc.pickle', 'wb')
pickle.dump(rfc, model)
model.close()
#load the pickled model
model = open('rfc.pickle', 'rb')
clf = pickle.load(model)
model.close()
#Checking if the model is able to predict with ransom input
data =[[3,100,80,23,1901,33,2.6,44]]
clf.predict(data)[0]