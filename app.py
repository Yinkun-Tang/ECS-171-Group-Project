#!/usr/bin/env python
# coding: utf-8


# Comparison
# Receiver Operating Characterisc curve


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import streamlit as st

# Load the data
dataset = pd.read_csv('riceclass.csv')
X = dataset.drop(['Class', 'id'], axis=1)
y = dataset['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Standardize/Scale the data
scaler_lr = MinMaxScaler(feature_range=(0, 1))
X_train_lr_scaled = scaler_lr.fit_transform(X_train)
X_test_lr_scaled = scaler_lr.transform(X_test)

scaler_knn = MinMaxScaler(feature_range=(0, 1))
X_train_knn_scaled = scaler_knn.fit_transform(X_train)
X_test_knn_scaled = scaler_knn.transform(X_test)

scaler_nn = MinMaxScaler(feature_range=(0, 1))
X_train_nn_scaled = scaler_nn.fit_transform(X_train)
X_test_nn_scaled = scaler_nn.transform(X_test)


# Encode labels for roc_curve
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_lr_scaled, y_train)
logreg_probs = logreg.predict_proba(X_test_lr_scaled)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test_encoded, logreg_probs)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=65)  # Assuming 65 is the best k value from your tuning
knn.fit(X_train_knn_scaled, y_train)
knn_probs = knn.predict_proba(X_test_knn_scaled)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test_encoded, knn_probs)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Neural Network
mlp = MLPClassifier(solver = 'adam', random_state = 42, activation = 'logistic', learning_rate_init = 0.01, batch_size = 300, hidden_layer_sizes = (12, 24, 48,), max_iter = 1000)
mlp.fit(X_train_nn_scaled, y_train)
nn_probs = mlp.predict_proba(X_test_nn_scaled)[:, 1]
fpr_nn, tpr_nn, _ = roc_curve(y_test_encoded, nn_probs)
roc_auc_nn = auc(fpr_nn, tpr_nn)



st.title('Which species of rice is it? :rice:')
st.markdown('Enter values for the variables below and see what Logistic Regression, K-Nearest Neighbors, and a Neural Network predict.')

area = st.slider('Area', 0,20000)
perimeter = st.text_input('Perimeter')
major_axis = st.text_input('Major Axis Length')
minor_axis = st.text_input('Minor Axis Length')
eccentricity = st.text_input('Eccentricity')
convex_area = st.slider('Convex Area', 0,20000)
extent = st.text_input('Extent')

obs = np.array([area, perimeter, major_axis, minor_axis, eccentricity, convex_area, extent])
obs = pd.DataFrame([obs])

def gui_predict():
    st.success('Logistic Regression predicted ' + logreg.predict(obs) + ', K-Nearest Neighbors predicted ' + knn.predict(obs) + ', and Neural Network predicted ' + mlp.predict(obs))

st.button('Predict', on_click=gui_predict)
