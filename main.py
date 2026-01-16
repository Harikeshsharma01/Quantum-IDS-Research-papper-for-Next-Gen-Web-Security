
# Quantum-Based Intrusion Detection System (Q-IDS)
# Author: Harikesh Mahendra Sharma

import pandas as pd
import numpy as np
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("dataset.csv")
df = df.dropna()

label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

X = df.drop('Label', axis=1).values
y = df['Label'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=4)
X_reduced = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(x, weights):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def quantum_feature_map(X, weights):
    return np.array([quantum_circuit(x, weights) for x in X])

def predict(X, weights):
    q_features = quantum_feature_map(X, weights)
    scores = np.mean(q_features, axis=1)
    return (scores > 0).astype(int)

weights = np.random.randn(n_qubits)
y_pred = predict(X_test, weights)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
