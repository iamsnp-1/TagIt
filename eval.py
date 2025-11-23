# eval.py
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

MODEL_FILE = "models/baseline_pipe.joblib"
ID2LABEL = "models/id_to_label.joblib"
DATA_FILE = "data/preprocessed.csv"
OUT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

pipe = joblib.load(MODEL_FILE)
id_to_label = joblib.load(ID2LABEL)
df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=['label'])

label_map = joblib.load("models/label_map.joblib")
df['label_id'] = df['label'].map(label_map)
df = df.dropna(subset=['label_id'])
df['label_id'] = df['label_id'].astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label_id'])
y_true = X_test['label_id'].values
X = X_test

y_pred = pipe.predict(X)
print("Classification report:")
print(classification_report(y_true, y_pred, digits=4))

cm = confusion_matrix(y_true, y_pred)
labels_sorted = [id_to_label[i] for i in sorted(id_to_label.keys())]

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_sorted, yticklabels=labels_sorted)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
print(f"✅ Saved confusion matrix to {OUT_DIR}/confusion_matrix.png")
