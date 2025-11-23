import pandas as pd
import numpy as np
import joblib
import sklearn
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from collections import defaultdict
import random

RNG = 42
random.seed(RNG)
np.random.seed(RNG)

DATA_FILE = "data/preprocessed.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=['label']).reset_index(drop=True)

with open("taxonomy.yaml", 'r', encoding='utf-8') as f:
    tax = yaml.safe_load(f)

label_to_id = {}
id_to_label = {}
for c in tax['categories']:
    label_to_id[c['name']] = c['id']
    id_to_label[c['id']] = c['name']

unique_labels = sorted(df['label'].unique())
label_map = {}
next_id = max(id_to_label.keys())+1 if id_to_label else 0
for lab in unique_labels:
    if lab in label_to_id:
        label_map[lab] = label_to_id[lab]
    else:
        label_map[lab] = next_id
        id_to_label[next_id] = lab
        next_id += 1

df['label_id'] = df['label'].map(label_map)

X = df
y = df['label_id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RNG, stratify=y)

TEXT_COL = 'merchant_clean'
CAT_COLS = ['merchant_token','amount_bucket','weekday']
NUM_COLS = ['hour','is_weekend','merchant_user_count','amount']

tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=7000)
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

preprocessor = ColumnTransformer([
    ('text', tfidf, TEXT_COL),
    ('cat', ohe, CAT_COLS),
    ('num', scaler, NUM_COLS)
], remainder='drop', sparse_threshold=0.0)

clf = LogisticRegression(max_iter=400, class_weight='balanced', n_jobs=-1)
pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', clf)
])

print("Training baseline model...")
pipe.fit(X_train, y_train)

joblib.dump(pipe, f"{MODEL_DIR}/baseline_pipe.joblib")
joblib.dump(label_map, f"{MODEL_DIR}/label_map.joblib")
joblib.dump(id_to_label, f"{MODEL_DIR}/id_to_label.joblib")
print("Model saved to models/")

y_pred = pipe.predict(X_test)
print("\nClassification report on test set:")
print(classification_report(y_test, y_pred, digits=4))
