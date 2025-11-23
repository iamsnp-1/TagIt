import pandas as pd
import joblib
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "models/baseline_pipe.joblib"
ID2LABEL = "models/id_to_label.joblib"
INPUT = "data/preprocessed.csv"
OUT = "data/predictions.csv"

pipe = joblib.load(MODEL_FILE)
id_to_label = joblib.load(ID2LABEL)

df = pd.read_csv(INPUT)
X = df

probs = pipe.predict_proba(X)
pred_ids = np.argmax(probs, axis=1)
confidences = np.max(probs, axis=1)

pred_labels = [id_to_label.get(int(pid), str(pid)) for pid in pred_ids]


pre = pipe.named_steps['pre']
clf = pipe.named_steps['clf']
tfidf = pre.named_transformers_['text']
try:
    feat_names = tfidf.get_feature_names_out()
except:
    feat_names = tfidf.get_feature_names()

text_feats = pre.transformers_[0][1].transform(df['merchant_clean']) if False else None

text_X = pre.named_transformers_['text'].transform(df['merchant_clean'])
coefs = clf.coef_  
n_text = text_X.shape[1]
top_token_list = []
for i in range(len(df)):
    row = text_X[i].toarray().ravel()
    cls = pred_ids[i]
    token_scores = row * coefs[cls, :n_text]
    top_idx = np.argsort(-token_scores)[:5]
    top_tokens = [(feat_names[idx], float(token_scores[idx])) for idx in top_idx if token_scores[idx] > 0]
    top_token_list.append(top_tokens)

out_df = df.copy()
out_df['pred_id'] = pred_ids
out_df['pred_label'] = pred_labels
out_df['confidence'] = confidences
out_df['top_tokens'] = [str(t) for t in top_token_list]

out_df.to_csv(OUT, index=False)
print(f"✅ Predictions saved to {OUT}")
