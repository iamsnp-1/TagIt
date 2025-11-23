# predict_transformer.py
"""
Load models/transformer_best.pt + tokenizer + metadata and run inference on data/preprocessed.csv
Outputs: data/predictions_transformer.csv with pred_id, pred_label, confidence
"""

import os
import joblib
import pandas as pd
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch.nn as nn
from train_transformer import TransformerTabularModel, NUM_COLS  # reuse definitions
from sklearn.preprocessing import StandardScaler, LabelEncoder

MODEL_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE = "data/preprocessed.csv"
OUT_FILE = "data/predictions_transformer.csv"

meta = joblib.load(os.path.join(MODEL_DIR, "transformer_metadata.joblib"))
label_map = joblib.load(os.path.join(MODEL_DIR, "transformer_label_map.joblib"))
id_to_label = joblib.load(os.path.join(MODEL_DIR, "transformer_id_to_label.joblib"))

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
scaler = joblib.load(meta['scaler'])
merchant_encoder = joblib.load(meta['merchant_encoder'])

n_classes = len(label_map)
merchant_vocab = len(merchant_encoder.classes_)
model = TransformerTabularModel(meta['model_name'], num_numeric=len(NUM_COLS), merchant_vocab_size=merchant_vocab, merchant_emb_dim=32, hidden_dim=256, n_classes=n_classes)
ckpt = torch.load(os.path.join(MODEL_DIR, "transformer_best.pt"), map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.to(DEVICE)
model.eval()

df = pd.read_csv(DATA_FILE)
df_proc = df.copy()

def prepare_row(row):
    enc = tokenizer(str(row['merchant_clean']), truncation=True, padding='max_length', max_length=64, return_tensors='pt')
    try:
        merch_id = int(merchant_encoder.transform([str(row['merchant_token'])])[0])
    except Exception:
        merch_id = 0
    nums = scaler.transform([row[NUM_COLS].fillna(0.0).values.astype(float)])[0]
    item = {
        'input_ids': enc['input_ids'].squeeze(0),
        'attention_mask': enc['attention_mask'].squeeze(0),
        'merchant_id': torch.tensor(merch_id, dtype=torch.long),
        'nums': torch.tensor(nums, dtype=torch.float32)
    }
    return item

pred_ids = []
pred_probs = []
pred_labels = []

with torch.no_grad():
    for _, row in df_proc.iterrows():
        item = prepare_row(row)
        input_ids = item['input_ids'].unsqueeze(0).to(DEVICE)
        attention_mask = item['attention_mask'].unsqueeze(0).to(DEVICE)
        merchant_id = item['merchant_id'].unsqueeze(0).to(DEVICE)
        nums = item['nums'].unsqueeze(0).to(DEVICE)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, merchant_id=merchant_id, nums=nums)
        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
        pid = int(np.argmax(probs))
        pred_ids.append(pid)
        pred_probs.append(float(np.max(probs)))
        pred_labels.append(id_to_label.get(pid, str(pid)))

out = df_proc.copy()
out['pred_id'] = pred_ids
out['pred_label'] = pred_labels
out['confidence'] = pred_probs
out.to_csv(OUT_FILE, index=False)
print(f"✅ Saved transformer predictions to {OUT_FILE}")
