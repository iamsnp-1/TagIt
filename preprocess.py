import pandas as pd
import numpy as np
import re
import os

IN_FILE = "data/transactions.csv"
OUT_FILE = "data/preprocessed.csv"

def normalize_text(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r'https?://\S+',' ', s)
    s = re.sub(r'[\d,]+(\.\d+)?', ' <AMOUNT> ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def amount_bucket(a):
    try:
        a = float(a)
    except:
        return "unknown"
    if a < 20: return "<20"
    if a < 100: return "20-99"
    if a < 500: return "100-499"
    if a < 2000: return "500-1999"
    return "2000+"

os.makedirs("data", exist_ok=True)
df = pd.read_csv(IN_FILE, parse_dates=["datetime"])

df['merchant_raw'] = df['merchant'].astype(str)
df['merchant_clean'] = df['merchant_raw'].apply(normalize_text)

df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday
df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)
df['month'] = df['datetime'].dt.month

df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
df['amount_bucket'] = df['amount'].apply(amount_bucket)

df['merchant_token'] = df['merchant_clean'].str.split().str[0].fillna('')

df['merchant_user_count'] = df.groupby(['user_id','merchant_token'])['id'].transform('count')

out = df[['id','user_id','datetime','merchant','merchant_clean','merchant_token','amount','amount_bucket','hour','weekday','is_weekend','month','merchant_user_count','label']]
out.to_csv(OUT_FILE, index=False)
print(f"✅ Preprocessed saved to {OUT_FILE}")
