import streamlit as st
import pandas as pd
import joblib
import torch
import numpy as np
import os
from transformers import DistilBertTokenizerFast
from train_transformer import TransformerTabularModel, NUM_COLS

st.set_page_config(page_title="💸 AI Transaction Categorizer", page_icon="💰", layout="centered")
st.title("💸 AI Transaction Categorizer")

st.sidebar.header("🧠 Model Info")
st.sidebar.write("Hybrid Categorization System")
st.sidebar.write("Baseline: TF-IDF + Logistic Regression")
st.sidebar.write("Transformer: DistilBERT + Tabular Features")
st.sidebar.markdown("---")
st.sidebar.write("Runs Locally — No API Calls")
st.sidebar.write("Adjustable Confidence Threshold")

BASE_MODEL_PATH = "models/baseline_pipe.joblib"
BASE_LABEL_PATH = "models/id_to_label.joblib"
if not os.path.exists(BASE_MODEL_PATH):
    st.error("❌ Baseline model not found. Please run `train_baseline.py` first.")
    st.stop()

pipe = joblib.load(BASE_MODEL_PATH)
base_id2label = joblib.load(BASE_LABEL_PATH)

TRANS_META_PATH = "models/transformer_metadata.joblib"
TRANS_CKPT_PATH = "models/transformer_best.pt"

transformer_loaded = os.path.exists(TRANS_META_PATH) and os.path.exists(TRANS_CKPT_PATH)
if transformer_loaded:
    meta = joblib.load(TRANS_META_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    label_map = joblib.load(meta["label_map_file"])
    id_to_label = joblib.load(meta["id_to_label_file"])
    scaler = joblib.load(meta["scaler"])
    merchant_encoder = joblib.load(meta["merchant_encoder"])
    n_classes = len(label_map)
    merchant_vocab = len(merchant_encoder.classes_)
    model = TransformerTabularModel(
        meta["model_name"], num_numeric=len(NUM_COLS),
        merchant_vocab_size=merchant_vocab, merchant_emb_dim=32,
        hidden_dim=256, n_classes=n_classes
    )
    ckpt = torch.load(TRANS_CKPT_PATH, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
else:
    st.sidebar.warning("Transformer model not found. Only baseline predictions available.")
    device = torch.device("cpu")

def transformer_predict(row):
    enc = tokenizer(str(row["merchant_clean"]),
                    truncation=True, padding="max_length",
                    max_length=64, return_tensors="pt")
    try:
        merch_id = int(merchant_encoder.transform([str(row["merchant_token"])])[0])
    except Exception:
        merch_id = 0
    nums = scaler.transform([row[NUM_COLS].fillna(0.0).values.astype(float)])[0]
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    merch = torch.tensor([merch_id], dtype=torch.long).to(device)
    nums = torch.tensor([nums], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn, merchant_id=merch, nums=nums)
        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
        pid = int(np.argmax(probs))
        return id_to_label.get(pid, "Unknown"), float(np.max(probs))

def hybrid_predict(df, conf_threshold=0.7):
    base_preds = pipe.predict(df)
    base_probs = pipe.predict_proba(df)
    base_conf = base_probs.max(axis=1)

    final_labels = []
    final_conf = []
    for i, row in df.iterrows():
        if base_conf[i] >= conf_threshold or not transformer_loaded:
            label = base_id2label.get(base_preds[i], "Unknown")
            conf = base_conf[i]
        else:
            label, conf = transformer_predict(row)
        final_labels.append(label)
        final_conf.append(conf)

    return final_labels, final_conf

conf_threshold = st.sidebar.slider("Confidence Threshold (Baseline → Transformer)", 0.5, 0.95, 0.7, 0.05)

st.subheader("🔎 Single Transaction Prediction")
merchant = st.text_input("Transaction description:", placeholder="e.g. Starbucks Coffee Pune")
amount = st.number_input("Transaction amount (₹):", min_value=0.0, max_value=100000.0, value=250.0, step=10.0)

if st.button("🔮 Predict Category"):
    if not merchant.strip():
        st.warning("⚠️ Please enter a valid transaction description.")
        st.stop()
    df = pd.DataFrame([{
        "merchant_clean": merchant.lower(),
        "merchant_token": merchant.split()[0].lower(),
        "amount": amount,
        "amount_bucket": "500-1999" if amount < 2000 else "2000+",
        "hour": 12, "weekday": 2, "is_weekend": 0,
        "merchant_user_count": 1
    }])
    labels, confs = hybrid_predict(df, conf_threshold)
    st.success(f"🧾 Predicted Category: {labels[0]}")
    st.info(f"📈 Confidence: `{confs[0]:.2f}`")
    if confs[0] < 0.6:
        st.warning("⚠️ Low confidence — consider manual review.")

st.markdown("---")
st.subheader("📂 Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload a CSV of transactions", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_cols = ["merchant_clean", "merchant_token", "amount", "hour", "weekday", "is_weekend", "merchant_user_count", "amount_bucket"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Missing columns. Required: {required_cols}")
    else:
        with st.spinner("Predicting categories..."):
            labels, confs = hybrid_predict(df, conf_threshold)
            df["pred_label"] = labels
            df["confidence"] = confs
            st.success("Predictions complete.")
            st.dataframe(df[["merchant_clean", "pred_label", "confidence"]])
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", data=csv, file_name="predictions_hybrid.csv", mime="text/csv")

st.markdown("---")
st.caption("Developed with ❤️ | Hybrid AI Transaction Categorization System v2.0")
