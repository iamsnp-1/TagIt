
import pandas as pd
import joblib
import torch
import numpy as np
from transformers import DistilBertTokenizerFast
from train_transformer import TransformerTabularModel, NUM_COLS


BASE_MODEL_PATH = "models/baseline_pipe.joblib"
BASE_LABEL_PATH = "models/id_to_label.joblib"
TRANS_META_PATH = "models/transformer_metadata.joblib"
TRANS_CKPT_PATH = "models/transformer_best.pt"

pipe = joblib.load(BASE_MODEL_PATH)
base_id2label = joblib.load(BASE_LABEL_PATH)

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


def prepare_row(row):
    enc = tokenizer(str(row["merchant_clean"]), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
    try:
        merch_id = int(merchant_encoder.transform([str(row["merchant_token"])])[0])
    except:
        merch_id = 0
    nums = scaler.transform([row[NUM_COLS].fillna(0.0).values.astype(float)])[0]
    item = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "merchant_id": torch.tensor([merch_id], dtype=torch.long).to(device),
        "nums": torch.tensor([nums], dtype=torch.float32).to(device),
    }
    return item

def hybrid_predict(df, conf_threshold=0.7):
    base_preds = pipe.predict(df)
    base_probs = pipe.predict_proba(df)
    base_conf = base_probs.max(axis=1)

    final_labels = []
    final_conf = []

    for i, row in df.iterrows():
        if base_conf[i] >= conf_threshold:
            label = base_id2label.get(base_preds[i], "Unknown")
            conf = base_conf[i]
        else:
            with torch.no_grad():
                item = prepare_row(row)
                logits = model(**item)
                probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
                pid = int(np.argmax(probs))
                label = id_to_label.get(pid, "Unknown")
                conf = float(np.max(probs))
        final_labels.append(label)
        final_conf.append(conf)

    return pd.DataFrame({
        "merchant": df["merchant_clean"],
        "pred_label": final_labels,
        "confidence": final_conf
    })


if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed.csv").head(20)
    preds = hybrid_predict(df, conf_threshold=0.7)
    print(preds)
    preds.to_csv("data/predictions_hybrid.csv", index=False)
    print("✅ Saved hybrid predictions to data/predictions_hybrid.csv")
