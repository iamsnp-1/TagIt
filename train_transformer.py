import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
DATA_FILE = "data/preprocessed.csv"
MODEL_DIR = "models"
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

NUM_COLS = ["hour", "is_weekend", "merchant_user_count", "amount"]

class TxnDataset(Dataset):
    def __init__(self, df, tokenizer, le, scaler, max_len=64, num_cols=NUM_COLS):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.le = le
        self.scaler = scaler
        self.num_cols = num_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        text = str(row["merchant_clean"])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        merchant_token = row.get("merchant_token", "")
        try:
            merch_id = int(self.le.transform([str(merchant_token)])[0])
        except:
            merch_id = 0

        nums = self.scaler.transform(
            [row[self.num_cols].fillna(0.0).values.astype(float)]
        )[0]
        label = int(row["label_id"])

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "merchant_id": torch.tensor(merch_id, dtype=torch.long),
            "nums": torch.tensor(nums, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }

class TransformerTabularModel(nn.Module):
    def __init__(
        self,
        transformer_name,
        num_numeric,
        merchant_vocab_size,
        merchant_emb_dim=32,
        hidden_dim=256,
        n_classes=2,
    ):
        super().__init__()
        self.transformer = DistilBertModel.from_pretrained(transformer_name)
        hidden_size = self.transformer.config.hidden_size
        self.merchant_emb = nn.Embedding(merchant_vocab_size, merchant_emb_dim)
        self.tab_proj = nn.Linear(num_numeric, 64)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + merchant_emb_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, input_ids, attention_mask, merchant_id, nums):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        merch = self.merchant_emb(merchant_id)
        tab = torch.relu(self.tab_proj(nums))
        concat = torch.cat([pooled, merch, tab], dim=1)
        return self.head(concat)

if __name__ == "__main__":
    print("🔹 Starting Transformer training...")

    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    labs = sorted(df["label"].unique())
    label_map = {lab: i for i, lab in enumerate(labs)}
    id_to_label = {i: lab for lab, i in label_map.items()}
    df["label_id"] = df["label"].map(label_map)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label_id"]
    )

    scaler = StandardScaler()
    scaler.fit(train_df[NUM_COLS].fillna(0.0))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "tabular_scaler.joblib"))

    le = LabelEncoder()
    le.fit(train_df["merchant_token"].fillna("").astype(str))
    joblib.dump(le, os.path.join(MODEL_DIR, "merchant_encoder.joblib"))

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(MODEL_DIR)

    train_ds = TxnDataset(train_df, tokenizer, le, scaler, max_len=MAX_LEN)
    test_ds = TxnDataset(test_df, tokenizer, le, scaler, max_len=MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_classes = len(label_map)
    merchant_vocab = len(le.classes_)
    model = TransformerTabularModel(
        MODEL_NAME,
        num_numeric=len(NUM_COLS),
        merchant_vocab_size=merchant_vocab,
        merchant_emb_dim=32,
        hidden_dim=256,
        n_classes=n_classes,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_steps)),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train")
        train_losses = []
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            merchant_id = batch["merchant_id"].to(DEVICE)
            nums = batch["nums"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                merchant_id=merchant_id,
                nums=nums,
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(train_losses))

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                merchant_id = batch["merchant_id"].to(DEVICE)
                nums = batch["nums"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    merchant_id=merchant_id,
                    nums=nums,
                )
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.cpu().numpy())

        epoch_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Epoch {epoch} validation macro-F1: {epoch_f1:.4f}")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            torch.save(
                {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()},
                os.path.join(MODEL_DIR, "transformer_best.pt"),
            )
            print(f"✅ Saved new best model (F1={best_f1:.4f})")

    joblib.dump(label_map, os.path.join(MODEL_DIR, "transformer_label_map.joblib"))
    joblib.dump(id_to_label, os.path.join(MODEL_DIR, "transformer_id_to_label.joblib"))
    meta = {
        "model_name": MODEL_NAME,
        "label_map_file": os.path.join(MODEL_DIR, "transformer_label_map.joblib"),
        "id_to_label_file": os.path.join(MODEL_DIR, "transformer_id_to_label.joblib"),
        "scaler": os.path.join(MODEL_DIR, "tabular_scaler.joblib"),
        "merchant_encoder": os.path.join(MODEL_DIR, "merchant_encoder.joblib"),
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "transformer_metadata.joblib"))
    print(f"🎯 Training complete. Best macro-F1: {best_f1:.4f}")
