---
```
████████╗ █████╗  ██████╗ ██╗████████╗
╚══██╔══╝██╔══██╗██╔════╝ ██║╚══██╔══╝
   ██║   ███████║██║  ███╗██║   ██║   
   ██║   ██╔══██║██║   ██║██║   ██║   
   ██║   ██║  ██║╚██████╔╝██║   ██║   
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝   ╚═╝   

    🔖  TAGIT — Smart Labels for Smart Money
```
---

# 🚀 TAGIT — AI Transaction Categorization System

### 🔖 Smart Labels for Smart Money  
A hybrid AI system that classifies financial transactions using **TF‑IDF + Logistic Regression**, **DistilBERT Transformers**, and a clean **Streamlit UI**.

---

# 🌟 Overview

TAGIT intelligently categorizes messy transaction strings like:

```
"UPI/ROHAN@OKHDFC/9823"
"AMZN MUMBAI 4093"
"POS 42342 CAFE COFFEE DAY"
"ZOMATO*ONLINE ORDER"
"HPCL/FUEL/PUNE"
```

It uses a two‑stage hybrid pipeline:

- ⚡ **Baseline Model (Fast):** TF‑IDF + Logistic Regression  
- 🤖 **Transformer Model (Accurate):** DistilBERT + Tabular Features  
- 🔀 **Hybrid Router:** If baseline is confident → use baseline, else fallback to powerful Transformer  

TAGIT also includes a sleek Streamlit interface for real-time testing and CSV batch predictions.

---

# 🧠 Architecture Diagram

```
                   ┌────────────────────────────┐
                   │         RAW INPUT           │
                   │  (UPI / POS / CARD / etc.)  │
                   └────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │       PREPROCESSOR         │
                    │ Clean text, numbers, dates │
                    │ Extract merchant token     │
                    └────────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
┌───────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│  BASELINE MODEL    │   │ TRANSFORMER MODEL │   │   RULE ENGINE      │
│   TF-IDF + LR      │   │ DistilBERT Hybrid │   │ (optional)         │
└───────────────────┘   └────────────────────┘   └────────────────────┘
          │                      │                      │
          └──────────────┬──────┴──────────────┬───────┘
                         ▼                     ▼
                   ┌────────────────────────────────────┐
                   │        TAGIT HYBRID ENGINE         │
                   │ Baseline if conf ≥ 0.70            │
                   │ Else Transformer                    │
                   └────────────────────────────────────┘
                                 ▼
                    ┌────────────────────────────┐
                    │       FINAL CATEGORY        │
                    └────────────────────────────┘
```

---

# ⚙️ Installation

### 1️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🛠️ Usage

## 🔧 Preprocess Data

```bash
python preprocess.py data/transactions.csv data/preprocessed.csv
```

---

# 🔥 Training the Transformer Model (DistilBERT + Tabular Features)

TAGIT uses a hybrid Transformer architecture that merges **DistilBERT embeddings** with **numeric features** (`amount`, `amount_bucket`, `weekday`, `month`) for superior classification accuracy.

---

## ✅ 1. Prepare Preprocessed Data

```bash
python preprocess.py data/transactions.csv data/preprocessed.csv
```

This generates:

```
merchant_clean
merchant_token
amount
amount_bucket
weekday
month
label
```

---

## ✅ 2. Train the Transformer Model

Run:

```bash
python train_transformer.py
```

This script will:

- Load preprocessed data  
- Tokenize merchant text using DistilBERT  
- Train hybrid encoder (Transformer + Tabular MLP)  
- Save all required model files  

### 📦 Saved Artifacts

| File | Purpose |
|------|---------|
| models/transformer_best.pt | Best model weights |
| models/transformer_label_encoder.joblib | Encodes label strings |
| models/transformer_scaler.joblib | Scales numeric features |
| models/tokenizer/ | DistilBERT tokenizer |
| models/transformer_metadata.joblib | Model metadata |

---

## ✅ 3. Predict Using Transformer

```bash
python predict_transformer.py
```

---

## ✅ 4. Hybrid Mode (Baseline + Transformer)

```bash
python smart_predict.py
```

Logic:

```
if baseline_confidence >= 0.70:
    use baseline
else:
    use transformer
```

Results saved to:

```
data/predictions_hybrid.csv
```

---

## ✅ 5. Evaluate Transformer

```bash
python eval.py
```

Outputs macro/weighted F1 and per‑label metrics.

---

## ⚡ GPU Acceleration (Optional but recommended)

Install CUDA‑enabled torch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## ⚡ Train Baseline Model

```bash
python train_baseline.py
```

Produces:

```
models/baseline_pipe.joblib
models/label_encoder.joblib
```

---

## 🔍 Predict (Baseline)

```bash
python predict.py
```

---

## 🧪 Evaluate

```bash
python eval.py
```

---

# 📱 Streamlit App (TAGIT Dashboard)

```bash
streamlit run app2.py
```

Visit:  
👉 http://localhost:8501

---

# ⭐ Highlights

- ⚡ Real-time baseline inference  
- 🤖 High-accuracy Transformer model  
- 🔀 Smart hybrid confidence routing  
- 🎨 Beautiful Streamlit dashboard  
- 🧹 Clean architecture & modular design  
- 🧩 Easy to extend  
- 📊 Professional metrics (macro/weighted F1)

---

# 📬 Team Diamonds

Made with ❤️ for innovation.
