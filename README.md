
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

# 📂 Project Structure

```
📦 TAGIT
│
├── EF/
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── predict.py
│   ├── predict_transformer.py
│   ├── smart_predict.py
│   ├── eval.py
│   ├── app2.py
│   ├── taxonomy.yaml
│   └── generate_synthetic.py
│
├── data/
│   └── sample_transactions.csv
│
├── models/ (ignored in git)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

### 1️⃣ Create virtual environment

```
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

# 🛠️ Usage

## 🔧 Preprocess Data

```
python EF/preprocess.py data/transactions.csv data/preprocessed.csv
```

---

## ⚡ Train Baseline Model

```
python EF/train_baseline.py
```

Produces:

```
models/baseline_pipe.joblib
models/label_encoder.joblib
```

---

## 🤖 Train Transformer Model (Optional)

Requires GPU for speed:

```
python EF/train_transformer.py
```

Produces:

```
models/transformer_best.pt
models/transformer_label_encoder.joblib
models/transformer_scaler.joblib
models/tokenizer/
```

---

## 🔍 Predict (Baseline)

```
python EF/predict.py
```

---

## 🧪 Evaluate

```
python EF/eval.py
```

Outputs macro/weighted F1, per-class metrics.

---

# 📱 Streamlit App (TAGIT Dashboard)

```
streamlit run EF/app2.py
```

Visit:  
👉 http://localhost:8501

### UI Overview

```
┌──────────────────────────────────────────────────────────┐
│                    💸 TAGIT Dashboard                    │
├──────────────────────────────────────────────────────────┤
│ 🔍 Enter Transaction Text                                │
│ [ UPI/ROHAN@OKHDFC/9843 ] [ Predict ]                    │
│ ✔ Category: P2P Transfer                                 │
│ ✔ Confidence: 0.93                                       │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ 📤 Upload CSV for Bulk Prediction                        │
│ [ Choose File ]                                          │
│                                                          │
│ merchant            predicted_label    confidence        │
│ -----------------------------------------------------    │
│ AMZN MUMBAI         Shopping            0.88             │
│ HPCL PUNE           Fuel                0.91             │
└──────────────────────────────────────────────────────────┘
```

---

# 🎨 TAGIT Branded ASCII Banner

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

# 📦 Requirements

```
pandas==2.1.2
numpy==1.26.4
scikit-learn==1.3.2
matplotlib==3.8.1
joblib==1.3.2
pyyaml==6.0

transformers==4.34.0
torch==2.2.0

streamlit==1.24.0

tqdm==4.66.1
```

---

# 🔥 .gitignore

```
__pycache__/
*.pyc
.venv/
models/
data/*.csv
!data/sample_transactions.csv
tokenizer/
*.pt
```

---

# 🏆 Hackathon Highlights

- ⚡ Real-time baseline inference  
- 🤖 High-accuracy Transformer model  
- 🔀 Smart hybrid confidence routing  
- 🎨 Beautiful Streamlit dashboard  
- 🧹 Clean architecture & modular design  
- 🧩 Easy to extend: add new merchants, new categories  
- 📊 Professional metrics (macro/weighted F1)

---

# 📬 Team TAGIT

Made with ❤️ for innovation.

