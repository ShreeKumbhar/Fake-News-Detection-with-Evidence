# 📰 Fake News Detection with Evidence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3DDC84?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A Hybrid AI approach combining Machine Learning with Real-Time Evidence Retrieval for explainable news credibility assessment.**

[Features](#-features) · [Installation](#-installation) · [Architecture](#-architecture) · [Roadmap](#-roadmap)

</div>

---

## 📌 Overview

**Fake News Detection with Evidence** goes beyond binary classification. It fuses a TF-IDF + Logistic Regression pipeline with Wikipedia-based evidence retrieval to give users not just a prediction, but a *reason* — making results transparent, explainable, and trustworthy.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **ML Classification** | Predicts "Fake" or "Real" using a TF-IDF + Logistic Regression pipeline |
| 🔍 **Evidence Retrieval** | Fetches real-time Wikipedia context for key claims |
| ⚖️ **Explainable Verdict** | Combines model confidence + evidence availability into a human-readable result |
| 📊 **Confidence Scoring** | Displays the model's certainty as a percentage |
| 🧠 **NLP Preprocessing** | Tokenization, stopword removal, and lemmatization via NLTK |
| 🌐 **Interactive UI** | Clean, easy-to-use Streamlit web interface |

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Web Framework:** Streamlit
- **ML Pipeline:** Scikit-Learn (TF-IDF Vectorizer + Logistic Regression)
- **NLP:** NLTK (Tokenization, Stopwords, Lemmatization)
- **Evidence Source:** Wikipedia API (`wikipedia-api`)
- **Serialization:** Pickle (`.pkl` model artifacts)

---

## 📁 Project Structure

```
Fake-News-Detection-with-Evidence/
│
├── model/
│   ├── model.pkl            # Trained Logistic Regression model
│   └── tfidf.pkl            # Fitted TF-IDF vectorizer
│
├── app.py                   # Streamlit web application
├── train_model.py           # Model training script
├── utils.py                 # Preprocessing & evidence retrieval utilities
├── requirements.txt         # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ShreeKumbhar/Fake-News-Detection-with-Evidence.git
cd Fake-News-Detection-with-Evidence
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Train the Model

Run this **once** to generate the model artifacts in the `model/` directory:

```bash
python train_model.py
```

This creates:
- `model/model.pkl` — Trained Logistic Regression classifier
- `model/tfidf.pkl` — Fitted TF-IDF vectorizer

### Step 2 — Launch the App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Step 3 — Analyze a Claim

1. Paste a news article or claim into the text area
2. Click **"Analyze Credibility"**
3. View the **Prediction**, **Confidence Score**, **Evidence Summary**, and **Final Verdict**

---

## 🧩 Architecture

```
User Input
    │
    ▼
Preprocessing (NLTK)
    │
    ├──────────────────────┐
    ▼                      ▼
TF-IDF Features       Keyword Extraction
    │                      │
    ▼                      ▼
ML Model              Wikipedia Evidence
(Logistic Regression)     Retrieval
    │                      │
    ▼                      ▼
Prediction &          Context Summary
Confidence Score
    │                      │
    └──────────┬───────────┘
               ▼
          Verdict Logic
               │
               ▼
        Final Output Display
     (Credible / Uncertain /
      Potentially Misleading)
```

### Verdict Mapping

| Condition | Verdict |
|---|---|
| High confidence + Evidence found | ✅ Credible |
| Medium confidence or No evidence | ⚠️ Uncertain |
| Low confidence + Contradictory signals | ❌ Potentially Misleading |

---

## 🔮 Roadmap

- [ ] **Transformer Models** — Upgrade to BERT / RoBERTa for semantic understanding
- [ ] **Real-time News API** — Cross-reference against live news sources
- [ ] **Knowledge Graph Integration** — Map entities and relationships for deeper fact-checking
- [ ] **Multi-language Support** — Extend detection beyond English
- [ ] **Browser Extension** — In-browser real-time detection

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👩‍💻 Author

**Shree Kumbhar**  
[![GitHub](https://img.shields.io/badge/GitHub-ShreeKumbhar-181717?style=flat&logo=github)](https://github.com/ShreeKumbhar)

---

<div align="center">
⭐ If you found this project useful, please consider giving it a star!
</div>
