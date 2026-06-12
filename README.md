# Fake-NEWS-Prediction-ML
# 📰 Fake News Prediction using Machine Learning

A machine learning project that classifies news articles as **real or fake** using Natural Language Processing (NLP) and Logistic Regression — achieving **98.67% test accuracy**.

---

## 📌 Project Overview

With the rise of misinformation, automatically detecting fake news has become critical. This project builds an end-to-end ML pipeline that:

- Processes raw news article text
- Applies NLP techniques (stemming, stopword removal)
- Converts text to numerical features using TF-IDF
- Trains a Logistic Regression classifier
- Predicts whether a given news article is real or fake

---

## 📊 Dataset

**Source:** [Fake and Real News Dataset — Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

| File | Articles | Label |
|------|----------|-------|
| `Fake.csv` | 23,481 | 1 (Fake) |
| `True.csv` | 21,417 | 0 (Real) |
| **Combined** | **44,898** | — |

**Columns used:** `title`, `text` (combined into `content`)

---

## 🧠 ML Pipeline

```
Raw CSVs → Label + Merge → Content Column → Stemming → TF-IDF → Train/Test Split → Logistic Regression → Prediction
```

| Step | Description |
|------|-------------|
| Data Loading | Load Fake.csv and True.csv into pandas |
| Labelling | Fake = 1, Real = 0 |
| Merging | Concatenate + shuffle (random_state=3) |
| Content Creation | `title + text` → single `content` column |
| Stemming | PorterStemmer + stopword removal |
| TF-IDF | Text → numerical feature matrix |
| Train/Test Split | 80% train / 20% test (stratified) |
| Model | Logistic Regression |

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Training Accuracy | **99.19%** |
| Testing Accuracy | **98.67%** |

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas** — data loading and manipulation
- **NLTK** — stopwords, PorterStemmer
- **scikit-learn** — TF-IDF, Logistic Regression, train/test split
- **NumPy** — array operations
- **re** — regex for text cleaning

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/allen745/fake-news-prediction.git
cd fake-news-prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn nltk
```

### 3. Download NLTK stopwords

```python
import nltk
nltk.download('stopwords')
```

### 4. Add the dataset

Download `Fake.csv` and `True.csv` from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place them in a `data/` folder.

### 5. Run the script

```bash
python fake_news_prediction.py
```

---

## 📁 Project Structure

```
fake-news-prediction/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── fake_news_prediction.py   # Main script
├── model.pkl                 # Saved model (after running)
├── vectorizer.pkl            # Saved TF-IDF vectorizer
└── README.md
```

---

## 🔮 Sample Prediction

```python
import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

news = ["Breaking: Scientists discover water on Mars"]
vector = vectorizer.transform([news[0]])
prediction = model.predict(vector)

print("REAL" if prediction[0] == 0 else "FAKE")
```

---

## 🔭 Future Improvements

- [ ] Deploy as a FastAPI REST endpoint
- [ ] Build a React frontend for live predictions
- [ ] Try advanced models (Random Forest, LSTM, BERT)
- [ ] Add confidence score to predictions

---

## 👨‍💻 Author

**Allen** — AI & ML Developer | Patent Holder  
B.Tech AI & Data Science — ADIT, Gujarat  
linkedin:- https://www.linkedin.com/in/allen-christian-708545409/
GitHub: [@allen745](https://github.com/allen745)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
