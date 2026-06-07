# 📱 SMS Spam Classifier

An end-to-end Machine Learning project that classifies SMS messages into **Ham** (legitimate) or **Spam**. This repository contains a complete pipeline, including data cleaning, exploratory data analysis (EDA), text preprocessing, model evaluation, and deployment-ready serialized model files.

---

## 🎯 Project Overview

In the era of digital communication, mobile spam remains a persistent issue. This project aims to build an accurate classifier to filter out spam messages. 

Since falsely classifying a legitimate, critical message as spam (**False Positive**) is highly disruptive to users, the primary objective of this project is to achieve **100% Precision** while maintaining high overall accuracy.

---

## 📊 Dataset Description

The model was trained on the `spam (1).xls` dataset.
- **Total Records:** 5,572 messages
- **Distribution:**
  - **Ham (Legitimate):** 87.37% (4,516 messages)
  - **Spam:** 12.63% (653 messages)
- **Features:** 
  - `target` (Label: `ham` or `spam`)
  - `text` (The raw message content)

---

## ⚙️ Project Pipeline

The pipeline is split into five key phases inside the Jupyter Notebook:

### 1. Data Cleaning
- Handled missing and duplicate values (removed 403 duplicate messages).
- Encoded target labels (`ham` $\rightarrow$ `0`, `spam` $\rightarrow$ `1`).
- Cleaned column names and dropped empty/unnecessary columns.

### 2. Exploratory Data Analysis (EDA)
- Analyzed and visualized the distribution of target classes.
- Used NLTK to compute character count, word count, and sentence count for each message.
- Generated histograms, pair plots, and heatmaps to explore correlations between message length and classification.
- Created **Word Clouds** to visualize the most dominant words in ham and spam messages.

### 3. Data Preprocessing
Every message undergoes a comprehensive preprocessing function:
1. **Lowercasing:** Standardizes all words to lower case.
2. **Tokenization:** Splits sentences into lists of words.
3. **Filtering Special Characters:** Keeps only alphanumeric tokens.
4. **Stopwords & Punctuation Removal:** Filters out common English stopwords (e.g., "is", "the", "and") and punctuation marks.
5. **Stemming:** Uses NLTK's `PorterStemmer` to reduce words to their base form (e.g., "loving" or "loved" $\rightarrow$ "love").

### 4. Feature Engineering
- Text inputs are converted to numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization**.
- The vectorizer is capped at the top **3,000 features** to maximize efficiency and prevent overfitting.

### 5. Model Training & Comparison
We trained and compared 11 different classification algorithms. Below is the performance summary on the test set:

| Algorithm | Accuracy | Precision |
| :--- | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | 90.52% | **100.00%** |
| **Multinomial Naive Bayes (MNB)** | **97.10%** | **100.00%** |
| **Random Forest Classifier** | 97.58% | 98.29% |
| **Support Vector Classifier (SVC)** | 97.58% | 97.48% |
| **Extra Trees Classifier (ETC)** | 97.49% | 97.46% |
| **Logistic Regression (LR)** | 95.84% | 97.03% |
| **XGBoost Classifier (xgb)** | 96.71% | 94.83% |
| **AdaBoost Classifier** | 96.03% | 92.92% |
| **Gradient Boosting (GBDT)** | 94.68% | 91.92% |
| **Bagging Classifier (BgC)** | 95.84% | 86.82% |
| **Decision Tree Classifier (DT)** | 93.04% | 81.73% |

Additionally, ensemble methods were tested:
* **Voting Classifier** (Soft Voting of SVC, MNB, and ETC): Accuracy **98.16%** | Precision **99.17%**
* **Stacking Classifier** (SVC, MNB, ETC meta-trained with Random Forest): Accuracy **98.16%** | Precision **95.42%**

---

## 🏆 Final Model Selection

The **Multinomial Naive Bayes (MNB)** model was selected as the final classifier. 
* **Reasoning:** It achieves a perfect **1.0 (100%) precision score** with a high accuracy of **97.10%**. This ensures that **no legitimate messages (ham) are classified as spam**.
* The trained vectorizer and model are saved as serialized pickle files:
  * [`vectorizer (1).pkl`](file:///c:/Users/rayan/Desktop/python-codes/sms-spam-classifier/vectorizer%20(1).pkl)
  * [`model (1).pkl`](file:///c:/Users/rayan/Desktop/python-codes/sms-spam-classifier/model%20(1).pkl)

---

## 📁 Repository Structure

```directory
sms-spam-classifier/
├── sms-spam-detection (1).ipynb   # Jupyter Notebook containing analysis & training code
├── spam (1).xls                   # The dataset containing spam and ham messages
├── model (1).pkl                  # Serialized Multinomial Naive Bayes model
├── vectorizer (1).pkl              # Serialized TF-IDF vectorizer object
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 📋 Prerequisites

To run this project, make sure you have the following libraries installed:

```bash
pip install pandas numpy scikit-learn nltk xlrd wordcloud matplotlib seaborn xgboost
```

### 💻 Loading the Model for Predictions

You can easily integrate the trained model into your python applications or web services using the snippet below:

```python
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Stemmer
ps = PorterStemmer()

def transform_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Tokenize
    text = nltk.word_tokenize(text)
    
    # 3. Keep alphanumeric tokens
    y = [i for i in text if i.isalnum()]
    
    # 4. Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    # 5. Apply Porter Stemming
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)

# Load Vectorizer and Model (adjust filenames if necessary)
try:
    tfidf = pickle.load(open("vectorizer (1).pkl", "rb"))
    model = pickle.load(open("model (1).pkl", "rb"))
except FileNotFoundError:
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))

def predict_message(message):
    # Preprocess
    cleaned_msg = transform_text(message)
    # Vectorize
    vectorized_input = tfidf.transform([cleaned_msg])
    # Predict
    prediction = model.predict(vectorized_input)[0]
    return "Spam" if prediction == 1 else "Ham"

# Test prediction
test_msg = "URGENT! You have won a 1-week free membership to our £100,000 Prize Draw! Claim now."
print(f"Message: {test_msg}")
print(f"Result: {predict_message(test_msg)}")
```

---

## 👥 License
This project is open-source and available under the MIT License.
