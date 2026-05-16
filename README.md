## Team Members
- Amrutha
- Indrajith V S
- Sanika J R

## Problem Statement
Classify emails as spam or ham using NLP techniques and machine learning models, and build a preprocessing pipeline to clean and prepare raw email text for classification.

## Dataset
SpamAssassin Public Corpus / Enron Email Dataset

## Motivation

Email spam detection is important for:
- Filtering unwanted and harmful emails automatically
- Protecting users from phishing and malicious content
- Reducing manual effort in managing email inboxes

## Dataset Description

- Source: SpamAssassin Public Corpus / Enron Email Dataset
- Records: Thousands of labeled spam and ham emails
- Features include:
  - Email subject line
  - Email body text
  - Sender information
  - HTML content and links
  - Word frequency patterns

### Target Variable
- **Spam (1)** → Unwanted or junk email
- **Ham (0)** → Legitimate email

---

## Methodology

### 1. Data Preprocessing
- Removed HTML tags and special characters
- Converted text to lowercase
- Tokenized email content into individual words
- Removed stop words (e.g., "the", "is", "and")
- Applied lemmatization to reduce words to their root form

### 2. Exploratory Data Analysis (EDA)
- Distribution of spam vs ham emails
- Most frequent words in spam and ham categories
- Email length and character count analysis
- Word cloud visualization of spam keywords

### 3. Feature Extraction
- Applied TF-IDF (Term Frequency–Inverse Document Frequency) for classical models
- Used Word Embeddings (Word2Vec / GloVe) for the LSTM deep learning model

### 4. Model Building
Models used:
- Naive Bayes
- Support Vector Machine (SVM)
- LSTM (Long Short-Term Memory)

### 5. Model Evaluation

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Naive Bayes | 97.0% | 0.96 |
| Support Vector Machine | 98.5% | 0.98 |
| LSTM | 99.0% | 0.99 |

LSTM performed best and was selected as the final model due to its ability to understand word context and sequential patterns.

---

## Model Explainability

We analyzed feature importance to interpret model predictions.

Example insights:
- Words like "free", "win", "click", and "offer" strongly indicate spam
- Emails with excessive links and HTML formatting are more likely spam
- Low word diversity and repetitive phrases contribute to spam classification

---

## Deployment

The model is deployed using Streamlit.

<img width="979" height="1019" alt="Screenshot 2026-05-16 121813" src="https://github.com/user-attachments/assets/7aced6c2-f973-42fe-aee6-9959056b5e3a" />
<img width="953" height="985" alt="Screenshot 2026-05-16 121925" src="https://github.com/user-attachments/assets/cacf86e1-df89-4656-84dc-484b4adb5d4f" />


🔗 Live App: https://predictive-project-grp-3-dworfudo6xj5t8qizby8du.streamlit.app/
