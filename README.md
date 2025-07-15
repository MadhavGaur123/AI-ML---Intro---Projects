# 🤖 AI/ML Mini Projects Collection

This repository contains five concise and practical machine learning projects developed using **Python** and popular libraries like **Scikit-learn**, **Pandas**, **NumPy**, and **Matplotlib**. Each project targets a different ML concept: classification, clustering, regression, and NLP-based sentiment analysis.

---

## 🌐 Project List Overview

| Project Name                   | Technique               | Focus                              |
| ------------------------------ | ----------------------- | ---------------------------------- |
| `DecisionTreeDiseaseDetection` | Classification          | Heart disease prediction           |
| `DecisionTreeLoanApproval`     | Classification + ROC    | Loan approval model with ROC       |
| `K-means-clustering`           | Unsupervised Clustering | Customer segmentation              |
| `LinearRegressionModel`        | Regression              | Housing price prediction (3D plot) |
| `SentimentAnalysis`            | NLP + Classification    | Text sentiment analysis (TF-IDF)   |

---

## 📊 Decision Tree: Disease Detection

**Goal:** Predict heart disease using medical attributes.

### Features Used:

* `chol` (cholesterol level)
* `oldpeak` (ST depression)
* `thalach` (maximum heart rate achieved)

### Highlights:

* Uses both **Decision Tree** and **Random Forest** classifiers.
* Evaluation with **confusion matrix**, **precision**, **recall**, and **ROC-AUC**.
* Compares the performance visually on the **ROC Curve**.

### Bonus:

* Model predicts user input via console (commented out for GUI-free testing).

---

## 📋 Decision Tree: Loan Approval with ROC

**Goal:** Classify loan approval decisions based on applicant attributes.

### Features Used:

* `LoanAmount`
* Categorical attributes mapped to numerical values (e.g., Gender, Education)

### Highlights:

* Uses a **Decision Tree** classifier.
* Computes **TPR**, **FPR**, and plots **ROC-like curve** manually.
* Interactive threshold input to observe classifier performance.

---

## 📊 K-Means Clustering: Customer Segmentation

**Goal:** Segment mall customers based on spending behavior.

### Features Used:

* `Annual Income`
* `Spending Score`

### Highlights:

* Applies **Elbow Method** to determine optimal clusters.
* Visualizes clusters with **different colors** and **centroids**.
* Uses `k-means++` initialization and `random_state=42` for reproducibility.

---

## 📈 Linear Regression: Housing Price Prediction

**Goal:** Predict house prices based on square footage and number of bedrooms.

### Features Used:

* `SqFt`
* `Bedrooms`

### Highlights:

* Uses **Linear Regression** from Scikit-learn.
* Evaluated with **R-squared score** and **MSE**.
* Warns if prediction confidence is low.
* Includes a beautiful **3D surface plot** with actual data.

---

## 🌐 Sentiment Analysis using TF-IDF

**Goal:** Classify product reviews as Positive or Negative.

### Steps Involved:

1. Loads CSV file of textual reviews.
2. Preprocesses text using:

   * Tokenization
   * Stopword removal
   * Lemmatization
   * Polarity word counting (basic lexicon-based filtering)
3. Converts processed text into **TF-IDF vectors**.
4. Trains a **Logistic Regression** model.

### Highlights:

* Evaluates with a **confusion matrix**.
* Custom test data used to verify sentiment predictions.
* Can download missing NLTK resources automatically.

---

## 📆 Dependencies

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `nltk`

Install via pip:

```bash
pip install pandas numpy scikit-learn matplotlib nltk
```

---

## 🚀 How to Run

Each file is self-contained and can be run directly using Python:

```bash
python filename.py
```

Ensure corresponding CSV files (datasets) are present at the correct path or adjust paths accordingly in the code.

---

## 📊 Final Thoughts

These projects illustrate real-world use cases for various ML techniques. They’re ideal for:

* Students learning ML
* Demonstrating project work in portfolios
* Building end-to-end ML pipelines

Feel free to fork, improve, or build on top of them!

---

Let me know if you'd like separate detailed `README.md` files for each project in a subfolder structure (`/DecisionTreeDiseaseDetection`, etc.).
