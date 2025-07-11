import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import sys
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

Final_Data_array = []
Sentiments = []

def download_and_verify_nltk_resources():
    resources = {
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4'
    }
    
    for resource, path in resources.items():
        print(f"Checking {resource}...")
        try:
            nltk.data.find(path)
            print(f"{resource} is already downloaded")
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource)
            print(f"{resource} has been downloaded")

#download_and_verify_nltk_resources()

try:
    print("Loading CSV file...")
    data = pd.read_csv(r"C:\\Users\\gaurm\\Downloads\\archive\\redmi6.csv", encoding='latin-1', usecols=[0])
    print("Initializing NLP tools...")
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    positive_words = ["good", "great", "excellent", "amazing", "happy", "love", "awesome", "fantastic", "fabulous", "killer","working","best","nice"]
    negative_words = ["bad", "poor", "terrible", "sad", "hate", "awful", "disappointing", "small","Useless","trash"]
    
    def preprocess_text(text):
        if pd.isna(text):
            print("Warning: Found NaN value in text")
            return [], "Neutral"
        try:
            text = str(text)
            text = text.lower()
            tokens = re.findall(r'\b\w+\b', text)
            tokens = [word for word in tokens if word not in stop_words]
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
            positive_count = sum(1 for word in lemmatized_tokens if word in positive_words)
            negative_count = sum(1 for word in lemmatized_tokens if word in negative_words)
            if positive_count > negative_count:
                sentiment = 1
            elif negative_count > positive_count:
                sentiment = 0
            else:
                sentiment =0
            return lemmatized_tokens, sentiment
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            return [], 0

    for i in range(0, 279):  
        row_text = data.iloc[i, 0]
        processed_tokens, sentiment = preprocess_text(row_text)
        Final_Data_array.append(processed_tokens)
        Sentiments.append(sentiment)
    
    
    documents = [' '.join(tokens) for tokens in Final_Data_array]
    print(documents)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    y_train = Sentiments
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(tfidf_matrix, y_train)
    test_data = ["This phone is amazing", "I hated this movie" , "Mi is best phone", "Hello","poor phone" , "very nice","Mid-Range Phone not that good","Could do with some improvements"]
    X_test = vectorizer.transform(test_data)
    predictions = model.predict(X_test)
    #predictions = model.predict_proba(X_test)
    ActualPredictions = [1,0,1,0,0,1,0,0]
    cm = confusion_matrix(ActualPredictions,predictions)
    print(cm)
    # for i in predictions:
    #     if i[1]>=0.6:
    #         Predictions.append("positive")
    #     elif 0.4<=i[1]<0.6:

    #         Predictions.append("neutral")
    #     else:
    #         Predictions.append("negative")
    print(predictions)
    
except FileNotFoundError:
    print("Error: Could not find the CSV file. Please check the file path.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
    sys.exit(1)
