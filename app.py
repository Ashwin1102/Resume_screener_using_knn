import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from PyPDF2 import PdfReader
from flask import Flask, render_template, request, flash
import string

# download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# load data
data = pd.read_csv('UpdatedResumeDataSet.csv')

# preprocess data
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # convert text to lowercase
    text = text.lower()

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # tokenize text
    tokens = word_tokenize(text)

    # remove stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # join tokens back into string
    return " ".join(filtered_tokens)

data['cleaned_resume'] = data['Resume'].apply(preprocess)

# create TF-IDF vectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['cleaned_resume'])
y = data['Category']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# build KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# make predictions on test data
y_pred = knn.predict(X_test)

# evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/convert', methods=['POST'])
def convert():
    pdf_file = request.files['pdf_file']
    text = extract_text_from_pdf(pdf_file)
    resume_cluster = cluster_resume(text)
    return render_template('result.html', text=text, resume_cluster=resume_cluster)

def extract_text_from_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        flash("Error occurred while processing the PDF file: " + str(e))
        return ""

def cluster_resume(text):
    # Convert the preprocessed resume text into a TF-IDF vector
    resume_vector = tfidf.transform([text])

    # Use the trained KNN classifier to predict the category of the resume
    predicted_category = knn.predict(resume_vector)[0]
    return predicted_category


if __name__ == '__main__':
    app.run()
