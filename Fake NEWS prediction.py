# work flow
# News data --> Data precessing --> training and test data --> Logistic Regression model -->  Trained Logistic Regression model <-- new data to Trained Logistic Regression model

# importing dependencies

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
# printing the stopwords in english
print(stopwords.words('english'))

# data pre Processing
# loading dataset to pandas dataframe.
true_news = pd.read_csv(r'C:\Users\allen\OneDrive\Desktop\understanding\data\News _dataset\True.csv')
fake_news = pd.read_csv(r'C:\Users\allen\OneDrive\Desktop\understanding\data\News _dataset\Fake.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(true_news.head())
print(fake_news.head())
# Add label
fake_news["label"] = 1  # FAKE
true_news["label"] = 0  # TRUE
print(fake_news.shape)
print(true_news.shape)

# Merge shuffle
dataframe = pd.concat([true_news,fake_news], ignore_index=True)
dataframe = dataframe.sample(frac=1, random_state=3).reset_index(drop=True)

# check
print(dataframe.head())
print(dataframe.shape)

# number of missing values in dataset
print(dataframe.isnull().sum())

# merging the title and text
dataframe['content'] = dataframe['title'] + ' ' + dataframe['text']
print(dataframe['content'])

# separating data and label
# Separating features and label
x = dataframe['content'].values  # only the processed text
y = dataframe['label'].values    # 0 or 1
print(x.shape)  # (44898,)
print(y.shape)  # (44898,)
print(x[0])     # verify one stemmed article

# merging the title and text
dataframe['content'] = dataframe['title'] + ' ' + dataframe['text']

# Stemming
stop_words = set(stopwords.words('english'))
poet_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [poet_stem.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

print("Stemming in progress...")
dataframe['content'] = dataframe['content'].apply(stemming)
print("Done! Sample:", dataframe['content'][0])

# Separating features and label (AFTER stemming)
x = dataframe['content'].values
y = dataframe['label'].values

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)

# Train / Test Split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, stratify=y, random_state=2)

# Train Model
model = LogisticRegression()
model.fit(x_train, y_train)

# Accuracy
train_acc = accuracy_score(y_train, model.predict(x_train))
test_acc  = accuracy_score(y_test,  model.predict(x_test))
print(f"\nTraining Accuracy : {train_acc:.4f}")
print(f"Testing  Accuracy : {test_acc:.4f}")

# Making a predictive system
x_new = x_test[0]
prediction = model.predict(x_new)
print(prediction)

if prediction[0]==0:
    print("THE NEWS IS REAL")
else:
    print("THE NEWS IS FAKE")