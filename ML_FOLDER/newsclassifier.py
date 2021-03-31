import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB


news_df= pd.read_csv(r'C:\Users\DELL\Desktop\NEWS\news.csv')

# print(news_df.info())

# vectorizer= CountVectorizer()
# counts = vectorizer.fit_transform(news_df['text'].values)


# print(news_df['text'].values)
# print(type(counts))

# classifier = PassiveAggressiveClassifier()
# news_df['label']= news_df['label'].replace(to_replace=['FAKE', 'REAL'], value=[0,1])
x = news_df['text']
y = news_df.label
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 7 )


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
vectorizer= CountVectorizer()

train_df= vectorizer.fit_transform(x_train)
test_df = vectorizer.transform(x_test)



# train_df= tfidf_vectorizer.fit_transform(x_train)
# test_df=tfidf_vectorizer.transform(x_test)


def clf(model):
    clf=model 
    clf.fit(train_df, y_train)

    y_pred= clf.predict(test_df)
    score= accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score*100,2)}%')


clf(PassiveAggressiveClassifier(max_iter=50))

clf(MultinomialNB())



# print(y)