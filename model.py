import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier


df = pd.read_json("dataset.json", lines=True)


path = "dataset.json"


def load_data(path):
    df = pd.read_json(path, lines=True)
    df["label"] = df.annotation.apply(lambda x: x.get('label'))
    df["label"] = df.label.apply(lambda x: x[0])
    x = df.content.values
    y = df.label.values
    return x, y


url = 'dataset.json'
X, y = load_data(url)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y)

vect = CountVectorizer()  # gets freq table fr tokens in all docs(comments)-Hence lemmatization applied
tfidf = TfidfTransformer()
clf = RandomForestClassifier()


# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
clf.fit(X_train_tfidf, y_train)  # fitting alg with vectors(by applyng countvectorization and tfidf)

# predict on test data
X_test_counts = vect.transform(X_test)
X_test_tfidf = tfidf.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)


pickle.dump(vect, open('transform.pkl', 'wb'))
pickle.dump(tfidf, open('transform1.pkl', 'wb'))


# test accuracy
print("Test accuracy")
print(clf.score(X_test_tfidf, y_test)*100)


x1 = vect.transform(X)
x1_tfidf = tfidf.transform(x1)
print(clf.score(x1_tfidf, y)*100)


pickle.dump(clf, open('model.pkl', 'wb'))
print('Sucess')
