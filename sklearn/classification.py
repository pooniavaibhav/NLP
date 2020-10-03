import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
nlp = spacy.load('en_core_web_sm')
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import datetime
from imblearn.over_sampling import BorderlineSMOTE
data = pd.read_csv("/home/webhav/datanext/src/python/AIService/datasets/suppresstraining.csv")

#(data.Label.value_counts())
start_time = datetime.datetime.now()
encoder = LabelEncoder()


class text_classification:

    def __init__(self):
        self.X = self.preprocess()
        data['Label'] = encoder.fit_transform(data['Label'])
        self.Y = data['Label']

    def run(self, prop, datanextId, component):
        self.classification(self.X,self.Y)

    def properties(self, prop):
        pass

    def output(self):
        pass

    def preprocess(self):
        data['tokenized'] = data.apply(lambda row: nlp(row['Message']), axis=1)
        data['lemmatized'] = data.apply(lambda row: self.lemmatization(row['tokenized']), axis=1)
        data['clean'] = data.apply(lambda row: self.stop_words(row['lemmatized']), axis=1)
        data['str'] = data.apply(lambda row: self.process(row['clean']), axis=1)
        return data['str']


    def lemmatization(self, X):
        return[x.lemma_ for x in X]

    def stop_words(self, X):
        temp_list =[]
        for i in X:
            if nlp.vocab[i].is_stop:
                continue
            else:
                temp_list.append(i)
        return temp_list

    def process(self, X):
        listTostr = ' '.join(str(elem) for elem in X)
        return listTostr


    def classification(self,X,Y):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        #text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', MultinomialNB())])
        vectorizer = TfidfVectorizer()
        # vectorizer2 = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        sm = BorderlineSMOTE()
        X_res, Y_res = sm.fit_sample(X_train_tfidf, y_train)
        clf = MultinomialNB()
        clf.fit(X_res, Y_res)
        prediction = clf.predict(X_test_tfidf)
        print(prediction)
        final_time = start_time - datetime.datetime.now()
        print(final_time)
        print(metrics.classification_report(y_test,prediction))
        print(metrics.roc_auc_score(y_test, prediction))
        #print(prediction)




if __name__ == "__main__":
    obj = text_classification()
    obj.run("a","b","c")