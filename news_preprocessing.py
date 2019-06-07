import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from nltk.tokenize import RegexpTokenizer
from gensim.models import Doc2Vec
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score


pd.set_option('max_colwidth', 40000)
cores = multiprocessing.cpu_count()
t = RegexpTokenizer(r'\w+')
train_size = 0.95
doc_size = 64
filename = 'log_model.sav'
lim = 1000


def reshape_data(data):
    dates = data['Date'].unique()
    data = data.set_index('Date')
    res_data = pd.DataFrame(columns=['item','Date', 'Tag', 'Top'])
    topics = ['Top' + str(i) for i in range(1, 26)]

    for i in dates:
        for j in topics:
            res_data = res_data.append({'item':i+j,'Date':i, 'Tag':data.loc[i]['Label'], 'Top': data.loc[i][j]}, ignore_index=True)
    res_data = res_data.set_index('item')
    res_data.to_csv('fin_news.csv')
    return res_data


data = pd.read_csv('fin_news.csv')
# data = data.sample(frac=1)
x = data['Top']
x = [t.tokenize(i.lower().replace('\\t', ' ')) for i in x]

y = np.array(data['Tag'])
print(data['Tag'].mean())
x_train, x_test, y_train, y_test = x[0:int(len(x)*train_size)], x[int(len(x)*train_size):len(x)],\
                                   y[0:int(len(y)*train_size)], y[int(len(y)*train_size):len(y)]
train_tagged = [TaggedDocument(x_train[i], str(y_train[i])) for i in range(int(len(x)*train_size))]

model_dbow = Doc2Vec(dm=0, vector_size=doc_size, min_count=1, hs=1, workers=cores, window=9)
model_dbow.build_vocab(train_tagged)
print(y_test.mean(), y_train.mean())
# model_dbow.train(train_tagged, total_examples=model_dbow.corpus_count, epochs=250)
# model_dbow.save('d2v.model_' + str(doc_size))
model_dbow = Doc2Vec.load('d2v.model_' + str(doc_size))

X_train = np.array([model_dbow.infer_vector(i, steps=20) for i in x_train])
X_test = np.array([model_dbow.infer_vector(i, steps=20) for i in x_test])

# clf = pickle.load(open(filename, 'rb'))
clf = MLPClassifier(hidden_layer_sizes=(100, 50, 10), activation='relu', max_iter=100, alpha=0.0001,
                     solver='adam', verbose=10,  random_state=21, tol=0.000000001)
clf.fit(X_train, y_train)
pickle.dump(clf, open(filename, 'wb'))


y_pred = clf.predict(X_train)
print('TRAIN')
print('Acc: ',accuracy_score(y_train, y_pred))
print('Precision: ',precision_score(y_train, y_pred))
print('Recall: ',recall_score(y_train, y_pred))
print('F1: ',f1_score(y_train, y_pred))
print('ROC_AUC: ', roc_auc_score(y_train, y_pred))

y_pred = clf.predict(X_test)
print('TEST')
print('Acc: ',accuracy_score(y_test, y_pred))
print('Precision: ',precision_score(y_test, y_pred))
print('Recall: ',recall_score(y_test, y_pred))
print('F1: ',f1_score(y_test, y_pred))
print('ROC_AUC: ', roc_auc_score(y_test, y_pred))

