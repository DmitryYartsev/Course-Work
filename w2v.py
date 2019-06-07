import pandas as pd
import multiprocessing
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec


pd.set_option('max_colwidth', 40000)
cores = multiprocessing.cpu_count()
t = RegexpTokenizer(r'\w+')
vec_size = 64
lim = 1000


data = pd.read_csv('fin_news.csv')
data = data.sample(frac=1)
x = data['Top']
x = [t.tokenize(i.lower().replace('\\t', ' ')) for i in x]


model = Word2Vec(x, size=vec_size, window=13, min_count=1, workers=4)

print(model.corpus_count)
print(model.wv.most_similar(positive=['france', 'russia'], negative=['paris']))
