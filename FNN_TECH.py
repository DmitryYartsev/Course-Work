import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


np.random.seed(42)
look_back = 50
back_step = 1
ma_price_winsize = 3
train_size = 0.8
epochs = 15


def open_data(company_name):
    row_data = pd.read_csv('Data/' + company_name + '.csv')
    data = pd.DataFrame()
    data['price'] = (row_data['<OPEN>'] + row_data['<HIGH>'] + row_data['<LOW>'])/3
    return data


def split_data(dataset, look_back = look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def get_ma(data, w_size):
    d_ma = np.zeros([len(data)-w_size])
    for i in range(len(data)-w_size):
        d_ma[i] = (data[i:w_size+i].sum())/w_size
    d = pd.DataFrame(d_ma, columns=['price'])
    return d


def get_dis(data):
    d = pd.DataFrame(columns=['i','x','y'])
    d['y'] = 1
    d['x'] = 0
    ct = 0
    c = 0
    ci = 0
    for i in range(1,len(data)-1):
        c = c + 1
        if(np.sign(data['price'].iloc[i]-data['price'].iloc[i-1]) != np.sign(data['price'].iloc[i+1]-data['price'].iloc[i])) or i == len(data) - 2:
            ci = ci + 1
            dx = c - ct
            ct = c
            dy = data['price'].iloc[i]/data['price'].iloc[i-1] - 1
            td = pd.DataFrame({'i':[ci],'x':[dx],'y':[dy]})
            d = d.append(td)

    d = d.set_index('i')
    return d


def normilize_data(data):
    d_n = np.zeros([len(data)])
    d_n[0] = 1
    for i in range(1,len(data)):
        d_n[i] = (data.loc[i])/data.loc[i-1]
    m = d_n.mean()
    std = d_n.std()
    d_n = (d_n - m)/std
    m_c = (1 - m)/std
    print(m_c, d_n.mean(), d_n.std())
    d = pd.DataFrame(d_n, columns=['price'])
    return d, m_c


def z_score(data):
    data = (data - data.mean())/data.std()
    return data


def check_ans(x, y, ans, const):
    c = 0
    tp = 0
    fp = 0
    fn = 0
    p = 0
    n = 0
    for i in range(len(x)):
        if np.sign(y[i]-const) == np.sign(ans[i]-const):
            c = c + 1
        if (np.sign(y[i] - const) < 0):
            n = n + 1
            if (np.sign(ans[i] - const) > 0):
                   fp = fp + 1
        if (np.sign(y[i] - const) > 0):
            p = p + 1
            if (np.sign(ans[i] - const) > 0):
                   tp = tp + 1
            else: fn = fn + 1

    recall = tp / (tp + fn + 0.01)
    precisson = tp / (tp + fp + 0.01)
    f1 = 2*recall*precisson/(recall + precisson + 0.01)
    c = c/len(y)
    # plt.plot(ans, y, '.')
    plt.plot(y)
    plt.plot(ans)

    plt.show()
    print('precisson:', precisson)
    print('recall:', recall)
    return f1, c, p, n


data = open_data('NLMK_d')
data = get_ma(data, ma_price_winsize)
data, m_c = normilize_data(data)
data = z_score(data)
x, y = split_data(np.array(data))
print(x.shape)

trainX, testX = x[0:int(train_size * len(x))], x[int(train_size * len(x)):len(x)]
trainY, testY = y[0:int(train_size * len(y))], y[int(train_size * len(y)):len(y)]


model = Sequential()
model.add(Dense(100, input_dim=x.shape[1], activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
model.compile(loss='mape', optimizer='adam')
model.fit(trainX, trainY, epochs=epochs, verbose=2, batch_size=8, validation_data=(testX, testY))
model.save_weights('fnn_1.h5')

ans = model.predict(testX)
print(r2_score(testY, ans))

score = check_ans(testX, testY, ans, m_c)
print(score)

