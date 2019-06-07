import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(42)
look_back = 30
back_step = 1
ma_price_winsize = 8
train_size = 0.9
epochs = 100
# tolerance_angle = 0.2


def open_data(company_name):
    row_data = pd.read_csv('Data/' + company_name + '.csv')[0:60000]
    data = pd.DataFrame()
    data['price'] = (row_data['<OPEN>'] + row_data['<HIGH>'] + row_data['<LOW>'])/3
    return data


def split_data(dataset, look_back=look_back):
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
    for i in range(1,len(data)):
        d_n[i] = (data.loc[i])/data.loc[i-1]
    d_n[0] = 1
    m = d_n.mean()
    std = d_n.std()
    d_n = (d_n - m) / std
    m_c = (1 - m) / std
    # m_c = 1
    d = pd.DataFrame(d_n, columns=['price'])
    return d, m_c


def z_score(data):
    data = (data - data.mean())/data.std()
    return data


def check_ans(x, y, ans, const):
    x = x[:,look_back-1,0]
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

    recall = tp/(tp + fn + 0.01)
    precisson = tp / (tp + fp + 0.01)
    f1 = 2*recall*precisson/(recall + precisson + 0.01)
    c = c/len(y)
    # plt.plot(ans, y, '.')
    # plt.plot(ans)
    # plt.plot(y)
    # plt.show()
    print('precisson:', precisson)
    print('recall:', recall)
    return f1, c, p, n


data = open_data('CHMF_d')
data = get_ma(data, ma_price_winsize)
data, c = normilize_data(data)
x, y = split_data(np.array(data))
x = np.reshape(x, [x.shape[0], x.shape[1], 1])

trainX, testX = x[0:int(train_size * len(x))], x[int(train_size * len(x)):len(x)]
trainY, testY = y[0:int(train_size * len(y))], y[int(train_size * len(y)):len(y)]

data2 = open_data('NLMK_d')
data2 = get_ma(data2, ma_price_winsize)
data2, c2 = normilize_data(data2)
x2, y2 = split_data(np.array(data2))
x2 = np.reshape(x2, [x2.shape[0], x2.shape[1], 1])

trainX2, testX2 = x2[0:int(train_size * len(x2))], x2[int(train_size * len(x2)):len(x2)]
trainY2, testY2 = y2[0:int(train_size * len(y2))], y2[int(train_size * len(y2)):len(y2)]

data3 = open_data('SNGS_d')
data3 = get_ma(data3, ma_price_winsize)
data3, c3 = normilize_data(data3)
x3, y3 = split_data(np.array(data3))
x3 = np.reshape(x3, [x3.shape[0], x3.shape[1], 1])

trainX3, testX3 = x3[0:int(train_size * len(x3))], x3[int(train_size * len(x3)):len(x3)]
trainY3, testY3 = y3[0:int(train_size * len(y3))], y3[int(train_size * len(y3)):len(y3)]


n_features = 1
n_steps = trainX.shape[1]
# define model
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mape')
# fit model
# model.fit(trainX, trainY, epochs=1000, verbose=0, validation_data=(testX, testY))
# model.load_weights('w_lo3.h5')

model.fit(trainX, trainY, epochs=epochs, verbose=2, batch_size=8, validation_data=(testX, testY))
model.fit(trainX3, trainY3, epochs=epochs, verbose=2, batch_size=8, validation_data=(testX3, testY3))
model.fit(trainX2, trainY2, epochs=epochs, verbose=2, batch_size=8, validation_data=(testX2, testY2))
model.save_weights('w_cnn_d1.h5')
# model.load_weights('w_lo_d1.h5')

# ans = model.predict(trainX)
# score = check_ans(trainX, trainY, ans, c)
# print(score)
# print(r2_score(trainY, ans))
ans = model.predict(testX)
score = check_ans(testX, testY, ans, c)
print(score)
print(r2_score(testY, ans))


# ans = model.predict(trainX2)
# score = check_ans(trainX2, trainY2, ans, c2)
# print(score)
# print(r2_score(trainY, ans))
ans = model.predict(testX2)
score = check_ans(testX2, testY2, ans, c2)
print(score)
print(r2_score(testY2, ans))


# ans = model.predict(trainX3)
# score = check_ans(trainX3, trainY, ans, c3)
# print(score)
# print(r2_score(trainY3, ans))
ans = model.predict(testX3)
score = check_ans(testX3, testY3, ans, c3)
print(score)
print(r2_score(testY3, ans))


