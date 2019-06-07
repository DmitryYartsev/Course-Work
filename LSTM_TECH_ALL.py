import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, RNN
from sklearn.metrics import r2_score


np.random.seed(42)
look_back = 6
back_step = 1
ma_price_winsize = 10
train_size = 0.9
epochs = 100

def open_data(company_name):
    row_data = pd.read_csv('Data/' + company_name + '.csv')
    data = pd.DataFrame()
    data['date'] = row_data['<DATE>']
    data['time'] = row_data['<TIME>']
    data[company_name + '_price'] = (row_data['<OPEN>'] + row_data['<HIGH>'] + row_data['<LOW>'])/3
    data[company_name + '_volume'] = row_data['<VOL>']
    return data


def merge_data(datas):
    data = pd.DataFrame()
    for i in datas:
        if (len(data)) == 0:
            data = i
        else:
            data = data.merge(i, on=['date', 'time'])
    return data


def split_data(data):
    days = data['date'].unique()
    features = data.columns.tolist()
    features.remove('date')
    features.remove('time')
    dataX, dataY = np.array([]), np.array([])
    for i in days:
        d = data[data['date'] == i][features]
        for j in range(len(d) - look_back):
            if  dataX.size == 0:
                x = np.array(d[j:j + look_back].values)
                dataX = x
                dataX = np.reshape(dataX,[1,look_back, len(features)])
                y = np.array(d.iloc[j+look_back][['CHMF_hh_price', 'NLMK_hh_price', 'MTLRP_hh_price', 'CHEP_hh_price']].values)
                dataY = y
                dataY = np.reshape(dataY, [1,1,4])
            else:
                x = np.array(d[j:j+look_back].values)
                x = np.reshape(x,[1,look_back, len(features)])
                dataX = np.concatenate((dataX, x), axis=0)
                y = np.array(d.iloc[j+look_back][['CHMF_hh_price', 'NLMK_hh_price', 'MTLRP_hh_price', 'CHEP_hh_price']])
                y = np.reshape(y,[1, 1, 4])
                dataY = np.concatenate((dataY, y), axis=0)
    return dataX, dataY


def get_ma(data):
    features = data.columns.tolist()
    n_data = data[ma_price_winsize - 1:len(data)]
    features.remove('date')
    features.remove('time')
    features.remove('CHMF_hh_volume')
    features.remove('NLMK_hh_volume')
    features.remove('MTLRP_hh_volume')
    features.remove('CHEP_hh_volume')
    for i in features:
        d_ma = data[[i]][ma_price_winsize - 1:len(data)]
        for j in range(len(data) - ma_price_winsize):
            d_ma.iloc[j][i] = data[[i]][j:ma_price_winsize + j].sum() / ma_price_winsize
        n_data[[i]] = d_ma[[i]]

    return n_data

def normilize_data(data):
    features = data.columns.tolist()
    features.remove('date')
    features.remove('time')
    features.remove('CHMF_hh_volume')
    features.remove('NLMK_hh_volume')
    features.remove('MTLRP_hh_volume')
    features.remove('CHEP_hh_volume')
    d_n = data[features]
    # print(d_n)
    dn2 = d_n.copy()
    d_n.iloc[0] = 1
    for i in range(1,len(data)):
        d_n.iloc[i] = (dn2.iloc[i])/dn2.iloc[i-1]
    m = d_n.mean()
    std = d_n.std()
    d_n = (d_n - m) / std
    m_c = (1 - m) / std
    data[features] = d_n[features]

    vol_f = ['CHEP_hh_volume', 'MTLRP_hh_volume', 'NLMK_hh_volume', 'CHMF_hh_volume']
    d_n = data[vol_f]
    m = d_n.mean()
    std = d_n.std()
    d_n = (d_n - m) / std
    data[vol_f] = d_n[vol_f]

    return data, m_c

def check_ans(y, ans, const):
    c = 0
    tp = 0
    fp = 0
    fn = 0
    p = 0
    n = 0
    for i in range(len(y)):
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
    print('f1:', f1)
    print('accuracy:', c)

# data_CHMF = open_data('CHEP_hh')
# data_NLMK = open_data('MTLRP_hh')
# data_MTLRP = open_data('NLMK_hh')
# data_TRMK = open_data('CHMF_hh')
#
# full_data = merge_data([data_CHMF, data_NLMK, data_MTLRP, data_TRMK])
#
# full_data = get_ma(full_data)
#
# full_data, mc = normilize_data(full_data)
#
# full_data = full_data.set_index('date')
# full_data.to_csv('data_hh.csv')
# data = full_data
data = pd.read_csv('data_hh.csv')
dx, dy = split_data(data)
np.save('x', dx)
np.save('y', dy)

dx = np.load('x.npy')
dy = np.load('y.npy')
dy = np.reshape(dy, [int(len(dy)), 4])


trainX, testX = dx[0:int(train_size * len(dx))], dx[int(train_size * len(dx)):len(dx)]
trainY, testY = dy[0:int(train_size * len(dy))], dy[int(train_size * len(dy)):len(dy)]


model = Sequential()
model.add(LSTM(60, input_shape=(look_back, 8), return_sequences=True))
model.add(LSTM(20))
model.add(Dense(20, activation='relu'))
model.add(Dense(4))
model.compile(loss='mape', optimizer='adam', metrics=['mape'])
model.fit(trainX, trainY, epochs=epochs, verbose=2, validation_data=(testX, testY))
model.save_weights('walo.h5')
ans = model.predict(trainX)

for i in range(4):
    print(r2_score(trainY[:, i], ans[:, i]))

plt.plot(ans[:,0])
plt.plot(trainY[:,0])
plt.show()
