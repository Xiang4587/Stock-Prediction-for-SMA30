
# coding: utf-8

# In[100]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[101]:


DF = pd.read_csv("SPY.csv")
DF.head()


# ## Data preprocessing

# ### Feature Augumentation

# In[102]:


def featureAug(df):
    dt = pd.to_datetime(df["Date"])
    Y = []
    M = []
    date = []
    day = []
    for i in range(len(dt)):
        Y.append(dt[i].year)
        M.append(dt[i].month)
        date.append(dt[i].day)
        day.append(dt[i].weekday())
    df['Year'] = Y
    df['Month'] = M
    df['date'] = date
    df['Day'] = day 
    return df


# In[229]:


DF_aug = featureAug(DF)
DF_aug.head()


# ### Normalization

# In[230]:


def normalize(df):
    df = df.drop(["Date"], axis=1)
    df_norm = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return df_norm


# In[231]:


DF_re = normalize(DF_aug)
DF_re.head()


# In[232]:


len(DF_re)


# ### Simple Moving Average for 30 days

# In[233]:


def MovingAvg(df, size):
    i = 0
    moving_avg = []
    while i < len(df)-size+1:
        this_size = df[i:i+size]
        avg = sum(this_size)/size
        moving_avg.append(avg)
        i += 1
    return moving_avg


# In[234]:


DF_rrr = DF_re.drop([i for i in range(29)])
DF_rrr = DF_rrr.reset_index()


# In[235]:


SMA30 = MovingAvg(DF_re['Close'], 30)
DF_rre = DF_re[29:].reset_index()
DF_rre['SMA30'] = SMA30


# In[333]:


print("Data processed:", len(DF_rre))
DF_rre.head()


# In[337]:


figure, ax = plt.subplots()
plt.plot(DF_rre['Close'], label='Close')
plt.plot(DF_rre['SMA30'], label='SMA30')
figure.set_size_inches(8, 4)
ax.legend(('Close', 'SMA30'), loc='best', framealpha=0.25, prop={'size': 'small', 'family': 'monospace'})

ax.set_title('Trend')
ax.set_xlabel('Date')
ax.set_ylabel('value')

ax.grid(True)
figure.tight_layout()


# In[326]:


k = np.array(DF_re.iloc[0:30])


# In[239]:


k.shape


# ### Using the past 30 days as features to predict the T+29 SMA

# #### Features within 30 days include: Open, High, Low, Close, Adj Close, Volume, Year, Month, date, and Day

# In[240]:


DF_re.shape


# In[241]:


DF_rre.shape


# In[242]:


X = []
Y = []
for i in range(len(DF_rre)-29):
    idx = DF_rre['index'][i]
    X.append(np.array(DF_re.iloc[idx-29:idx+1]))
    Y.append(DF_rre['SMA30'][i+29])
X = np.array(X)
Y = np.expand_dims(Y, axis=1)
print("Input dim:", X.shape)
print("Output dim:", Y.shape)


# ### Split Data

# In[243]:


def split(X, Y, rate):
    total = X.shape[0]
    X_train_val = X[:int(round(total*rate))]
    Y_train_val = Y[:int(round(total*rate))]
    last = total - int(round(total*rate))
    X_test = X[-last:]
    Y_test = Y[-last:]
    return X_train_val, Y_train_val, X_test, Y_test


# In[244]:


X_train_val, Y_train_val, X_test, Y_test = split(X,Y,0.85) ## training set 70%, validation set 15%, test set 15%
X_train, Y_train, X_val, Y_val = split(X_train_val, Y_train_val, 0.7) # Split a portion as validation set
#Y_train = Y_train[:,:,np.newaxis]
#Y_val = Y_val[:,:,np.newaxis]


# In[245]:


X_train.shape, Y_train.shape


# ## Build Model

# In[288]:


def LSTM_seq(shape):
    model = Sequential()
    model.add(LSTM(units=20, return_sequences=False, input_shape=(shape[1], shape[2])))
    model.add(Dropout(0.2))
        
    # output shape: (1, 1)
    model.add(Dense(units=1))
    
    model.compile(loss="mse", optimizer="adam", metrics=['mse'])
    model.summary()
    return model


# In[289]:


model = LSTM_seq(X_train.shape)


# ## Start training

# In[290]:


# Hyperparameters
epoch = 100
batch_size = 64 
early_stopping = EarlyStopping(monitor="loss", patience=10, verbose=2, mode="auto")


# In[291]:


log = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size, 
                validation_data=(X_val,Y_val),shuffle=False,
                callbacks=[early_stopping])


# In[336]:


figure, ax = plt.subplots()
plt.plot(log.history['loss'], label='loss')
plt.plot(log.history['val_loss'], label='val_loss')
figure.set_size_inches(6, 3)
ax.legend(('loss', 'val_loss'), loc='best', framealpha=0.25, prop={'size': 'small', 'family': 'monospace'})

ax.set_title('training loss')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

ax.grid(True)
figure.tight_layout()


# In[296]:


model.save('LSTM_stock.h5')  


# ## Testing & Evaluation

# In[309]:


Y_predict = model.predict(X_test)
test_loss = model.evaluate(X_test, Y_test)


# In[312]:


print("Mean Square error:", test_loss[0])


# ## Visaulization the results

# In[329]:


fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
plt.plot(Y_test, label = 'Real SMA at T+29') 
plt.plot(Y_predict, label = 'Predicted SMA at T+29')
plt.title('30 days simple moving average')
plt.xlabel('SMA value')
plt.ylabel('Time')
plt.legend()
plt.show()

