#This is the simple implementation of Stock price prediction using LSTM.
#Steps are pretty simple, Extract columns on which we want to train model and make predictions.
#Normalize the values and process data into 60 time steps.
#As you can see here, there's no validation data because i have already checked model performance on train and validation data so now i trained model on whole dataset.
#Used Model checkpoint to save best model and load it to make prediction on test dataset.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input,LSTM,Dense,Bidirectional,Dropout,Flatten
from sklearn.model_selection import train_test_split
from keras.models import Model,load_model
from keras.optimizers import Adam,RMSprop
from keras.callbacks import ModelCheckpoint,TensorBoard,ReduceLROnPlateau

#Read Data
df=pd.read_csv('/home/anubhav/Downloads/Stock Dataset/GOOG.csv')

#Choose columns that are going to be used for training
colums=['Open','High','Low','Close']
data=df[colums].values

#Split Dataset
#train_data,val_data=train_test_split(data,test_size=0.2,shuffle=False)

#Normalize values
scaler=MinMaxScaler()
train_data=scaler.fit_transform(data)

#Steps are choosen 60.
Steps=60

#Process data and transform it into required format
#Training Data
train_X,train_y=[],[]
for i in range(Steps,len(train_data)):
    train_X.append(train_data[i-Steps:i,])
    train_y.append(train_data[i])

train_X=np.array(train_X)
train_y=np.array(train_y)

# =============================================================================
# 
# #Valdation Data
# val_data=scaler.transform(val_data)
# val_X,val_y=[],[]
# for i in range(Steps,len(val_data)):
#     val_X.append(val_data[i-Steps:i,])
#     val_y.append(val_data[i])
# 
# val_X=np.array(val_X)
# val_y=np.array(val_y)
# =============================================================================



#Callbacks
mc=ModelCheckpoint('/home/anubhav/Documents/output/stock_predictor.h5',monitor='loss',
                   period=1,mode='min',save_best_only=True)
rop=ReduceLROnPlateau(monitor='loss',factor=0.1,patience=9,verbose=1)


#Model
inp=Input(shape=(Steps,4))
x=LSTM(100,return_sequences=True)(inp)
x=LSTM(50)(x)
fc1=Dense(256,activation='relu')(x)
x=Dropout(0.4)(x)
fc2=Dense(100,activation='relu')(x)
out=Dense(4,activation=None)(x)
model=Model(inp,out)
model.summary()

model.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=0.01))
model.fit(train_X,train_y,epochs=100,batch_size=64,callbacks=[mc])



#Test
#Load Pretrained model
model=load_model('/home/anubhav/Documents/Stock Prediction/stock_predictor.h5') #Load Checkpoint model

#Test
#Load Test Data
test_data=pd.read_csv('/home/anubhav/Downloads/Stock Dataset/GOOG_test.csv')
colums=['Open','High','Low','Close']
test_data=test_data[colums].values

#Normalize
scaler=MinMaxScaler()
test_data=scaler.fit_transform(test_data)

Steps=60

#Training Data
test_X,test_y=[],[]
for i in range(Steps,len(test_data)):
    test_X.append(test_data[i-Steps:i,])
    test_y.append(test_data[i])

test_X=np.array(test_X)
test_y=np.array(test_y)

#Predict Values
preds=model.predict(test_X)

#Transform values back to original
preds_inv=scaler.inverse_transform(preds)
val_y_iv=scaler.inverse_transform(test_y)

#Plot True vs prediction values
plt.figure()
plt.title('Open_TrueVsPreds')
plt.plot(preds_inv[0:,0],label='Pred')
plt.plot(val_y_iv[0:,0],label='True')
plt.legend()
plt.show()

plt.figure()
plt.title('High_TrueVsPreds')
plt.plot(preds_inv[0:,1],label='Pred')
plt.plot(val_y_iv[0:,1],label='True')
plt.legend()
plt.show()

plt.figure()
plt.title('Low_TrueVsPreds')
plt.plot(preds_inv[0:,2],label='Pred')
plt.plot(val_y_iv[0:,2],label='True')
plt.legend()
plt.show()

plt.figure()
plt.title('Close_TrueVsPreds')
plt.plot(preds_inv[0:,3],label='Pred')
plt.plot(val_y_iv[0:,3],label='True')
plt.legend()
plt.show()


    
