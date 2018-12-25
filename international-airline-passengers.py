import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
dataframe=pd.read_csv('G:\\python_work\\international-airline-passengers.csv',usecols=[1],engine='python',skipfooter=3)
dateset=dataframe.values
dateset=dateset.astype('float32')
'''
将一列变成两列，第一列是 t 月的乘客数，第二列是 t+1 列的乘客数。 look_back 就是预测下一步所需要的 time steps：
timesteps 就是 LSTM 认为每个输入数据与前多少个陆续输入的数据有联系。
例如具有这样用段序列数据 “…ABCDBCEDF…”，当 timesteps 为 3 时，在模型预测中如果输入数据为“D”，
那么之前接收的数据如果为“B”和“C”则此时的预测输出为 B 的概率更大，之前接收的数据如果为“C”和“E”，
则此时的预测输出为 F 的概率更大。
'''
def create_dateset(dateset,look_back):
    datax,datay=[],[]
    for i in range(len(dateset)-look_back-1):
        a=dateset[i:(i+look_back),0]
        datax.append(a)
        datay.append(dateset[i+look_back,0])
    return np.array(datax),np.array(datay)
np.random.seed(7)
#当激活函数为 sigmoid 或者 tanh 时，要把数据正则话，此时 LSTM 比较敏感 ,设定 67% 是训练数据，余下的是测试数据
scaler=MinMaxScaler(feature_range=(0,1))
dateset=scaler.fit_transform(dateset)
train_size=int(len(dateset)*0.67)
test_size=len(dateset)-train_size
train,test=dateset[0:train_size,:],dateset[train_size:len(dateset),:]
#X=t and Y=t+1 时的数据，并且此时的维度为 [samples, features]
look_back=3
trainx,trainy=create_dateset(train,look_back)
testx,testy=create_dateset(test,look_back)
#投入到 LSTM 的 X 需要有这样的结构： [samples, time steps, features]，所以做一下变换
trainx=np.reshape(trainx,(trainx.shape[0],1,trainx.shape[1]))
testx=np.reshape(testx,(testx.shape[0],1,testx.shape[1]))
print(trainx,trainy)
#建立 LSTM 模型： 输入层有 1 个input，隐藏层有 4 个神经元，输出层就是预测一个值，激活函数用 sigmoid，迭代 100 次，batch size 为 1
model=Sequential()
model.add(LSTM(128,input_shape=(1,look_back)))
model.add(Dropout(0.5))
model.add(Dense(1))
#model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainx,trainy,epochs=100,batch_size=1,verbose=2)
#预测
trainpredict=model.predict(trainx)
testpredict=model.predict(testx)
#计算误差之前要先把预测数据转换成同一单位
trainpredict=scaler.inverse_transform(trainpredict)
trainy=scaler.inverse_transform([trainy])
testpredict=scaler.inverse_transform(testpredict)
testy=scaler.inverse_transform([testy])
#计算 mean squared erro
trainscore=math.sqrt(mean_squared_error(trainy[0],trainpredict[:,0]))
print('Train Score: %.2f RMSE' % (trainscore))
testscore=math.sqrt(mean_squared_error(testy[0],testpredict[:,0]))
print('Test Score: %.2f RMSE' % (testscore))
#画出结果
trainpredictplot=np.empty_like(dateset)
trainpredictplot[:,:]=np.nan
trainpredictplot[look_back:len(trainpredict)+look_back,:]=trainpredict
testpredictplot=np.empty_like(dateset)
testpredictplot[:,:]=np.nan
testpredictplot[len(trainpredict)+(look_back*2)+1:len(dateset)-1,:]=testpredict
plt.plot(scaler.inverse_transform(dateset))
plt.plot(trainpredictplot)
plt.plot(testpredictplot)
plt.show()



