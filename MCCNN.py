import numpy as np
import math
import gc
from keras.layers import Dense, Dropout, Flatten, Concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import netCDF4 as nc
from sklearn.preprocessing import StandardScaler
from keras import regularizers, Model, Input




def r2_score(y_true, y_pred):
    '''
    R^2 (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a R^2 score of 0.0.

    Read more in the :ref:`User Guide <r2_score>`.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    '''


    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - np.average(y_true)) ** 2).sum()
    r2=1-numerator/denominator

    return r2
def define_model(xx_train):
    #MCCNN model
    # channel 1
    In_1 = Input(shape=(xx_train.shape[1], xx_train.shape[2], 1))
    model_1 = Conv2D(filters=32,strides=1,kernel_size=4, activation='tanh')(In_1)

    model_1= MaxPooling2D(pool_size=2)(model_1)
    model_1= Conv2D(filters=8, kernel_size=4,strides=1, activation='tanh')(model_1)

    model_1 = MaxPooling2D(pool_size=2)(model_1)
    model_1= Flatten()(model_1)
    # channel 2
    In_2 = Input(shape=(xx_train.shape[1], xx_train.shape[2], 1))
    model_2 = Conv2D(filters=32, kernel_size=4, strides=1,activation='tanh')(In_2)

    model_2 = MaxPooling2D(pool_size=2)(model_2)
    model_2 = Conv2D(filters=8, kernel_size=4, strides=1,activation='tanh')(model_2)

    model_2 = MaxPooling2D(pool_size=2)(model_2)
    model_2 = Flatten()(model_2)
    # channel 3
    In_3 = Input(shape=(xx_train.shape[1],xx_train.shape[2],1)) #shape
    model_3 = Conv2D(filters=32, kernel_size=4, strides=1,activation='tanh')(In_3)

    model_3 = MaxPooling2D(pool_size=2)(model_3)
    model_3 = Conv2D(filters=8, kernel_size=4, strides=1,activation='tanh')(model_3)

    model_3 = MaxPooling2D(pool_size=2)(model_3)
    model_3 = Flatten()(model_3)
    # channel 4
    In_4 = Input(shape=(xx_train.shape[1],xx_train.shape[2],1))#shape
    model_4 = Conv2D(filters=32, kernel_size=4, strides=1,activation='tanh')(In_4)

    model_4 = MaxPooling2D(pool_size=2)(model_4)
    model_4 = Conv2D(filters=8, kernel_size=4, strides=1,activation='tanh')(model_4)

    model_4 = MaxPooling2D(pool_size=2)(model_4)
    model_4 = Flatten()(model_4)
    # channel 5
    In_5 = Input(shape=(xx_train.shape[1],xx_train.shape[2],1)) #shape
    model_5 = Conv2D(filters=32, kernel_size=4, strides=1,activation='tanh')(In_5)

    model_5 = MaxPooling2D(pool_size=2)(model_5)
    model_5 = Conv2D(filters=8, kernel_size=4, strides=1,activation='tanh')(model_5)

    model_5 = MaxPooling2D(pool_size=2)(model_5)
    model_5 = Flatten()(model_5)
    # channel 6
    In_6 = Input(shape=(xx_train.shape[1],xx_train.shape[2],1)) #shape(model_6)
    model_6 = Conv2D(filters=32, kernel_size=4, strides=1,activation='tanh')(In_6)

    model_6 = MaxPooling2D(pool_size=2)(model_6)
    model_6 = Conv2D(filters=8, kernel_size=4, strides=1,activation='tanh')(model_6)

    model_6 = MaxPooling2D(pool_size=2)(model_6)
    model_6 = Flatten()(model_6)
    # channel 7
    In_7 = Input(shape=(xx_train.shape[1],xx_train.shape[2],1))#shape(model_7)
    model_7 = Conv2D(filters=32, kernel_size=4, strides=1,activation='tanh')(In_7)

    model_7 = MaxPooling2D(pool_size=2)(model_7)
    model_7 = Conv2D(filters=8, kernel_size=4, strides=1,activation='tanh')(model_7)

    model_7 = MaxPooling2D(pool_size=2)(model_7)
    model_7 = Flatten()(model_7)
    #combine
    merged = Concatenate()([model_1, model_2, model_3,model_4,model_5,model_6,model_7]) #merged
    dense1 = Dense(256, activation='tanh',use_bias=True,kernel_regularizer= regularizers.l1(0.01)) (merged)# interpretation
    output = Dense(1,use_bias=True,kernel_regularizer= regularizers.l1(0.01)) (dense1)
    model = Model(inputs=[In_1,In_2,In_3,In_4,In_5,In_6,In_7], outputs=output)
    # compile
    adam1 = Adam(learning_rate=0.001)
    model.compile(loss='mse',optimizer=adam1,metrics=['mse'])
    return model


#import files of marine topography and residual DOVs
Data=nc.Dataset('top.nc') # marine topography
h_gw= (Data.variables["elevation"][:].data).T
Data2=nc.Dataset('e.nc')#residual DOV for east(altimeter-derived DOV-reference DOV)
Data3=nc.Dataset('n.nc')#residual DOV for north(altimeter-derived DOV-reference DOV)
e_gw=(Data2.variables["z"][:].data).T
n_gw=(Data3.variables["z"][:].data).T

#import ship-borne data for training
ifile='train0.dat'
# Z1 training file: lon,lat,residual gravity anomaly (ship-borne -reference)
Z1 = np.loadtxt(ifile)
y_train = Z1[:, 3]  # residual gravity anomaly for train
#inputs for train
x_train = np.zeros((len(Z1[:, 1]), 64, 64, 7))
for i in range(0,len(Z1[:,1])):
    lon_num_min=math.floor((Z1[i,0]-124.002)*60.0*4.0)-31
    lon_num_max=math.floor((Z1[i,0]-124.002)*60.0*4.0)+33
    lat_num_min=math.floor((Z1[i,1]-9.0021)*60.0*4.0)-31
    lat_num_max = math.floor((Z1[i, 1]-9.0021) * 60.0*4.0) +33
    for k in range(0,64):
        for l in range(0,64):
            x_train[i,  k, l,0] = lon_num_min / 60.0/4.0 + 124.002 + 1.0 / 60.0*0.25 * k
            x_train[i,  k, l,1] = lat_num_min / 60.0/4.0 + 9.0021 + 1.0 / 60.0 * 0.25*l
            x_train[i,  k, l,2] = lon_num_min / 60.0/4.0 + 124.002 + 1.0 / 60.0 * 0.25*k- Z1[i, 0]
            x_train[i,  k, l,3] = lat_num_min / 60.0/4.0 + 9.0021 + 1.0 / 60.0 *0.25* l-Z1[i, 1]
    x_train[i, 0:64, 0:64,6] = h_gw[lon_num_min:lon_num_max, lat_num_min:lat_num_max]
    x_train[i, 0:64, 0:64,5] = e_gw[lon_num_min:lon_num_max, lat_num_min:lat_num_max]
    x_train[i, 0:64, 0:64,4] = n_gw[lon_num_min:lon_num_max, lat_num_min:lat_num_max]
del h_gw
del e_gw
del n_gw
gc.collect()

#inputs are standardized by removing the mean and scaling them to unit variance in each channel
mean_train = np.zeros((7))
std_train = np.zeros((7))
scaler = StandardScaler()
for i in range(0,7):
    mean_train[i]=np.mean(x_train[:,:,:,i])
    std_train[i]=np.std(x_train[:,:,:,i])
    x_train[:,:,:,i]=(x_train[:,:,:,i]-mean_train[i])/std_train[i]
    print(mean_train[i],std_train[i])

#training model
print('training~~~')
model=define_model(x_train)
deta=0.02
model.fit([x_train[:,:,:,0],x_train[:,:,:,1],x_train[:,:,:,2],x_train[:,:,:,3],x_train[:,:,:,4],x_train[:,:,:,5],x_train[:,:,:,6]],y_train,batch_size=512, shuffle=True,validation_split=0.05,epochs=20, callbacks=[EarlyStopping(monitor='mse',min_delta=deta,patience=3)])

# calculate the r2_score for train
score_train = r2_score(y_train, np.transpose(model.predict([x_train[:,:,:,0],x_train[:,:,:,1],x_train[:,:,:,2],x_train[:,:,:,3],x_train[:,:,:,4],x_train[:,:,:,5],x_train[:,:,:,6]])))
print(score_train)

#save CNN model
model.save('CNN20.h5')




