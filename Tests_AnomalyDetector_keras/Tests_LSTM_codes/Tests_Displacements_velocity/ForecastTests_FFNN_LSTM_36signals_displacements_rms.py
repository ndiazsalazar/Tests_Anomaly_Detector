#!/usr/bin/env python3

'''
Script to study the forecast problem using a simple feedforward neural network and an LSTM neural network.
The test functions are a simple cosine function, a straight line, random noise and an anomaly in the form of an arctan function
'''


#### To run in the cluster these following lines have to be decommented.
"""
# Fix issue with OMP
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Fix issue with matplotlib
import matplotlib
matplotlib.use('agg')
"""
#########################################


#########################################
# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
import sklearn.preprocessing as prep

# Import keras libraries
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import load_model

#import forecast_module as fm
import h5py
import numpy as np
import os
from itertools import combinations

import pandas as pd
###########################################
# Functions

def Secular(t, m ):
    d_secular = 0.0 + m * (t - t[0])
    return d_secular


def Seasonal(t, amplitude = 1.0, periodo = 1.0 ):
    # Seasonal signal (annual + semiannual)
    d_seasonal = amplitude * (1.0 * np.cos(periodo * 2.0 * np.pi * (t - 0.25)) -
                  2.5 * np.cos(periodo * 2.0 * np.pi / 0.5 * (t - 0.75)))
    return d_seasonal

def Noise(Nt, amplitude = 1.0):
    np.random.seed(0)
    d_noise = amplitude * np.random.randn(Nt)
    return d_noise


def Transient(t, n_start = 1260, lenght = 0.02):
    n = n_start #np.int((start_time-t[0])*365)
    print('t[n]', t[n])

    shift = -1.8  #arctan displaced in shift
    transient1 = np.zeros(n)
    #lenght = 0.02
    disp = np.arctan((shift) / lenght)
    transient2 = -20.0 * (np.arctan((t[n:] - t[n] + shift) / lenght) - disp)
    d_transient = np.concatenate([transient1, transient2])
    return d_transient


def FeedForward(Nneurons, batch_size, epochs, trainX, trainY, input_dim, outputnumber = 1):
    model = Sequential()

    # Create Dense layer with Nneurons
    model.add(Dense(Nneurons, input_dim=input_dim))

    # Create the output layer with 1 output
    model.add(Dense(outputnumber))
    opt = Adam(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()
    #plot_model(model, to_file='modelSimple.png', show_shapes=True)
    #ann_viz(model, title='model')

    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.25)

    return model




def LSTM_NeuralNetwork(Nunits, batch_size, epochs, trainX, trainY, input_shape, outputdim=1):

    model = Sequential()
    LSTMobject = LSTM(Nunits, input_shape=input_shape)
    model.add(LSTMobject)

    #model.add(Dropout(0.3))
    model.add(Dense(outputdim))
    opt = Adam(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()
    model.fit(trainX, trainY, epochs = epochs, batch_size = batch_size, verbose = 0, validation_split=0.25)
    return model


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def create_dataset2(dataset, look_back=1, steps_ahead = 5):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-steps_ahead+1):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i + look_back: i + look_back + steps_ahead, 0])
    return np.array(dataX), np.array(dataY)








def plot_signal(T_original, dataset, trainPredictPlot, testPredictPlot, original, scaler, key, title='', line_width=1, alpha=1, color='k', subplots=False, show_grid=True, fig_size=(12, 5), transient = None, t_transient = None, velocity=True):

    # Plot 1: predicted vs original data
    fig = plt.figure(figsize=fig_size)

    fig.suptitle(title)
    #####subplot1
    ax1=plt.subplot(3, 1, 1)
    plt.plot(T_original, original, '.-b', label='Original data', linewidth=0.4, markersize=0.8)
    ax1.tick_params(labelbottom=False)


    if 'transient' in key:
        plt.plot(T_original, transient, '.-', color='darkorange', label='transient', linewidth=0.4, markersize=0.8)

    plt.legend()

    ##### subplot2
    ax2 = plt.subplot(3, 1, 2)

    if velocity:
        plt.plot(T_original[1:], scaler.inverse_transform(dataset), '.-b', label='Original data', linewidth=0.4, markersize=0.8)
        plt.plot(T_original[1:], scaler.inverse_transform(trainPredictPlot), '.r', label='Forecast on the training set', markersize=0.8)
        plt.plot(T_original[1:], scaler.inverse_transform(testPredictPlot), '.g', label='Forecast on the testing set', markersize=0.8)
        plt.legend()

    else:

        plt.plot(T_original, scaler.inverse_transform(dataset), '.-b', label='Original data', linewidth=0.4, markersize=0.8)
        plt.plot(T_original, scaler.inverse_transform(trainPredictPlot), '.r', label='Forecast on the training set', markersize=0.8)
        plt.plot(T_original, scaler.inverse_transform(testPredictPlot), '.g', label='Forecast on the testing set', markersize=0.8)
        plt.legend()


    if 'transient' in key:

        plt.plot((t_transient, t_transient), plt.gca().get_ylim(), '--', color='orange', lw=2,
                 label='Start time of the transient (' + str(np.round(t_transient, 1)) + ')')
        plt.axvspan(t_transient, T_original[-1], color='orange', alpha=0.25)


    #plt.title(title)
    #plt.xlabel('time [years]')
    plt.legend()

    #ax2.axes.get_xaxis().set_visible(False)
    ax2.tick_params(labelbottom=False)

    ######subplot3
    plt.subplot(3, 1, 3)

    errorTrain = scaler.inverse_transform(dataset) - scaler.inverse_transform(trainPredictPlot)
    errorTest = scaler.inverse_transform(dataset) - scaler.inverse_transform(testPredictPlot)

    rmsTrain = np.sqrt(errorTrain ** 2)
    rmsTest = np.sqrt(errorTest ** 2)



    if velocity:
        plt.plot(T_original[1:], rmsTrain, '.-r', label='rms training set',linewidth=0.4, markersize=2.5)
        plt.plot(T_original[1:], rmsTest, '.-g', label='rms test set',linewidth=0.4, markersize=2.5)

    else:
        plt.plot(T_original, rmsTrain, '.-r', label='rms training set', linewidth=0.4, markersize=2.5)
        plt.plot(T_original, rmsTest, '.-g', label='rms test set', linewidth=0.4, markersize=2.5)

    if 'transient' in key:

        plt.plot((t_transient, t_transient), plt.gca().get_ylim(), '--', color='orange', lw=2,
                 label='Start time of the transient (' + str(np.round(t_transient, 1)) + ')')
        plt.axvspan(t_transient, T_original[-1], color='orange', alpha=0.25)


    plt.ylabel('rms ')
    plt.xlabel('time [years]')
    plt.ticklabel_format(useOffset=False)
    plt.legend()

    #plt.savefig(folderN + 'Pred_Diff' + key + '_FFNN.png', dpi=900, bbox_inches='tight', pad_inches=0.05)
    #case_number += 1


    return fig








###########################################
# Code

# First, create time vector (units of years)
Nt = 5475
t = np.linspace(2000.0, 2014.0, Nt)
startyear = 2000
endyear = 2014


m = 1

d_secular = Secular(t, m )
d_seasonal = Seasonal(t)
d_noise = Noise(Nt)

n_startT = 4470
d_transient = Transient(t, n_start = n_startT, lenght=0.2)

n_startT2 = n_startT - 200
d_transient2 = Transient(t, n_start = n_startT2 , lenght=0.5)

t_transient = t[n_startT]
t_transient2 = t[n_startT2]

print('t_transient', t_transient)


Dsyn = {}
Dsyn['seasonal'] = d_seasonal
Dsyn['secular'] = d_secular
Dsyn['transient1'] = d_transient
Dsyn['noise'] = d_noise


#tapering signal 1
d_transient[n_startT:] = 3 * d_transient[n_startT:]
ntaper=45
for i in range(len(d_transient) - n_startT):
    d_transient[n_startT + i] = np.mean(d_transient[n_startT - ntaper + i : n_startT + i + 1])


#tapering signal2
d_transient2[n_startT2:] = 2 * d_transient2[n_startT2:]
ntaper=45
for j in range(len(d_transient2) - n_startT2):
    d_transient2[n_startT2 + j] = np.mean(d_transient2[n_startT2 - ntaper + j : n_startT2 + j + 1])



'''
# tapering transient
Signal = d_transient2
plt.figure()
plt.plot(t, Signal, '.')
plt.xlabel('Year')
plt.ylabel('Displacement')
plt.show()
plt.close()


Signald = np.diff(Signal)
plt.figure()
plt.plot(t[1:], Signald, '.')
plt.xlabel('Year')
plt.ylabel('velocity')
plt.title('Diff Synthetic signal')
plt.show()
plt.close()
'''
#############################################
# Plot the original signals and save the files
strFiles = ['seasonal', 'secular', 'transient1', 'noise']
listComb = []
nmax= 4






for i in range(nmax):
    listComb.extend(combinations(strFiles, i + 1))


SynObsSignals = {}



###################################
# Folder for plots
folder = 'FFNN_LSTM_displacement_rms_white_noise2_1stepahead' + '/'

if not os.path.exists(folder):
    os.makedirs(folder)


# Plot signals
folderS = folder + 'Synthetic_36Signals' + '/'

if not os.path.exists(folderS):
    os.makedirs(folderS)

if not os.path.exists(folderS + 'Files/'):
    os.makedirs(folderS + 'Files/')


####################################


casenumber = 0
for comps in listComb:
    Signal = np.zeros(Nt)
    print(comps)
    for key in comps:

        Signal = Signal + Dsyn[key]
        print(type(Dsyn[key]))

    str_all = '_'.join(comps)

    str_title = ' + '.join(comps)
    casenumber = casenumber + 1
    keysig = str(casenumber) + '_' + str_all
    SynObsSignals[keysig] = Signal

    np.savetxt(folderS + 'Files/' + str(casenumber) + '_Syn_'+ str_all + '.txt', Signal)

    plt.figure()
    plt.plot(t, Signal, '.')
    plt.xlabel('Year')
    plt.ylabel('Displacement')
    plt.title('Synthetic signal: ' + str_title)
    plt.savefig(folderS + str(casenumber) + '_Syn_'+ str_all + '.png', dpi = 300, format = 'png')
    #plt.show()
    plt.close()

#################################################
# More signals to the dictionary

train_size = int(Nt * 0.6)
test_size = Nt - train_size

# Seasonal 2
seasonal1 = Seasonal(t, amplitude = 1.0 )
seasonal2 = Seasonal(t, amplitude = 2.0 )

sig1 = seasonal1[:train_size]
sig2 = seasonal2[train_size-10:-10]
d_seasonal2 = np.concatenate([sig1, sig2])
SynObsSignals['16_seasonal2'] = d_seasonal2



# Seasonal 3
T3=765
correction=1 #just to make the seignal more continuous
sig1 = seasonal1[:train_size + T3]
sig2 = seasonal2[train_size+T3-correction:-1*correction]
d_seasonal3 = np.concatenate([sig1, sig2])
SynObsSignals['17_seasonal3T3'] = d_seasonal3


# seasonal 4
seasonal1 = Seasonal(t, amplitude = 1.0 )
seasonal2 = Seasonal(t, amplitude = 2.0 , periodo = 0.5)

sig1 = seasonal1[:train_size]
sig2 = seasonal2[train_size-157:-157]
d_seasonal4 = np.concatenate([sig1, sig2])
SynObsSignals['18_seasonal4'] = d_seasonal4


# Noise 2
noise1 = Noise(Nt, amplitude = 1.0)
noise2 = Noise(Nt, amplitude = 2.0)

sig1 = noise1[:train_size]
sig2 = noise2[train_size:]
d_noise2 = np.concatenate([sig1, sig2])
SynObsSignals['19_noise2'] = d_noise2


# Noise 3
noise1 = Noise(Nt, amplitude = 1.0)
noise2 = Noise(Nt, amplitude = 2.0)

sig1 = noise1[:train_size + T3]
sig2 = noise2[train_size + T3:]
d_noise3 = np.concatenate([sig1, sig2])
SynObsSignals['20_noise3T3'] = d_noise3


#linear trend cut
rect1 = Secular(t, 1 )
rect2 = Secular(t, 0.5 )
sig1 = rect1[:train_size]
sig2 = rect2[train_size:] + sig1[train_size-1]-rect2[train_size]
d_line2 = np.concatenate([sig1, sig2])


SynObsSignals['21_line2'] = d_line2


#linear trend cut 2
rect1 = Secular(t, 1 )
rect2 = Secular(t, 0.5 )
sig1 = rect1[:train_size+T3]
sig2 = rect2[train_size + T3:] + sig1[train_size+T3-1]-rect2[train_size+T3]
d_line3 = np.concatenate([sig1, sig2])
SynObsSignals['22_line3T3'] = d_line3


#linear trend cut 3
rect1 = Secular(t, 0.5 )
rect2 = Secular(t, 1.5 )
sig1 = rect1[:train_size+T3]
sig2 = rect2[train_size+T3:] + sig1[train_size+T3-1]-rect2[train_size+T3]
d_line4 = np.concatenate([sig1, sig2])
SynObsSignals['23_line4'] = d_line4


#same with noise

SynObsSignals['24_seasonal2_noise'] = d_seasonal2 + d_noise

SynObsSignals['25_seasonal3T3_noise'] = d_seasonal3 + d_noise

SynObsSignals['26_seasonal4_noise'] = d_seasonal4 + d_noise

SynObsSignals['27_line2_noise'] = d_line2 + d_noise

SynObsSignals['28_line3T3_noise'] = d_line3 + d_noise

SynObsSignals['29_transient2'] = d_transient2

SynObsSignals['30_transient2_noise'] = d_transient2 + d_noise

d_noise01 = Noise(Nt, amplitude = 0.1)
d_noise005 = Noise(Nt,amplitude = 0.05)
d_noise001 = Noise(Nt,amplitude=0.01)

SynObsSignals['31_transient2_noise01'] = d_transient2 + d_noise01
SynObsSignals['32_transient2_noise005'] = d_transient2 + d_noise005
SynObsSignals['33_transient2_noise001'] = d_transient2 + d_noise001

SynObsSignals['34_transient2_seasonal_noise01'] = d_transient2 + d_noise01 + d_seasonal
SynObsSignals['35_transient2_seasonal_noise005'] = d_transient2 + d_noise005 + d_seasonal
SynObsSignals['36_transient2_seasonal_noise001'] = d_transient2 + d_noise001 + d_seasonal


other_signals = ['16_seasonal2','17_seasonal3T3', '18_seasonal4', '19_noise2','20_noise3T3','21_line2','22_line3T3','23_line4','24_seasonal2_noise', '25_seasonal3T3_noise', '26_seasonal4_noise', '27_line2_noise', '28_line3T3_noise', '29_transient2', '30_transient2_noise']



for key in SynObsSignals.keys():
    Signal = SynObsSignals[key]
    str_title = key
    plt.figure()
    plt.plot(t, Signal, '.')
    plt.xlabel('Year')
    plt.ylabel('Displacement')
    plt.title('Synthetic signal: ' + str_title)
    plt.savefig(folderS + 'Syn_' + key + '.png', dpi=300, format='png')
    #plt.show()
    plt.close()

print(SynObsSignals.keys())

#################################################

##Save the first element of the original signal



SynObsSignalsDiff = {}
#Calculating differences
for key in SynObsSignals.keys():
    original_signal = SynObsSignals[key]
    SynObsSignalsDiff[key] = np.diff(original_signal)




#Plot new signals
for key in SynObsSignalsDiff.keys():
    Signal = SynObsSignalsDiff[key]
    str_title = key
    plt.figure()
    plt.plot(t[1:], Signal, '.')
    plt.xlabel('Year')
    plt.ylabel('velocity')
    plt.title('Diff Synthetic signal: ' + str_title)
    plt.savefig(folderS + 'DiffSyn_' + key + '.png', dpi=300, format='png')
    #plt.show()
    plt.close()


####################################################################################



folders = folder + '/' + 'Scores_VaryingNneurons/'
if not os.path.exists(folders):
    os.makedirs(folders)












##########################################################################################33
# Processing the signals using a Feedforward network


case_number = 0


stepsahead = 1
Ndays = 20
N_neurons = [5]
caso = ['11_seasonal_secular_transient1'] #'29_transient2'] ,'30_transient2_noise' ]



for key in SynObsSignalsDiff.keys():
    Signal = SynObsSignals[key]
    array_comps = key.split('_')
    str_comps = ' + '.join(array_comps[1:])
    # Rescale the observations
    signal_reshaped = Signal.reshape(Signal.size, 1)
    scaler = prep.MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(signal_reshaped)
    dataset = scaler.transform(signal_reshaped)

    # split into train and test sets
    train_size = int(len(dataset) * 0.6)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    counter=0
    scores = np.zeros(len(N_neurons))
    
    for Nneurons in N_neurons:

        look_back = Ndays
        #Nneurons = 5
        epochs = 50
        batch_size = 100
        input_dim = look_back

        folderN = folder + '/' + 'FeedForward_' + str(look_back) +'lookback_Nneurons_' + str(Nneurons) + '/'
        if not os.path.exists(folderN):
            os.makedirs(folderN)


        trainX, trainY = create_dataset2(train, look_back, steps_ahead = stepsahead)
        testX, testY = create_dataset2(test, look_back, steps_ahead = stepsahead)



        model = FeedForward(Nneurons, batch_size, epochs, trainX, trainY, input_dim, outputnumber = stepsahead)



        ######################### Plots
        # Plot the performance of the training
        print('keys', model.history.history.keys())
        plt.figure(figsize = (8, 5))
        plt.plot(model.history.history['loss'], label='mean squared error on the training set')
        plt.plot(model.history.history['val_loss'], label='mean squared error on the validation set')
        plt.legend()
        plt.title('key ' + key + ' Nneurons = ' + str(Nneurons))
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squared Error')
        plt.savefig(folderN + 'Performances_' + key + '.png' , dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close()




        # make predictions
        trainPredict = model.predict(trainX, batch_size=batch_size)
        testPredict = model.predict(testX, batch_size=batch_size)
        print('lentestpred', len(testPredict))

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        print(trainPredict.shape)
        print(trainPredictPlot.shape)
        trainPredictPlot[:, :] = np.nan

        n_trainsamples, ahead = trainPredict.shape
        n_aheads = (n_trainsamples-look_back)//ahead

        for i in range(n_aheads):
             trainPredictPlot[i * ahead + look_back: i * ahead + ahead + look_back, 0] = trainPredict[i * ahead, :]

        print(trainPredictPlot.shape)


        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan

        n_testsamples, ahead = testPredict.shape
        n_aheads_test = (n_testsamples - look_back) // ahead

        for i in range(n_aheads_test):
            testPredictPlot[train_size + i * ahead + look_back: train_size + i * ahead + ahead + look_back, 0] = testPredict[i * ahead, :]





        #Evaluate scores
        scores[counter] = model.evaluate(testX, testY, verbose=0)
        counter = counter + 1


        #Plots
        T_original = t
        original = SynObsSignals[key]

        if 'transient' in key:
            if 'transient1' in key:
                transient = d_transient
                T_transient = t_transient
            elif 'transient2' in key:
                transient = d_transient2
                T_transient = t_transient2

        else:
             transient = False
             T_transient = None


        title = 'Synthetic signal: ' + str_comps + ' , FFNN, Nneurons = ' + str(Nneurons)
        fig = plot_signal(T_original, dataset, trainPredictPlot, testPredictPlot, original, scaler, key, title = title, line_width = 1, alpha = 1, color = 'k', subplots = False, show_grid = True, fig_size = (12, 5), transient = transient, t_transient = T_transient, velocity=False)

        plt.savefig(folderN + key + '_FFNN.png', dpi=900, bbox_inches='tight', pad_inches=0.05)
        case_number += 1

        #plt.show()
        plt.close()




    ################# Loss function
    plt.figure(figsize=(12, 5))
    plt.plot(Ndays, scores, '-*b')
    plt.grid(True)
    plt.title('Loss function test set, FFNN, ' + key)
    plt.xlabel('Nneurons')
    plt.ylabel('loss function')
    plt.savefig(folders + key + '_scoresNneurons_' + '_FFNN.png', dpi=301, bbox_inches='tight', pad_inches=0.05)
    #plt.show()
    plt.close()





#####################################################################################################################
## Processing signals using LSTM

#caso=['2_secular','8_secular_transient1','11_seasonal_secular_transient1','14_secular_transient1_noise','15_seasonal_secular_transient1_noise']
case_number = 0

#stepsahead=1
#t=T
for key in SynObsSignalsDiff.keys():
    Signal = SynObsSignals[key]
    array_comps = key.split('_')
    str_comps = ' + '.join(array_comps[1:])
    # Rescale the observations
    signal_reshaped = Signal.reshape(Signal.size, 1)
    scaler = prep.MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(signal_reshaped)
    dataset = scaler.transform(signal_reshaped)

    # split into train and test sets
    train_size = int(len(dataset) * 0.6)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    counter=0
    scores = np.zeros(len(N_neurons))

    for Nunits in N_neurons:

        look_back=Ndays
        #Nunits = 5
        epochs = 50
        batch_size = 100
        input_shape = (look_back,1)


        folderN = folder + '/' + 'LSTM_' + str(Ndays) + 'lookback_Nunits_' + str(Nunits) + '/'
        if not os.path.exists(folderN):
            os.makedirs(folderN)

        trainX, trainY = create_dataset2(train, look_back, steps_ahead = stepsahead)
        testX, testY = create_dataset2(test, look_back, steps_ahead= stepsahead)

        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        model = LSTM_NeuralNetwork(Nunits, batch_size, epochs, trainX, trainY, input_shape, outputdim = stepsahead)



        ######################################
        # Plot predictions
        # Plot the performance of the training
        print('keys', model.history.history.keys())
        plt.figure(figsize = (8, 5))
        plt.plot(model.history.history['loss'], label='mean squared error on the training set')
        plt.plot(model.history.history['val_loss'], label='mean squared error on the validation set')
        plt.legend()
        plt.title('key ' + key + ', LSTM, Nunits = ' + str(Nunits))
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squared Error')
        plt.savefig(folderN + 'Performances_' + key + '.png' , dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close()


        # make predictions
        trainPredict = model.predict(trainX, batch_size=batch_size)
        testPredict = model.predict(testX, batch_size=batch_size)
        print('lentestpred', len(testPredict))


###################################################################3

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        print(trainPredict.shape)
        print(trainPredictPlot.shape)
        trainPredictPlot[:, :] = np.nan

        n_trainsamples, ahead = trainPredict.shape
        n_aheads = (n_trainsamples-look_back)//ahead

        for i in range(n_aheads):
             trainPredictPlot[i * ahead + look_back: i * ahead + ahead + look_back, 0] = trainPredict[i * ahead, :]

        print(trainPredictPlot.shape)


        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan

        n_testsamples, ahead = testPredict.shape
        n_aheads_test = (n_testsamples - look_back) // ahead

        for i in range(n_aheads_test):
            testPredictPlot[train_size + i * ahead + look_back: train_size + i * ahead + ahead + look_back, 0] = testPredict[i * ahead, :]



###################################################################

        #Evaluate scores
        scores[counter] = model.evaluate(testX, testY, verbose=0)
        counter = counter + 1

######################################################

        # Plots
        T_original = t
        original = SynObsSignals[key]

        if 'transient' in key:
            if 'transient1' in key:
                transient = d_transient
                T_transient = t_transient
            elif 'transient2' in key:
                transient = d_transient2
                T_transient = t_transient2

        else:
            transient = False
            T_transient = None

        title = 'Synthetic signal : ' + str_comps + ' , LSTM, Nneurons = ' + str(Nunits)
        fig = plot_signal(T_original, dataset, trainPredictPlot, testPredictPlot, original, scaler, key, title = title, line_width = 1, alpha = 1, color = 'k', subplots = False, show_grid = True, fig_size = (12, 5), transient = transient, t_transient = T_transient, velocity=False)


        plt.savefig(folderN + key + '_LSTM.png', dpi=900, bbox_inches='tight', pad_inches=0.05)
        case_number += 1

        plt.close()

        #######################################################




    ########## Loss function

    plt.figure(figsize=(12, 5))
    plt.plot(Ndays, scores, '-*b')
    plt.grid(True)
    plt.title('Loss function test set, LSTM, ' + key)
    plt.xlabel('Nunits')
    plt.ylabel('loss function')
    plt.savefig(folders + key + '_scoresNunits_' + '_LSTM.png', dpi=301, bbox_inches='tight', pad_inches=0.05)
    plt.close()







