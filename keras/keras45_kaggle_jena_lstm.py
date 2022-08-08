<<<<<<< HEAD
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import tensorflow as tf


#1. 데이터
path = './_data/kaggle_jena/'
dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv', 
                        index_col=0)

print(dataset.shape)    # (420551, 14)
print(dataset.describe)



import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import tensorflow as tf
print(tf.__version__)   # 2.8.2


#1. 데이터
path = './_data/kaggle_jena/'
dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv')

print(dataset.shape)    # (420551, 14)
print(dataset.describe)

# 데이터 describe & visualizing
titles = ['Pressure', 'Temperature', 'Temperature', 'Temperature in Kelvin',
         'Temperature(dew point)', 'Relative Humidity', 'Saturation vapor pressure',
         'Vapor pressure', 'Vapor pressure deficit', 'Specific humidity',
         'Water vapor concentration', 'Airtight', 'Wind speed', 'Maximum wind speed',
         'Wind direction in degrees']

feature_keys = ["p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
                "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
                "wv (m/s)", "max. wv (m/s)", "wd (deg)"]

colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

date_time_key = 'Date Time'

def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(           # subplots로 그래프 한번에 보기                                  
        nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor='w',
        edgecolor='k'
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax = axes[i // 2, i % 2],   # axes는 plot으로 생각하는 하나의 그래프
            color = c,
            title = '{} - {}'.format(titles[i], key),
            rot=25
            )
        ax.legend([titles[i]])
        plt.tight_layout
        
show_raw_visualization(dataset)

def show_heatmap(data):     #두 변수들의 조합인 heatmap으로 확인
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()

show_heatmap(dataset)

# 데이터 split
split_fraction = 0.715
train_split = int(split_fraction * int(dataset.shape[0]))

step = 6
past = 720
future = 72
learning_rate = 0.001

# MinMaxscale
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

print(
    "The selected parameters are:",
    ", ".join([titles[i] for i in [0, 1, 5, 7, 8, 10, 11]]),
)

# 분석을 위한 feature selection
selected_features = [feature_keys[i] for i in [0, 1, 5, 7, 8, 10, 11]]
features = dataset[selected_features]
features.index = dataset[date_time_key]
features.head()

features = normalize(features.values, train_split)
features = pd.DataFrame(features)
features.head()

train_data = features.loc[0 : train_split - 1]
val_data = features.loc[train_split:]

#train, validation 데이터
#train 데이터
start = past + future
end = start + train_split

x_train = train_data[[i for i in range(7)]].values 
y_train = features.iloc[start:end][[1]] # pandas iloc함수는 행번호를 통해 행 데이터를 가져옴
print('train size : ', len(x_train))    # train size :  300693

sequence_length = int(past/step)
print('window size : ', sequence_length)    # window size :  120

dataset_train = tf.keras.preprocessing.timeseries_dataset_from_array(
    x_train, y_train, sequence_length=sequence_length,
    sampling_rate=step, batch_size=128
) 

# validation 데이터
x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end][[i for i in range(7)]].values
y_val = features.iloc[label_start:][[1]]

dataset_val = tf.keras.preprocessing.timeseries_dataset_from_array(
    x_val, y_val, sequence_length=sequence_length, 
    sampling_rate=step, batch_size=128
)

for batch in dataset_train.take(1):
    inputs, targets = batch
    
print('input shape : ', inputs.numpy().shape)   #  (128, 120, 7)
print('target shape : ', targets.numpy().shape) # (128, 1)


#2. 모델구성
model = Sequential()
model.add(LSTM(units=100, return_sequences=False,
               activation='relu', input_shape=(120, 7)))
# model.add(LSTM(128, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='linear'))
model.add(Dense(32, activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary() 


#3. 훈련
model.compile(loss='mae', optimizer='adam', metrics='mse')  

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")   
print(date)

filepath = './_ModelCheckPoint/k44/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '_', date, '_', filename])
                      )
start_time = time.time() 
history = model.fit(dataset_train, epochs=10, batch_size=128,
                 validation_data=dataset_val,
                 callbacks=[earlyStopping, mcp])
end_time = time.time() - start_time


#4. 평가, 예측

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    print("Final val loss: ", val_loss)

visualize_loss(history, "Training and Validation Loss")

dataset_test = tf.keras.preprocessing.timeseries_dataset_from_array(
    x_val, y_val, sequence_length=sequence_length,
    sequence_stride=int(sequence_length * 6),
    sampling_rate=step, batch_size=1,
)

def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    return

for x, y in dataset_test.take(5):
    prediction = model.predict(x)
    prediction = prediction[0]
    print('prediction:', prediction)
    print('truth:', y[0].numpy())
    show_plot(
        [x[0][:, 1].numpy(), y[0], prediction],
        12,
        "Single Step Prediction",
    )
    



# Final val loss:  [0.8690679669380188, 1.0051515102386475, 0.39944490790367126, 0.46245890855789185, 
#                   0.4090108275413513, 0.44347405433654785, 0.5601844191551208, 0.5197162628173828, 0.7583364248275757, 0.43876129388809204]
# prediction: [0.16216938]
# truth: [0.2258011]
# prediction: [0.3217088]
# truth: [0.43425469]
# prediction: [0.44898885]
# truth: [0.03356057]
# prediction: [0.31530237]
# truth: [0.98434055]
# prediction: [0.49841094]
# truth: [0.41109318]
