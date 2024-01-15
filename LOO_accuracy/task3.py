import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
warnings.filterwarnings('ignore')
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import glob
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout, Conv2D  ,MaxPooling2D, Reshape
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.optimizers import Adam
import tensorflow as tf

class CustomEncoder:
    def __init__(self):
        print('hello')
        self.class_mapping = {
            'sitting_standing': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'lyingLeft':        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'lyingRight':       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'lyingBack':        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'lyingStomach':     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'normalWalking':    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'running':          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'descending':       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'ascending':        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'shuffleWalking':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'miscMovement':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }

    def fit_transform(self, y):
        return np.array([self.class_mapping[cls] for cls in y])

    def inverse_transform(self, y):
        reverse_mapping = {v: k for k, v in self.class_mapping.items()}
        return np.array([reverse_mapping[val] for val in y])

def prepare_data():
    ### Preparing Data
    respeck_filepaths = glob.glob("../Respeck/*")
    df1 = pd.DataFrame()
    for rfp in respeck_filepaths:
        files = glob.glob(f"{rfp}/*")
        
        for file in files:
            # [main_act,sub_act] = file.split(".csv")[0].split('_')[-2:]
            main_activity = " ".join(file.split(".csv")[0].split('_')[-2:])
            
            df = pd.read_csv(file,index_col=0)
            df['activity'] = main_activity
            df['user'] = rfp.split('\\')[-1]
            # print(df)
            df1 = df1.append(df)

    df1['activity'] = df1['activity'].apply(lambda x: x.replace('standing','sitting/standing'))
    df1['activity'] = df1['activity'].apply(lambda x: x.replace('sitting ','sitting/standing '))
    
    columns = ['user','activity','timestamp', 'accel_x', 'accel_y', 'accel_z']
    df_har = df1[columns]
    # removing null values
    df_har = df_har.dropna()
    df_har.shape
    # transforming the user to float
    df_har['user'] = df_har['user'].str.replace('s', '')
    df_har['user'] = df_har['user'].apply(lambda x:int(x))

    classes = ['lyingBack breathingNormal', 'lyingBack coughing',
        'lyingBack hyperventilating', 'lyingBack laughing',
        'lyingBack singing', 'lyingBack talking',
        'lyingLeft breathingNormal', 'lyingLeft coughing',
        'lyingLeft hyperventilating', 'lyingLeft laughing',
        'lyingLeft singing', 'lyingLeft talking',
        'lyingRight breathingNormal', 'lyingRight coughing',
        'lyingRight hyperventilating', 'lyingRight laughing',
        'lyingRight singing', 'lyingRight talking',
        'lyingStomach breathingNormal', 'lyingStomach coughing',
        'lyingStomach hyperventilating', 'lyingStomach laughing',
        'lyingStomach singing', 'lyingStomach talking',
        'sitting/standing breathingNormal', 'sitting/standing coughing',
        'sitting/standing eating', 'sitting/standing hyperventilating',
        'sitting/standing laughing', 'sitting/standing singing',
        'sitting/standing talking']


    df_har = df_har[df_har['activity'].isin(classes)] 
    df_har.to_csv('./t3_data/raw_data.csv',index=False)
    
def load_data():
    # ONLY RUN THIS AFTER CSV GENERATION
    all_df = pd.read_csv('./t3_data/raw_data.csv')
    return all_df

def segments_no_overlap(data):
    segments = []
    labels = []

    for i in range(0,  data.shape[0]- n_time_steps, step):  

        xs = data['accel_x'].values[i: i + n_time_steps]
        ys = data['accel_y'].values[i: i + n_time_steps]
        zs = data['accel_z'].values[i: i + n_time_steps]
        label = stats.mode(data['activity'][i: i + n_time_steps])[0][0]

        segments.append([xs, ys, zs])
        labels.append(label)
        
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)
    labels = np.asarray(labels).reshape(-1,1)
    return reshaped_segments, labels

def get_segment_label(train_df,test_df):
    # saving training and testing data
    user = test_df['user'].unique()[0]
    train_df.to_csv(f'./t3_data/train/{user}_train.csv')
    test_df.to_csv(f'./t3_data/test/{user}_test.csv')
    
    # segmenting the data into windows 
    train_segments, train_labels = segments_no_overlap(train_df)
    test_segments, test_labels = segments_no_overlap(test_df)
    # transforming the labels into OneHotEncoding
    enc = OneHotEncoder(handle_unknown='ignore').fit(train_labels)
    train_labels_encoded = enc.transform(train_labels).toarray()
    test_labels_encoded = enc.transform(test_labels).toarray()
    
    return train_segments, train_labels_encoded, test_segments, test_labels_encoded, enc.categories_

def model_cnn(trainX, trainy):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Reshape((n_timesteps, n_features, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3,1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=n_epochs, batch_size=batch_size, verbose=1)
    # evaluate model
    return model
    
    
    
if __name__ == '__main__':
    os.mkdir('./t3_data')
    os.mkdir('./t3_data/train')
    os.mkdir('./t3_data/test')
    os.mkdir('./t3_data/models')
    
    
    prepare_data()
    all_df = load_data()
       
    n_time_steps = 50 
    n_features = 3 
    step = 10
    n_epochs = 20  
    batch_size = 32
    
    accuracies = {}
    for user in all_df['user'].unique():
        # if user != 91 :
        #     continue
        
        train_df = all_df[all_df['user'] != user]
        test_df = all_df[all_df['user'] == user]
        
        X_train, y_train, X_test, y_test, categories = get_segment_label(train_df,test_df)
        
        # Train and evaluate the model 
        model  = model_cnn(X_train,y_train)
        loss, accuracy = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
        
        # Store current accuracy
        print(f"Test Accuracy ({user}):", accuracy)
        print(f"Test Loss ({user}):", loss)
        accuracies[user] = {'loss':loss, 'accuracy': accuracy}
        
        # Convert the model.
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the model.
        with open(f'./t3_data/models/cnn_model_t3_u{user}_{n_time_steps}_{step}_{n_features}.tflite', 'wb') as f:
            f.write(tflite_model)
        
        # break

    a_df = pd.DataFrame(accuracies)
    a_df.to_csv('./t3_data/t3_loo_accuracies.csv')
        