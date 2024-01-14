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

class CustomLabelEncoder:
    def _init_(self):
        self.class_mapping = {
            'sitting_standing': 0,
            'lyingLeft': 1,
            'lyingRight': 2,
            'lyingBack': 3,
            'lyingStomach': 4,
            'normalWalking': 5,
            'running': 6,
            'descending': 7,
            'ascending': 8,
            'shuffleWalking': 9,
            'miscMovement':10
        }

    def fit_transform(self, y):
        return [self.class_mapping[cls] for cls in y]

    def inverse_transform(self, y):
        reverse_mapping = {v: k for k, v in self.class_mapping.items()}
        return [reverse_mapping[val] for val in y]

def prepare_data():
    ### Preparing Data
    respeck_filepaths = glob.glob("../Respeck/*")
    df1 = pd.DataFrame()
    for rfp in respeck_filepaths:
        files = glob.glob(f"{rfp}/*")
        
        for file in files:
            [main_act,sub_act] = file.split(".csv")[0].split('_')[-2:]
            # main_activity = file.split(".csv")[0].split('_')[-2]
            
            df = pd.read_csv(file,index_col=0)
            df['activity'] = main_act
            df['sub_activity'] = sub_act
            df['user'] = rfp.split('\\')[-1]
            # print(df)
            df1 = pd.concat([df, df1], axis=0)

    df1 = df1[df1['sub_activity'] == 'breathingNormal']     
    df1.loc[df1['activity'].isin(('sitting', 'standing')),'activity'] = 'sitting_standing'
    columns = ['user','activity','timestamp', 'accel_x', 'accel_y', 'accel_z']

    # df1 = df1[columns]
    df_har = df1[columns]

    # removing null values
    df_har = df_har.dropna()
    df_har.shape

    # transforming the user to float
    df_har['user'] = df_har['user'].str.replace('s', '')
    df_har['user'] = df_har['user'].apply(lambda x:int(x))

    df_har.to_csv('./t1_data/raw_data.csv',index=False)
    
def load_data():
    # ONLY RUN THIS AFTER CSV GENERATION
    general_act_df = pd.read_csv('./t1_data/raw_data.csv')
    return general_act_df

def segments_overlap(data):
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

    # le = LabelEncoder()
    # labels = le.fit_transform(labels)
    enc = OneHotEncoder(handle_unknown='ignore').fit(labels)
    labels = enc.transform(labels).toarray()
    # labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)
    # print(enc.categories_)
    return reshaped_segments,labels,enc.categories_

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

def save_train_test_data(user,train_df,test_df):
    train_df.to_csv(f'./t1_data/train/{user}_train.csv')
    test_df.to_csv(f'./t1_data/test/{user}_test.csv')
    
if __name__ == '__main__':
    os.mkdir('./t1_data')
    os.mkdir('./t1_data/train')
    os.mkdir('./t1_data/test')
    os.mkdir('./t1_data/models')
    
    
    prepare_data()
    general_act_df = load_data()
    
    random_seed = 42   
    n_time_steps = 50 
    n_features = 3 
    step = 10
    n_epochs = 1      
    batch_size = 32
    
    accuracies = {}
    for user in general_act_df['user'].unique():
        
        train_df = general_act_df[general_act_df['user'] != user]
        test_df = general_act_df[general_act_df['user'] == user]
        save_train_test_data(user,train_df,test_df)

        X_train, y_train, categories = segments_overlap(train_df)
        X_test, y_test, cat2 = segments_overlap(test_df)
        
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
        with open(f'./t1_data/models/cnn_model_t1_u{user}_{n_time_steps}_{step}_{n_features}.tflite', 'wb') as f:
            f.write(tflite_model)
        
        # break

    a_df = pd.DataFrame(accuracies)
    a_df.to_csv('./t1_data/t1_loo_accuracies.csv')
        