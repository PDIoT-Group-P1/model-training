import os
import glob
import shutil
import argparse
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from keras.layers import LSTM, Dense, Flatten, Dropout, Reshape
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, BatchNormalization

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the Group P1\'s Task 2 Script')

    parser.add_argument('--save_training', action='store_true', help='Training data should be saved')
    parser.add_argument('--prepare_data', action='store_true', help='To generate the raw training data from scratch')
    parser.add_argument('--best_model_only', action='store_true', help='Only train the best model again')
    
    args = parser.parse_args()
    
    return args

class CustomEncoder:
    def __init__(self):
        self.classes = ['lyingBack breathingNormal', 'lyingBack coughing',
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
        
        self.categories_ = {}
        for i in range(len(self.classes)):
            enc = [0]*len(self.classes)
            enc[i] = 1
            
            self.categories_[self.classes[i]] = enc 

    def fit_transform(self, y):
        return np.array([self.categories_[cls[0]] for cls in y])

    def inverse_transform(self, y):
        reverse_mapping = {v: k for k, v in self.categories_.items()}
        return np.array([reverse_mapping[val] for val in y])

def prepare_data():
    # Preparing Data
    respeck_filepaths = glob.glob("../Respeck/*")
    df_list = []

    for rfp in respeck_filepaths:
        files = glob.glob(f"{rfp}/*")

        for file in files:
            main_activity = " ".join(file.split(".csv")[0].split('_')[-2:])

            df = pd.read_csv(file, index_col=0)
            df['activity'] = main_activity
            df['user'] = int(rfp.split('\\')[-1].replace('s', ''))
            df_list.append(df)

    df_combined = pd.concat(df_list)

    # Combine 'standing' and 'sitting' activities
    df_combined['activity'] = df_combined['activity'].replace({'standing': 'sitting/standing', 'sitting ': 'sitting/standing '})

    columns = ['user', 'activity', 'timestamp', 'accel_x', 'accel_y', 'accel_z','gyro_x','gyro_y','gyro_z']
    df_har = df_combined[columns]

    # Drop null values
    df_har = df_har.dropna()

    # Transform the 'user' column to integers
    df_har['user'] = df_har['user'].astype(int)

    classes = [
        'lyingBack breathingNormal', 'lyingBack coughing', 'lyingBack hyperventilating', 'lyingBack laughing',
        'lyingBack singing', 'lyingBack talking', 'lyingLeft breathingNormal', 'lyingLeft coughing',
        'lyingLeft hyperventilating', 'lyingLeft laughing', 'lyingLeft singing', 'lyingLeft talking',
        'lyingRight breathingNormal', 'lyingRight coughing', 'lyingRight hyperventilating', 'lyingRight laughing',
        'lyingRight singing', 'lyingRight talking', 'lyingStomach breathingNormal', 'lyingStomach coughing',
        'lyingStomach hyperventilating', 'lyingStomach laughing', 'lyingStomach singing', 'lyingStomach talking',
        'sitting/standing breathingNormal', 'sitting/standing coughing', 'sitting/standing eating',
        'sitting/standing hyperventilating', 'sitting/standing laughing', 'sitting/standing singing',
        'sitting/standing talking'
    ]

    df_har = df_har[df_har['activity'].isin(classes)]
    df_har.to_csv('./t3_data/raw_data.csv', index=False)

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
        
        gxs = data['gyro_x'].values[i: i + n_time_steps]
        gys = data['gyro_y'].values[i: i + n_time_steps]
        gzs = data['gyro_z'].values[i: i + n_time_steps]
        
        # print(data['activity'][i: i + n_time_steps].mode()[0])
        label = data['activity'][i: i + n_time_steps].mode()[0]

        segments.append([xs, ys, zs,gxs,gys,gzs])
        labels.append(label)
        
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)
    labels = np.asarray(labels).reshape(-1,1)
    return reshaped_segments, labels

def get_segment_label(train_df,test_df):
    # saving training and testing data
    user = test_df['user'].unique()[0]
    
    if args.save_training:
        train_df.to_csv(f'./t3_data/train/{user}_train.csv')
    
    test_df.to_csv(f'./t3_data/test/{user}_test.csv')
    
    # segmenting the data into windows 
    train_segments, train_labels = segments_no_overlap(train_df)
    test_segments, test_labels = segments_no_overlap(test_df)
    
    # transforming the labels into OneHotEncoding
    # enc = OneHotEncoder(handle_unknown='ignore').fit(train_labels)
    enc = CustomEncoder()
    train_labels_encoded = enc.fit_transform(train_labels)
    test_labels_encoded = enc.fit_transform(test_labels)
    
    return train_segments, train_labels_encoded, test_segments, test_labels_encoded, enc

def model_cnn(trainX, trainy):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Reshape((n_timesteps, n_features, 1)))
    model.add(Conv2D(filters=128, kernel_size=(3,1), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,1), activation='relu'))
    
    model.add(Dropout(0.7))
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=n_epochs, batch_size=batch_size, verbose=1)
    # evaluate model
    return model
    
def train_all():
    print("Calculating LOO accuracy for all users")
    accuracies = []
    for user in general_act_df['user'].unique():
    # for user in [48,43,98,38,16]:    
        train_df = general_act_df[general_act_df['user'] != user]
        test_df = general_act_df[general_act_df['user'] == user]
        
        X_train, y_train, X_test, y_test, categories = get_segment_label(train_df,test_df)
        
        # Train and evaluate the model 
        model  = model_cnn(X_train,y_train)
        loss, accuracy = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
        
        # Store current accuracy
        print(f"Test Accuracy ({user}):", accuracy)
        print(f"Test Loss ({user}):", loss)
        accuracies.append({'user':user, 'loss':loss, 'accuracy': accuracy})
        
        # Convert the model.
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the model.
        with open(f'./t3_data/models/cnn_model_t3_u{user}_{n_time_steps}_{step}_{n_features}.tflite', 'wb') as f:
            f.write(tflite_model)

    a_df = pd.DataFrame(accuracies)
    a_df.to_csv('./t3_data/t3_loo_accuracies.csv')
    return accuracies
    
def train_best():
    print("Training best LOO accuracy model")
    accuracies = []
    user = 98
        
    train_df = general_act_df[general_act_df['user'] != user]
    test_df = general_act_df[general_act_df['user'] == user]
    
    X_train, y_train, X_test, y_test, categories = get_segment_label(train_df,test_df)
    
    # Train and evaluate the model 
    model  = model_cnn(X_train,y_train)
    loss, accuracy = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
    
    # Store current accuracy
    print(f"Test Accuracy ({user}):", accuracy)
    print(f"Test Loss ({user}):", loss)
    accuracies.append({'user':user, 'loss':loss, 'accuracy': accuracy})
    
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(f'./t3_data/models/cnn_model_t3_u{user}_{n_time_steps}_{step}_{n_features}.tflite', 'wb') as f:
        f.write(tflite_model)

    a_df = pd.DataFrame(accuracies)
    a_df.to_csv('./t3_data/t3_loo_accuracies.csv')
    return accuracies
   
def create_dirs():
    main_folder_path = './t3_data'
    train_folder_path = './t3_data/train'
    test_folder_path = './t3_data/test'
    model_folder_path = './t3_data/models'
    
    # make the required directories 
    if not os.path.exists(main_folder_path):
        os.mkdir(main_folder_path)
        
    if args.save_training and not os.path.exists(train_folder_path):
        os.mkdirs(train_folder_path)
    
    if os.path.exists(test_folder_path):
        shutil.rmtree(test_folder_path)
    os.makedirs(test_folder_path)
    
    if os.path.exists(model_folder_path):
        shutil.rmtree(model_folder_path)
    os.makedirs(model_folder_path)    
    

if __name__ == '__main__':
    args = get_args()
    create_dirs()
    
    # Compile and Preprocess the required data 
    if args.prepare_data or not os.path.exists('./t3_data/raw_data.csv'):
        prepare_data()
        
    # Load the data 
    general_act_df = load_data()
    
    # Set parameters
    random_seed = 42   
    n_time_steps = 125 
    n_features = 6 
    step = 15
    n_epochs = 20      
    batch_size = 32
    
    accuracies = train_best() if args.best_model_only else train_all()
    print(accuracies)
    