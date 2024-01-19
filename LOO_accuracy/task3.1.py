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

class CustomEncoderMain:
    def __init__(self):
        self.classes = [ 'lyingBack','lyingLeft','lyingRight', 'lyingStomach', 
            'sitting/standing',]
        
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
  
class CustomEncoderSub:
    def __init__(self):
        self.classes = sub_classes = [ 'breathingNormal', 'coughing', 'hyperventilating', 'laughing',
            'singing', 'talking', 'eating',]
        
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
            activity = file.split(".csv")[0].split('_')[-2:]
            # print(activity)
            main_activity = activity[0]
            sub_activity = activity[-1]

            df = pd.read_csv(file, index_col=0)
            df['main_activity'] = main_activity
            df['sub_activity'] = sub_activity 
            df['user'] = int(rfp.split('\\')[-1].replace('s', ''))
            df_list.append(df)
            # print(df)
    df_combined = pd.concat(df_list)

    # Combine 'standing' and 'sitting' activities
    df_combined['main_activity'] = df_combined['main_activity'].replace({'standing': 'sitting/standing', 'sitting ': 'sitting/standing '})

    columns = ['user', 'main_activity', 'sub_activity', 'timestamp', 'accel_x', 'accel_y', 'accel_z']
    df_har = df_combined[columns]

    # Drop null values
    df_har = df_har.dropna()

    # Transform the 'user' column to integers
    df_har['user'] = df_har['user'].astype(int)
    
    main_classes = [
        'lyingBack','lyingLeft','lyingRight', 'lyingStomach', 
        'sitting/standing',
    ]

    sub_classes = [
        'breathingNormal', 'coughing', 'hyperventilating', 'laughing',
        'singing', 'talking', 'eating',
    ]
    df_har = df_har[df_har['main_activity'].isin(main_classes)]
    df_har = df_har[df_har['sub_activity'].isin(sub_classes)]
    
    df_har.to_csv('./t31_data/raw_data.csv', index=False)

def load_data():
    # ONLY RUN THIS AFTER CSV GENERATION
    all_df = pd.read_csv('./t31_data/raw_data.csv')
    return all_df

def segments_no_overlap(data):
    segments = []
    main_labels = []
    sub_labels = []

    for i in range(0,  data.shape[0]- n_time_steps, step):  

        xs = data['accel_x'].values[i: i + n_time_steps]
        ys = data['accel_y'].values[i: i + n_time_steps]
        zs = data['accel_z'].values[i: i + n_time_steps]
        # print(data['activity'][i: i + n_time_steps].mode()[0])
        main_label = data['main_activity'][i: i + n_time_steps].mode()[0]
        sub_label = data['sub_activity'][i: i + n_time_steps].mode()[0]
        
        segments.append([xs, ys, zs])
        main_labels.append(main_label)
        sub_labels.append(sub_label)
        
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)
    main_labels = np.asarray(main_labels).reshape(-1,1)
    sub_labels = np.asarray(sub_labels).reshape(-1,1)
    
    return reshaped_segments, main_labels, sub_labels

def get_segment_label(train_df,test_df):
    # saving training and testing data
    user = test_df['user'].unique()[0]
    
    if args.save_training:
        train_df.to_csv(f'./t31_data/train/{user}_train.csv')
    
    test_df.to_csv(f'./t31_data/test/{user}_test.csv')
    
    # segmenting the data into windows 
    train_segments, train_main_labels, train_sub_labels = segments_no_overlap(train_df)
    test_segments, test_main_labels, test_sub_labels = segments_no_overlap(test_df)
    
    # transforming the labels into OneHotEncoding
    # enc = OneHotEncoder(handle_unknown='ignore').fit(train_labels)
    main_enc = CustomEncoderMain()
    train_main_labels_encoded = main_enc.fit_transform(train_main_labels)
    test_main_labels_encoded = main_enc.fit_transform(test_main_labels)
    
    sub_enc = CustomEncoderSub()
    train_sub_labels_encoded = sub_enc.fit_transform(train_sub_labels)
    test_sub_labels_encoded = sub_enc.fit_transform(test_sub_labels)
    
    return train_segments, train_main_labels_encoded, train_sub_labels_encoded, test_segments, test_main_labels_encoded, test_sub_labels_encoded, main_enc, sub_enc

def model_cnn(trainX, trainy):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Reshape((n_timesteps, n_features, 1)))
    model.add(Conv2D(filters=256, kernel_size=(3,1), activation='relu'))  
    
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
    
def model_cnn4(trainX, trainy):
    print("training model 4")
    n_time_steps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[0]
    
    m = Sequential()
    m.add(Conv1D(64, 3, activation='relu', input_shape=(n_time_steps, 3))) 
    m.add(BatchNormalization())
    m.add(Dropout(0.7))
    
    m.add(Conv1D(64, 3, activation='relu', input_shape=(n_time_steps, 3)))
    m.add(BatchNormalization())
    m.add(Dropout(0.7))
    
    m.add(Conv1D(64, 3, activation='relu', input_shape=(n_time_steps, 3)))
    m.add(BatchNormalization())
    m.add(Dropout(0.7))
    
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dense(64, activation='relu'))
    m.add(Dense(n_outputs, activation='softmax')) # Change this to the number of classes you have

    # Compile model
    optimizer = Adam(learning_rate=0.001)
    m.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    m.fit(trainX, trainy, epochs = n_epochs, verbose=1)
    
    return m
   
def model_cnn5(trainX, trainy):
    print("training model 5")
    n_time_steps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_time_steps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_time_steps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_time_steps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_time_steps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_time_steps, n_features)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    
    # model.add(MaxPooling1D(pool_size=2))
    # Flatten layer to transition from convolutional to dense layers

    # Dense layers with ReLU activation
    # model.add(Dense(64, activation='relu'))

    # Output layer with Softmax activation for classification
    model.add(Dense(n_outputs, activation='softmax'))

    # Compile the model
    # optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Compile model
    # optimizer = Adam(learning_rate=0.001)
    # m.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(trainX, trainy, epochs = n_epochs, verbose=1)
    
    return model

def model_cnn6(trainX, trainy):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    print(trainX.shape,trainy.shape)
    model = Sequential()
    
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_time_steps, n_features)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_time_steps, n_features)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_time_steps, n_features)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Flatten())    
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    
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
        with open(f'./t31_data/models/cnn_model_t31_u{user}_{n_time_steps}_{step}_{n_features}.tflite', 'wb') as f:
            f.write(tflite_model)

    a_df = pd.DataFrame(accuracies)
    a_df.to_csv('./t31_data/t31_loo_accuracies.csv')
    return accuracies
    
def train_best():
    print("Training best LOO accuracy model")
    accuracies = []
    user = 98
        
    train_df = general_act_df[general_act_df['user'] != user]
    test_df = general_act_df[general_act_df['user'] == user]
    
    X_train, y_main_train, y_sub_train, X_test, y_main_test, y_sub_test, main_enc, sub_enc = get_segment_label(train_df,test_df)
    
    # Train and evaluate the model 
    model_sub  = model_cnn(X_train,y_sub_train)
    model_main  = model_cnn(X_train,y_main_train)
    
    loss_main, accuracy_main = model_main.evaluate(X_test, y_main_test, batch_size = batch_size, verbose = 1)
    loss_sub, accuracy_sub = model_sub.evaluate(X_test, y_sub_test, batch_size = batch_size, verbose = 1)
    
    
    # Store current accuracy
    print(f"Test Accuracy ({user}):", accuracy_main)
    print(f"Test Loss ({user}):", loss_main)
    print(f"Test Accuracy ({user}):", accuracy_sub)
    print(f"Test Loss ({user}):", loss_sub)
    accuracies.append({'user':user, 'loss':loss_main, 'accuracy_main': accuracy_main, 'loss_sub':loss_sub, 'accuracy_sub': accuracy_sub})
    
    # Convert the model.
    converter_main = tf.lite.TFLiteConverter.from_keras_model(model_main)
    tflite_model_main = converter_main.convert()
    
    converter_sub = tf.lite.TFLiteConverter.from_keras_model(model_sub)
    tflite_model_sub = converter_sub.convert()

    # Save the model.
    with open(f'./t31_data/models/cnn_model_main_t31_u{user}_{n_time_steps}_{step}_{n_features}.tflite', 'wb') as f:
        f.write(tflite_model_main)
    with open(f'./t31_data/models/cnn_model_sub_t31_u{user}_{n_time_steps}_{step}_{n_features}.tflite', 'wb') as f:
        f.write(tflite_model_sub)

    a_df = pd.DataFrame(accuracies)
    a_df.to_csv('./t31_data/t31_loo_accuracies.csv')
    return accuracies
   
def create_dirs():
    main_folder_path = './t31_data'
    train_folder_path = './t31_data/train'
    test_folder_path = './t31_data/test'
    model_folder_path = './t31_data/models'
    
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
    if args.prepare_data or not os.path.exists('./t31_data/raw_data.csv'):
        prepare_data()
        
    # Load the data 
    general_act_df = load_data()
    
    # Set parameters
    random_seed = 42   
    n_time_steps = 125 
    n_features = 3 
    step = 15
    n_epochs = 20      
    batch_size = 32
    
    accuracies = train_best() if args.best_model_only else train_all()
    print(accuracies)
    