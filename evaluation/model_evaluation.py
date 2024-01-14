
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import argparse
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the Group P1\'s model evaluation script')

    parser.add_argument('--model_folder_path', nargs="?", type=str, help='Folder containing exported models')
    parser.add_argument('--test_data_path', nargs="?", type=str, help='Path to the test data')
    
    args = parser.parse_args()
    
    return args

def get_segments(data,n_time_steps,n_features):
    step = 10 
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

def load_tflite_model(model_path):
    # Load the TFLite model using TensorFlow Lite Interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def convert_to_one_hot(y_pred):
    y_pred = np.concatenate(y_pred,axis=0)
    
    # Find the index of the maximum probability for each sample
    max_indices = np.argmax(y_pred, axis=1)

    # Convert indices to one-hot encoding
    one_hot_encoding = to_categorical(max_indices, num_classes=y_pred.shape[1])

    return one_hot_encoding

if __name__ == '__main__':
    args = get_args()  # get arguments from command line
    
    tflite_interpreter = load_tflite_model(args.model_folder_path)
    
    # Get input and output tensors
    input_details = tflite_interpreter.get_input_details()
    [_,n_timesteps,n_features] = input_details[0]['shape']
    output_details = tflite_interpreter.get_output_details()
    
    test_data = pd.read_csv(args.test_data_path)
    X_test, y_test, _ = get_segments(test_data,n_timesteps,n_features)
    
    y_pred = []
    for i in range(len(X_test)):
        tflite_interpreter.set_tensor(input_details[0]['index'], np.float32(X_test[i:i+1]))
        tflite_interpreter.invoke()
        output_data = tflite_interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(output_data)
        
   
    y_pred_encoded = convert_to_one_hot(y_pred)
    # Generate classification report
    report = classification_report(y_test, y_pred_encoded)
    # print(y_test)
    # print(y_pred_encoded)
    # Print the classification report
    print(report)
