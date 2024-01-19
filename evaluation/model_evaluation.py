
import numpy as np 
import pandas as pd
import os
import matplotlib as plt
import tensorflow as tf
import argparse
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score



def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the Group P1\'s model evaluation script')

    parser.add_argument('--model_path', nargs="?", type=str, help='path of the exported model or folder path for task 3')
    parser.add_argument('--test_data_path', nargs="?", type=str, help='Path to the test data')
       
    args = parser.parse_args()
    
    return args

class CustomEncoder1:
    def __init__(self):
        
        self.classes = [
            'sitting_standing', 
            'lyingLeft',
            'lyingRight',
            'lyingBack',
            'lyingStomach',
            'normalWalking',
            'running',
            'descending',
            'ascending',
            'shuffleWalking',
            'miscMovement',
        ]
        
        self.categories_ = {}
        for i in range(len(self.classes)):
            enc = [0]*len(self.classes)
            enc[i] = 1
            
            self.categories_[self.classes[i]] = enc 

    def fit_transform(self, y):
        return np.array([self.categories_[cls[0]] for cls in y])

    def inverse_transform(self, y):
        inv_transform = []
        for e in y :
            class_index = list(e).index(1)
            # print(class_index)
            inv_transform.append(self.classes[class_index])
            
        return np.array(inv_transform)
    
class CustomEncoder2:
    def __init__(self):
        self.classes = ['lyingBack breathingNormal', 'lyingBack coughing',
       'lyingBack hyperventilating', 'lyingLeft breathingNormal',
       'lyingLeft coughing', 'lyingLeft hyperventilating',
       'lyingRight breathingNormal', 'lyingRight coughing',
       'lyingRight hyperventilating', 'lyingStomach breathingNormal',
       'lyingStomach coughing', 'lyingStomach hyperventilating',
       'sitting_standing breathingNormal', 'sitting_standing coughing',
       'sitting_standing hyperventilating']
        
        self.categories_ = {}
        for i in range(len(self.classes)):
            enc = [0]*len(self.classes)
            enc[i] = 1
            
            self.categories_[self.classes[i]] = enc 

    def fit_transform(self, y):
        return np.array([self.categories_[cls[0]] for cls in y])

    def inverse_transform(self, y):
        inv_transform = []
        for e in y :
            class_index = list(e).index(1)
            # print(class_index)
            inv_transform.append(self.classes[class_index])
            
        return np.array(inv_transform)

class CustomEncoder3:
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
            
        # print(self.categories_)

    def fit_transform(self, y):
        return np.array([self.categories_[cls[0]] for cls in y])

    def inverse_transform(self, y):
        inv_transform = []
        for e in y :
            class_index = list(e).index(1)
            # print(class_index)
            inv_transform.append(self.classes[class_index])
            
        return np.array(inv_transform)

def segments_no_overlap(data):
    segments = []
    labels = []
    step = 15

    for i in range(0,  data.shape[0]- n_time_steps, step):  

        xs = data['accel_x'].values[i: i + n_time_steps]
        ys = data['accel_y'].values[i: i + n_time_steps]
        zs = data['accel_z'].values[i: i + n_time_steps]
        # print(data['activity'][i: i + n_time_steps].mode()[0])
        label = data['activity'][i: i + n_time_steps].mode()[0]

        segments.append([xs, ys, zs])
        labels.append(label)
        
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)
    labels = np.asarray(labels).reshape(-1,1)
    return reshaped_segments, labels

def segments_no_overlap3(data):
    segments = []
    labels = []
    step = 50

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

def get_segments(test_df,output_details):
    
    # transforming the labels into OneHotEncoding
    model = output_details[0]['shape'][1]
    if model == 11:
        test_segments, test_labels = segments_no_overlap(test_df)
        enc = CustomEncoder1()
        test_labels_encoded = enc.fit_transform(test_labels)
    elif model == 15:
        test_segments, test_labels = segments_no_overlap(test_df)
        enc = CustomEncoder2()
        test_labels_encoded = enc.fit_transform(test_labels)
    else: 
        test_segments, test_labels = segments_no_overlap3(test_df)
        enc = CustomEncoder3()
        test_labels_encoded = enc.fit_transform(test_labels)
    
    return test_segments, test_labels_encoded, enc

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
    
    tflite_interpreter = load_tflite_model(args.model_path)
    
    # Get input and output tensors
    input_details = tflite_interpreter.get_input_details()
    [_,n_time_steps,n_features] = input_details[0]['shape']
    output_details = tflite_interpreter.get_output_details()
    
    test_data = pd.read_csv(args.test_data_path)
    X_test, y_test, enc = get_segments(test_data,output_details)
    
    y_pred = []
    for i in range(len(X_test)):
        tflite_interpreter.set_tensor(input_details[0]['index'], np.float32(X_test[i:i+1]))
        tflite_interpreter.invoke()
        output_data = tflite_interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(output_data)
        
    y_pred_encoded = convert_to_one_hot(y_pred)
   
    # Generate classification report
    report = classification_report(y_test, y_pred_encoded)
    # Print the classification report
    print(report)
    
    
#  python .\evaluation\model_evaluation.py --model_path=./LOO_accuracy/t1_data/models/cnn_model_t1_u98_125_15_3.tflite --test_data_path=./LOO_accuracy/t1_data/test/98_test.csv