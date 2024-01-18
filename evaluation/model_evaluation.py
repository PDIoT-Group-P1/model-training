
import numpy as np 
import pandas as pd
import os
import matplotlib as plt
# from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import argparse
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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

def get_segments(test_df,output_details):

    # segmenting the data into windows 
    test_segments, test_labels = segments_no_overlap(test_df)
    
    # transforming the labels into OneHotEncoding
    # enc = OneHotEncoder(handle_unknown='ignore').fit(train_labels)
    # train_labels_encoded = enc.transform(train_labels).toarray()
    # test_labels_encoded = enc.transform(test_labels).toarray()
    model = output_details[0]['shape'][1]
    if model == 11:
        enc = CustomEncoder1()
        test_labels_encoded = enc.fit_transform(test_labels)
    elif model == 15:
        enc = CustomEncoder2()
        test_labels_encoded = enc.fit_transform(test_labels)
    else: 
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
    
    tflite_interpreter = load_tflite_model(args.model_folder_path)
    
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

    reverse_y_test = enc.inverse_transform(y_test)
    reverse_y_pred = enc.inverse_transform(y_pred_encoded)
    # confusion_mat = confusion_matrix(reverse_y_test, reverse_y_pred,labels=enc.classes)
    disp = ConfusionMatrixDisplay.from_predictions(reverse_y_test, reverse_y_pred)
    disp.ax_.set_xticklabels(range(15),rotation=90)
    disp.ax_.set_yticklabels(range(15))
    
    disp.figure_.subplots_adjust(bottom=0.15)
    disp.figure_.savefig("cfm.png")
#  python .\evaluation\model_evaluation.py --model_folder_path=./LOO_accuracy/t1_data/models/cnn_model_t1_u98_125_15_3.tflite --test_data_path=./LOO_accuracy/t1_data/test/98_test.csv