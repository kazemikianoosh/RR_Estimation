#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
When using this resource, please cite the original publication:
Kianoosh Kazemi, Iman Azimi, Amir M. Rahmani, and Pasi Liljeberg, “Robust End-to-End PPG-Based Respiration
Rate Estimation Method: A Transfer Learning
Approach”


authors: Kianoosh Kazemi

"""

import argparse
import time
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.losses import Huber
from tf_model import Multi_class_CNN
from keras import backend as K

def load_data():
    with open('data/raw_signal.pkl', 'rb') as f:
        raw_data = pkl.load(f)
    raw_data = np.transpose(raw_data, (0, 2, 1))

    annotation = pd.read_pickle('data/annotation.pkl')
    reference_rr = annotation['Reference_RR'].values.reshape(-1, 1)
    activity_id = annotation['activity_id'].values.reshape(-1, 1)

    raw_data = np.around(raw_data, decimals=4)
    reference_rr = np.around(reference_rr, decimals=4)

    training_ids = annotation['patient_id'] < 10

    train_rr_ref = tf.convert_to_tensor(reference_rr[training_ids.values], dtype='float32')
    test_rr_ref = tf.convert_to_tensor(reference_rr[~training_ids.values], dtype='float32')
    train_sig_raw = tf.convert_to_tensor(raw_data[training_ids.values], dtype='float32')
    test_sig_raw = tf.convert_to_tensor(raw_data[~training_ids.values], dtype='float32')

    train_activity_id = activity_id[training_ids.values]
    test_activity_id = activity_id[~training_ids.values]

    return train_rr_ref, test_rr_ref, train_sig_raw, test_sig_raw, train_activity_id, test_activity_id

def create_datasets(train_sig_raw, train_rr_ref, test_sig_raw, test_rr_ref):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_sig_raw, train_rr_ref))
    train_dataset = train_dataset.shuffle(len(train_sig_raw)).batch(128)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_sig_raw, test_rr_ref))
    test_dataset = test_dataset.batch(128)
    return train_dataset, test_dataset

def create_model(model_input_shape):
    model = Multi_class_CNN(model_input_shape)
    loss_fn = Huber()
    model(tf.ones((128, model_input_shape[0], model_input_shape[1])))
    model.load_weights('Trained_model.h5')
    return model, loss_fn

def perform_inference(model, test_dataset):
    final_output = tf.convert_to_tensor([])
    ref_rr = tf.convert_to_tensor([])

    for step, (x_batch_test_raw, x_batch_test_ref_rr) in enumerate(test_dataset):
        output = model(x_batch_test_raw)

        if step == 0:
            final_output = output
            ref_rr = x_batch_test_ref_rr
        else:
            final_output = tf.concat([final_output, output], axis=0)
            ref_rr = tf.concat([ref_rr, x_batch_test_ref_rr], axis=0)

    return final_output, ref_rr

def calculate_errors(final_output, ref_rr):
    final_output_rr = final_output.numpy().reshape(final_output.shape[0], final_output.shape[1])
    avg_ref_breath = ref_rr.numpy().reshape(-1, 1)
    avg_ref_breath = avg_ref_breath.reshape(avg_ref_breath.shape[0], avg_ref_breath.shape[1])

    error = np.abs(avg_ref_breath - final_output_rr)
    mask = ~np.isnan(error)
    mae = np.mean(error[mask])
    rmse = np.sqrt(np.mean(error[mask]**2))
    return mae, rmse, error, mask

if __name__ == '__main__':
    # Load Data
    train_rr_ref, test_rr_ref, train_sig_raw, test_sig_raw, train_activity_id, test_activity_id = load_data()

    # Create Datasets
    train_dataset, test_dataset = create_datasets(train_sig_raw, train_rr_ref, test_sig_raw, test_rr_ref)

    # Define Model
    model_input_shape = (2048, 4)
    model, loss_fn = create_model(model_input_shape)

    # Perform Inference
    final_output, ref_rr = perform_inference(model, test_dataset)

    # Calculate Errors
    mae, rmse, error, mask = calculate_errors(final_output, ref_rr)

    # Display Results
    model.summary()
    print('Mean Absolute Error is:', mae)
    print('Root Mean Square Error is:', rmse)
    print('The length of the Reference RR is', len(ref_rr))
    print('The length of the final output is', len(final_output))

    # Create a DataFrame for analysis
    array_fusion = np.concatenate((error[mask].reshape(-1, 1), np.zeros((len(error[mask]), 1)),
                                    test_activity_id[mask].reshape(-1, 1)), axis=1)

    final_array = array_fusion
    data_frame = pd.DataFrame(final_array, columns=['Absolute Error(BrPM)', 'Modality', 'Activity_id'])
    data_frame['Modality'] = data_frame['Modality'].astype('category')
    data_frame['Activity_id'] = data_frame['Activity_id'].astype('category')
    data_frame['Modality'] = data_frame['Modality'].cat.rename_categories(['Final RR'])
    data_frame['Activity_id'] = data_frame['Activity_id'].cat.rename_categories(['Transit', 'Baseline', 'Stress', 'Amusement', 'Meditation'])

    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(x="Activity_id", y="Absolute Error(BrPM)", hue="Modality", data=data_frame, showfliers=False, width=0.8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    plt.setp(ax.get_legend().get_title(), fontsize=15)
    plt.setp(ax.get_legend().get_texts(), fontsize=16)
    plt.xlabel('Activity', fontsize=20)
    plt.ylabel('Absolute Error(BrPM)', fontsize=15)
    plt.yticks(fontsize=20)
    plt.show()
    # Plot raw data
    sampling_rate = 64
    time_array = np.arange(0, len(train_sig_raw[5][:, 0])) / 64
    plt.figure(figsize=(10, 10.5))
    acc_axis = ['x','y','z']
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(time_array, train_sig_raw[5][:, i])
        plt.xlabel('Time (seconds)')
        if i==0:
            plt.title('PPG Signal')
        else:
            plt.title('ACC'+ acc_axis[i-1] +' Signal')
    plt.show()
