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
from tensorflow.keras.optimizers import Adam
from tf_model import Multi_class_CNN
from keras import backend as K

def load_data(DaLia_WESAD):
    """
    Loads and processes raw PPG signal data and associated annotations.

    Input:
    - None (file-based function, requires 'data/raw_signal.pkl' and 'data/annotation.pkl' files in directory)

    Output:
    - train_rr_ref (TensorFlow Tensor): Training reference respiration rate (float32)
    - test_rr_ref (TensorFlow Tensor): Testing reference respiration rate (float32)
    - train_sig_raw (TensorFlow Tensor): Training raw signal data (float32)
    - test_sig_raw (TensorFlow Tensor): Testing raw signal data (float32)
    - train_activity_id (numpy array): Training activity IDs (integer)
    - test_activity_id (numpy array): Testing activity IDs (integer)
    """
    # Load raw PPG signal data from a pickle file
    if DaLia_WESAD == 'DaLiA':
        with open('data/PPG_DaLiA_Raw_Signal.pkl', 'rb') as f:
            raw_data = pkl.load(f)
        raw_data = np.transpose(raw_data[:, 1:5, :], (0, 2, 1))
        annotation = pd.read_pickle('data/PPG_DaLiA_Annotation.pkl')

    elif DaLia_WESAD == 'WESAD':
        with open('data/WESAD_Raw_Signal.pkl', 'rb') as f:
            raw_data = pkl.load(f)
        raw_data = np.transpose(raw_data, (0, 2, 1))
        annotation = pd.read_pickle('data/WESAD_Annotation.pkl')

    else:
        raise ValueError('Please Choose "WESAD" or "DaLiA" Dataset')

    reference_rr = annotation['Reference_RR'].values.reshape(-1, 1)
    activity_id = annotation['activity_id'].values.reshape(-1, 1)

    # Round raw data and reference respiration rate to 4 decimal places
    raw_data = np.around(raw_data, decimals=4)
    reference_rr = np.around(reference_rr, decimals=4)

    # Split data based on patient ID for training (<10) and testing (>=10)
    training_ids = annotation['patient_id'] < 10

    # Convert training and testing data to TensorFlow tensors
    train_rr_ref = tf.convert_to_tensor(reference_rr[training_ids.values], dtype='float32')
    test_rr_ref = tf.convert_to_tensor(reference_rr[~training_ids.values], dtype='float32')
    train_sig_raw = tf.convert_to_tensor(raw_data[training_ids.values], dtype='float32')
    test_sig_raw = tf.convert_to_tensor(raw_data[~training_ids.values], dtype='float32')

    # Separate activity IDs for training and testing
    train_activity_id = activity_id[training_ids.values]
    test_activity_id = activity_id[~training_ids.values]

    return train_rr_ref, test_rr_ref, train_sig_raw, test_sig_raw, train_activity_id, test_activity_id

def create_datasets(train_sig_raw, train_rr_ref, test_sig_raw, test_rr_ref):
    """
    Creates TensorFlow datasets for training and testing with batching and shuffling.

    Input:
    - train_sig_raw (TensorFlow Tensor): Training raw signal data
    - train_rr_ref (TensorFlow Tensor): Training reference respiration rate
    - test_sig_raw (TensorFlow Tensor): Testing raw signal data
    - test_rr_ref (TensorFlow Tensor): Testing reference respiration rate

    Output:
    - train_dataset (tf.data.Dataset): Shuffled and batched training dataset
    - test_dataset (tf.data.Dataset): Batched testing dataset
    """
    # Create TensorFlow datasets for training and testing with batching and shuffling
    train_dataset = tf.data.Dataset.from_tensor_slices((train_sig_raw, train_rr_ref))
    train_dataset = train_dataset.shuffle(len(train_sig_raw)).batch(128)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_sig_raw, test_rr_ref))
    test_dataset = test_dataset.batch(128)
    return train_dataset, test_dataset

def create_model(model_input_shape):
    """
    Initializes and loads weights for a multi-class CNN model.

    Input:
    - model_input_shape (tuple): Shape of the model input, e.g., (sequence_length, channels)

    Output:
    - model (TensorFlow Model): Initialized multi-class CNN model
    - loss_fn (TensorFlow Loss): Huber loss function
    """
    # Initialize the multi-class CNN model with the given input shape
    model = Multi_class_CNN(model_input_shape)
    # Define the Huber loss function
    loss_fn = Huber()
    # Initialize model weights with a dummy input, then load pretrained weights
    model(tf.ones((128, model_input_shape[0], model_input_shape[1])))
    model.load_weights('RR_Estimation_Model.h5')
    return model, loss_fn

def FineTuning(model, loss_fn, train_dataset, test_dataset):
    """
    Fine-tunes the model on the training dataset with validation on the test dataset.

    Input:
    - model (TensorFlow Model): The initialized CNN model
    - loss_fn (TensorFlow Loss): Loss function for the model
    - train_dataset (tf.data.Dataset): Training dataset
    - test_dataset (tf.data.Dataset): Testing dataset

    Output:
    - model (TensorFlow Model): Fine-tuned CNN model
    """
    # Freeze layers up to the last 5 for fine-tuning
    for layer in model.layers[:6]:
        layer.trainable = False

    # Define scheduler function to adjust learning rate over epochs
    def scheduler(epoch):
        if epoch <= 3:
            lr = 1e-2
        else:
            lr = 1e-4
        return lr

    # Define Adam optimizer and compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=loss_fn)
    num_fine_tuning_epochs = 20  # Number of fine-tuning epochs

    # Track mean loss for training and testing
    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)

    print("Starting fine-tuning...")
    for epoch in range(num_fine_tuning_epochs):
        start = time.time()
        lr = scheduler(epoch)  # Update learning rate based on epoch
        optimizer = Adam(learning_rate=lr)
        print(f"Fine-tuning Epoch [{epoch + 1}/{num_fine_tuning_epochs}]")

        train_loss_list = []  # Collect train loss for each step

        # Training loop for each batch in the dataset
        for step, (x_batch_train_raw, x_batch_train_ref_rr) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                output = model(x_batch_train_raw, training=True)
                loss_value = loss_fn(x_batch_train_ref_rr, output)
                train_loss_list.append(loss_value)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_loss(loss_value)

            # Print loss every 10 steps
            if step % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_fine_tuning_epochs}], Iter [{step + 1}] Loss: {loss_value:.4f}")

        print(f"Fine-tuning net loss -- {np.mean(np.array(train_loss_list))}")

        # Validation loop for testing dataset
        test_loss_list = []
        best_loss = 100000  # Initialize best loss for comparison

        for step, (x_batch_test_raw, x_batch_test_ref_rr) in enumerate(test_dataset):
            test_output = model(x_batch_test_raw)
            test_loss_val = loss_fn(x_batch_test_ref_rr, test_output)
            test_loss(test_loss_val)
            test_loss_list.append(test_loss_val)

        mean_loss = sum(test_loss_list) / len(test_loss_list)
        if mean_loss < best_loss:
            best_loss = mean_loss  # Update best loss if current loss is lower
        print(f"Fine-tuning validation loss -- {mean_loss}")
        train_loss.reset_states()
        test_loss.reset_states()
        end = time.time()
        print(f"Epoch time: {end - start:.2f} seconds")

    print("Fine-tuning complete.")
    return model 

def perform_inference(model, test_dataset):
    """
    Performs inference on the test dataset and collects model predictions.

    Input:
    - model (TensorFlow Model): The trained CNN model
    - test_dataset (tf.data.Dataset): Testing dataset

    Output:
    - final_output (TensorFlow Tensor): Model predictions for the test dataset
    - ref_rr (TensorFlow Tensor): Reference respiration rate for the test dataset
    """
    # Initialize tensors to store outputs and reference respiration rates
    final_output = tf.convert_to_tensor([])
    ref_rr = tf.convert_to_tensor([])

    # Iterate through test batches to collect predictions
    for step, (x_batch_test_raw, x_batch_test_ref_rr) in enumerate(test_dataset):
        output = model(x_batch_test_raw)

        # Concatenate outputs and references across batches
        if step == 0:
            final_output = output
            ref_rr = x_batch_test_ref_rr
        else:
            final_output = tf.concat([final_output, output], axis=0)
            ref_rr = tf.concat([ref_rr, x_batch_test_ref_rr], axis=0)

    return final_output, ref_rr

def calculate_errors(final_output, ref_rr):
    """
    Calculates the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) between predictions and references.

    Input:
    - final_output (TensorFlow Tensor): Model predictions
    - ref_rr (TensorFlow Tensor): Reference respiration rate values

    Output:
    - mae (float): Mean Absolute Error between predictions and reference respiration rate
    - rmse (float): Root Mean Square Error between predictions and reference respiration rate
    - error (numpy array): Array of absolute errors
    - mask (numpy array): Boolean mask indicating non-NaN values
    """
    # Reshape model predictions and reference RR values for error calculation
    final_output_rr = final_output.numpy().reshape(final_output.shape[0], final_output.shape[1])
    avg_ref_breath = ref_rr.numpy().reshape(-1, 1)

    # Calculate absolute error and mask out NaN values
    error = np.abs(avg_ref_breath - final_output_rr)
    mask = ~np.isnan(error)
    # Compute Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
    mae = np.mean(error[mask])
    rmse = np.sqrt(np.mean(error[mask]**2))
    return mae, rmse, error, mask

if __name__ == '__main__':
    # Load the training and testing data (signals, reference respiration rate, and activity ID)
    train_rr_ref, test_rr_ref, train_sig_raw, test_sig_raw, train_activity_id, test_activity_id = load_data('WESAD')

    # Create TensorFlow datasets for training and testing, with batching and shuffling applied
    train_dataset, test_dataset = create_datasets(train_sig_raw, train_rr_ref, test_sig_raw, test_rr_ref)

    # Define model input shape and initialize the model and loss function
    model_input_shape = (2048, 4)  # Example shape: (sequence_length, channels)
    model, loss_fn = create_model(model_input_shape)
    
    # Fine-tune the model on the training dataset with validation on the test dataset
    model = FineTuning(model, loss_fn, train_dataset, test_dataset)

    # Perform inference on the test dataset to get predictions and the reference RR
    final_output, ref_rr = perform_inference(model, test_dataset)

    # Calculate evaluation metrics: Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
    mae, rmse, error, mask = calculate_errors(final_output, ref_rr)

    # Display model summary and evaluation metrics
    model.summary()
    print('Mean Absolute Error is:', mae)
    print('Root Mean Square Error is:', rmse)
    print('The length of the Reference RR is', len(ref_rr))
    print('The length of the final output is', len(final_output))

    # Create a DataFrame to analyze the error by activity and modality
    array_fusion = np.concatenate(
        (error[mask].reshape(-1, 1), np.zeros((len(error[mask]), 1)), test_activity_id[mask].reshape(-1, 1)), axis=1
    )

    final_array = array_fusion
    data_frame = pd.DataFrame(final_array, columns=['Absolute Error(BrPM)', 'Modality', 'Activity_id'])
    data_frame['Modality'] = data_frame['Modality'].astype('category')
    data_frame['Activity_id'] = data_frame['Activity_id'].astype('category')

    # Rename categories for modality and activity ID for better interpretability
    data_frame['Modality'] = data_frame['Modality'].cat.rename_categories(['Final RR'])
    data_frame['Activity_id'] = data_frame['Activity_id'].cat.rename_categories(
        ['Transit', 'Baseline', 'Stress', 'Amusement', 'Meditation']
    )

    # Plot absolute error distribution across activities, grouped by modality
    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(x="Activity_id", y="Absolute Error(BrPM)", hue="Modality", data=data_frame, showfliers=False, width=0.8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    plt.setp(ax.get_legend().get_title(), fontsize=15)
    plt.setp(ax.get_legend().get_texts(), fontsize=16)
    plt.xlabel('Activity', fontsize=20)
    plt.ylabel('Absolute Error(BrPM)', fontsize=15)
    plt.yticks(fontsize=20)
    plt.show()

    # Plot raw signals for visualization, showing one sample's PPG and accelerometer data
    sampling_rate = 64  # Sampling rate of the signals (64 Hz)
    time_array = np.arange(0, len(train_sig_raw[5][:, 0])) / sampling_rate  # Time in seconds
    plt.figure(figsize=(10, 10.5))
    acc_axis = ['x', 'y', 'z']  # Accelerometer axes

    # Plot each channel of the raw data separately: PPG and three accelerometer channels
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(time_array, train_sig_raw[5][:, i])
        plt.xlabel('Time (seconds)')
        if i == 0:
            plt.title('PPG Signal')
        else:
            plt.title('ACC ' + acc_axis[i - 1] + ' Signal')
    plt.show()
