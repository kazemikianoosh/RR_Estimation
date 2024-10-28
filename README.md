# Deep Transfer Learning Approach for Respiration Rate Estimation
End-to-End explainable RR estimation method based on multi-scale CNN for noise-corrupted PPG and tri-axial ACC signals. 

# Description
The goal this project is develop a transfer learning-based method for Respiration Rate estimation using raw PPG and tri-axial ACC collected with smartwatches. The proposed method comprised two parts. In the first part, a CNN-based network, including Multi-scale convolutions augmented with dilated residual inception, 1D convolution layers, and dense layers, was trained using the Health Monitoring dataset to build the pre-trained model. The Multi-Scale Convolution enhanced the feature extraction ability across different scales and effectively mitigated the vanishing gradient problem. Moreover, increasing the receptive field of the inception blocks through dilated residuals enhanced the model’s efficiency. In the second part, the pre-trained model was fine-tuned on the target dataset. The model was evaluated on two benchmark datasets (i.e., PPG-DaLiA and WESAD) collected with wrist-worn devices. We performed several experiments to compare the efficacy and accuracy of our model with four existing state-of-the-art methods. Moreover, the performance of the methods across different activity types was assessed. Furthermore, the performance of the proposed method was studied from data quality perspective using the SNR metric, revealing that our model obtained the lowest error throughout the entire SNR range. Finally, by using the SHAP method, we demonstrated that PPG and y-axis ACC signals had more contribution in the final RR estimation.
![pipeline2](https://github.com/kazemikianoosh/RR_Estimation/assets/51022509/854210b7-df19-4ded-85e1-ff9874d748e5)
*Proposed RR estimation pipeline including pre-training phase and fine-tuning phase.*
# Model Architecture
Our foundation model architecture includes two major components: 1) Segmentation and Normalization and 2) Multi-Scale Residual CNN. A view of the architecture is illustrated in Figure below. The Segmentation and Normalization are applied to raw PPG and tri-axial ACC before feeding them into the deep learning model. Multi-Scale Residual CNN: To estimate RR, We employ and customize a deep neural network, to derive RR from the signals. This method consists of two distinct modules: a Multi-Scale Residual Convolution module and an RR estimator module.

![model](https://github.com/kazemikianoosh/RR_Estimation/assets/51022509/f0fc2bf8-e183-4c80-a36d-bf58e5974986)
# Data collection
We use a dataset obtained as part of a Health Monitoring study to train the model. Participants were instructed to wear Samsung Gear Sport smartwatches on their non-dominant wrists and a Shimmer3 deviceon their chests for continuous 24-hour data collection as shown in Figure below. During data collection, participants were engaged in their typical daily activities to track their vital signs, physical activity and sleep data
![model2](https://github.com/kazemikianoosh/RR_Estimation/assets/51022509/e74a82b4-4b32-491c-9330-a6f6ff170e0c)

## Repository Structure
```
.
├── RR_Estimation
│   ├── requirements.txt
│   ├── RR_Estimation_Model.h5
│   ├── sample_code.py
│   ├── tf_model.py
│   ├── data
│   │   ├── PPG_DaLiA_Annotation.pkl
│   │   ├── WESAD_Annotation.pkl
│   │   └── link to the processed PPG DaLiA and WESAD Signals
│   │       └── (https://drive.google.com/drive/folders/1uQAfajvmxtSCSRP6Ihc_tDlgsITaePBI?usp=drive_link)
```

# Example usage
Steps to run the example localy:
### Datasets

  1. Install the Requirements Packages

> Run `pip install -r requirements.txt` to install the necessary Python Packages.

  2. [Processed PPG DaLiA and WESAD Dataset](https://drive.google.com/drive/folders/1uQAfajvmxtSCSRP6Ihc_tDlgsITaePBI?usp=drive_link)
  3. download the data file.
  24 Run sample_code.py
     *                      train_rr_ref, test_rr_ref, train_sig_raw, test_sig_raw, train_activity_id, test_activity_id = load_data()*              
       - This line will import the data and create the data for training and testing phase
      
         
      *                     train_dataset, test_dataset = create_datasets(train_sig_raw, train_rr_ref, test_sig_raw, test_rr_ref)*              
       - Create the tensor files for feeding the model
    
     
![DaLiA_activity_type](https://github.com/kazemikianoosh/RR_Estimation/assets/51022509/fe2d51be-879d-4070-8a10-cc7648c4db47)




