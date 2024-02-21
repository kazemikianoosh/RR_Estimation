# Deep Transfer Learning Approach for Respiration Rate Estimation
End-to-End explainable RR estimation method based on multi-scale CNN for noise-corrupted PPG and tri-axial ACC signals. 

# Description
The goal this project is develop a transfer learning-based method for Respiration Rate estimation using raw PPG and tri-axial ACC collected with smartwatches. The proposed method comprised two parts. In the first part, a CNN-based network, including Multi-scale convolutions augmented with dilated residual inception, 1D convolution layers, and dense layers, was trained using the Health Monitoring dataset to build the pre-trained model. The Multi-Scale Convolution enhanced the feature extraction ability across different scales and effectively mitigated the vanishing gradient problem. Moreover, increasing the receptive field of the inception blocks through dilated residuals enhanced the model’s efficiency. In the second part, the pre-trained model was fine-tuned on the target dataset. The model was evaluated on two benchmark datasets (i.e., PPG-DaLiA and WESAD) collected with wrist-worn devices. We performed several experiments to compare the efficacy and accuracy of our model with four existing state-of-the-art methods. Moreover, the performance of the methods across different activity types was assessed. Furthermore, the performance of the proposed method was studied from data quality perspective using the SNR metric, revealing that our model obtained the lowest error throughout the entire SNR range. Finally, by using the SHAP method, we demonstrated that PPG and y-axis ACC signals had more contribution in the final RR estimation.

![DaLiA_activity_type](https://github.com/kazemikianoosh/RR_Estimation/assets/51022509/fe2d51be-879d-4070-8a10-cc7648c4db47)

![model2](https://github.com/kazemikianoosh/RR_Estimation/assets/51022509/e74a82b4-4b32-491c-9330-a6f6ff170e0c)
![model](https://github.com/kazemikianoosh/RR_Estimation/assets/51022509/f0fc2bf8-e183-4c80-a36d-bf58e5974986)
![pipeline2](https://github.com/kazemikianoosh/RR_Estimation/assets/51022509/854210b7-df19-4ded-85e1-ff9874d748e5)
