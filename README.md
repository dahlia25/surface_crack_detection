# Project: Surface Crack Detection

**Affiliation:** San Jose State University <br>
**Program:** M.S. Data Analytics (MSDA) <br>
**Course:** DATA 255 - Deep Learning <br>
**Teammates:** Ashima Malik, Catherine Cho, Dahlia Ma, Jie Hu, Tanyu Yin 
<br><br>

## Introduction
This project compared the performance of the following six image classification neural networks for detecting surface cracks: VGG-16, MobileNet, ResNet50, CNN, LeNet5, and OLeNet. 
<br><br>

## Contributions
|Team Member  |Contributions                                                       |
|-------------|--------------------------------------------------------------------|
|Ashima Malik |MobileNet model, Final Report                                       |
|Catherine Cho|VGG-16 model, Final Report, LaTeX Final Report                      |
|Dahlia Ma    |LeNet15 model, OLeNet model, Data Augmentation, README, Final Report|
|Jie Hu       |CNN model, Final Report                                             |
|Tanyu Yin    |ResNet50 model, Final Report, Topic Proposal                        |
<br>

## Python Run Code Instructions 
To download and run code for your model of interest, navigate to the <u>**code**</u> folder, then select the contributor's folder name that corresponds to the model as listed in the the table in the *Contributions* section. 
<br><br>

**a) Download dataset in the following:** <a href="https://data.mendeley.com/datasets/5y9wdsg2zt/2">METU dataset download</a>
<p>
Dataset contains concrete images collected from METU campus buildings. Contains a total of 40,000 images in which each class (crack / no crack) has 20,000 images with 227 x227 pixels with RGB channels.
</p><br>

**b) Required Python Packages**
* Keras
* Tensorflow
* PyTorch or Torch
* Numpy, Pandas, Matplotlib, Seaborn
* Sklearn
* os, glob, cv2
<br><br>

**c) How to run VGG-16 model (by Catherine Cho)** 
<br><br>
To train the model (using Keras):
1. Download the METU dataset, and change the directory name to be "0" for negative (no crack images) and "1" for positive (has crack images). 
2. The trained model was saved and included as ‘VGG16_crack_detection_v1.h5’ and also the history file was saved as ‘history_vgg16_v1.csv’

To perform test accuracy with the augmented dataset:
1. Load the saved model (‘VGG16_crack_detection_v1.h5’) and, 
2. Load the Pytorch object which contains 1,000 augmented image tensor (‘small_1000_set.pth’)
<br>

**d) How to run MobileNet model (by Ashima Malik)**
<br><br>
