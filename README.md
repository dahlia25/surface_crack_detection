# Project: Surface Crack Detection

**Affiliation:** San Jose State University <br>
**Program:** M.S. Data Analytics (MSDA) <br>
**Course:** DATA 255 - Deep Learning <br>
**Teammates:** Ashima Malik, Catherine Cho, Dahlia Ma, Jie Hu, Tanyu Yin 
<br><br>

## Introduction
This project compared the performance of the following six image classification neural networks for detecting surface cracks: VGG-16, MobileNet, ResNet50, CNN, LeNet5, and OLeNet. 

The project report generated using LaTeX can be found in the **project_report** directory.
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

**c) part 1: How to perform image augmentation**
1. Download the "data_augmentation.ipynb" file located in **code/dahlia** directory.
2. In the downloaded file, update the following variables to the correct file path set up in your local:
    * **img_files**: the directory file path to either negative (0) or positive (1) images
    * **img_folder**: the directory file path to either negative (0) or positive (1) images
    * **pos_img_save_folder**: the directory file path to save positive augmented images
    * **pos_img_src_folder**: the directory file path that contains the positive (crack) images from original dataset
    * **neg_img_save_folder**: the directory file path to save negative augmented images
    * **neg_img_src_folder**: the directory file path that contains the negative (no crack) images from original dataset
3. After updating the mentioned variables in step 2, run all code cells in Jupyter notebook.
<br><br>

**c) part 2: Selecting 1000 augmented images as extra test set**
* You can download the "small_1000_set.pth" file (located in a <a href="https://drive.google.com/file/d/1pYUV_8Pd656YnewAWOyW6pSttnmE7jbV/view?usp=sharing">Google Drive here</a>; as file is too large to push to GitHub with Git large file storage) that is a Pytorch object that contains a tensor with 1000 augmented images for further evaluating test accuracy, OR
* If you cannot download the "small_1000_set.pth" file, then download and run the "create_augmented_test_set.ipynb" file in the same directory. 

To run the "create_augmented_test_set.ipynb" file, update the following variables:
* **neg_data_dir**: the file path of the directory that contains original negative images
* **pos_data_dir**: the file path of the directory that contains original positive images
* **neg_aug_data_dir**: the file path of the directory that contains augmented negative images
* **pos_aug_data_dir**: the file path of the directory that contains augmented positive images
* **data_dir**: the file path of the directory containing the original data (both negative and positive images)
* **aug_data_dir**: the file path of the directory containing the augmented data (both negative and positive images)
* **path**: the file path to store the selected 1000 augmented images as tensor in Pytorch object
<br><br>

**d) How to run VGG-16 model (by Catherine Cho)** 
<br>
To train the model (using Keras):
1. Download the METU dataset, and change the directory name to be "0" for negative (no crack images) and "1" for positive (has crack images). 
2. The trained model was saved and included as ‘VGG16_crack_detection_v1.h5’ and also the history file was saved as ‘history_vgg16_v1.csv’

To perform test accuracy with the augmented dataset:
1. Load the saved model (‘VGG16_crack_detection_v1.h5’) and, 
2. Load the Pytorch object which contains 1,000 augmented image tensor (‘small_1000_set.pth’)
<br><br>

**e) How to run MobileNet model (by Ashima Malik)**
<br>
To train the model:
1. After downloading the METU dataset, place the Positive and Negative directories under data/Train directory. 
2. The trained model was saved under the **code/ashima** directory named as "Crack_Detection_MobileNet_model.h5", and the history file was saved as "history_MobileNet_model.csv".

To perform test accuracy with the augmented dataset:
1. Load the saved model (‘Crack_Detection_MobileNet_model.h5’) and, 
2. Load the Pytorch object which contains 1,000 augmented image tensor (‘small_1000_set.pth’)
<br><br>

**f) How to run ResNet50 model (by Tanyu Yin)**
<br>
To train the model:
1. After downloading the METU dataset, create negative/positive directories to store the dataset by class; where "Negative" refers to 0, and "Positive" refers to 1. 
2. Use the "ResNet50.ipynb" file located in **code/Tanyu** directory, and update the file path to your own. Then run the whole notebook.

To perform test accuracy with the augmented dataset:
1. Download and open "ResNet50_augmt.ipynb" located in **code/Tanyu** directory.
2. Load the saved model (‘ResNet50_model.h5’).
3. Load the Pytorch object which contains 1,000 augmented image tensor (‘small_1000_set.pth’), and run the codes afterwards.
<br><br>

**g) How to run CNN model (by Jie Hu)**
<br>
To preprocess and organize dataset:
1. After downloading the METU dataset, create negative/positive directories to store the dataset by class; where "Negative" refers to 0, and "Positive" refers to 1. 
2. Download and open the "Crack_Images_augument.ipynb" file located in **code/hujie** directory. 
3. Use the "data_split.ipynb" to choose 10,000 random images from augmented data, and then save all selected images to local (where "Negative" refers to 0 and "Positive" refers to 1 in directory name to store images for testing).

To train and test the model:
1. Use the "Project_crack.ipynb" file located in **code/hujie** directory, and 
2. Update the data file path to your own. Then run the whole notebook.
<br><br>

**h) How to run LeNet5 and OLeNet models (by Dahlia Ma)**
<br>
To train the model:
1. Download the IPYNB file with the model architecture of your interest, following are the options:
    * In **code/dahlia/lenet5_models** directory:
        * **LeNet5_relu_model_noAugmentation_lr0.001.ipynb**: this contains LeNet5 model with ReLU and learning rate of 0.001 (final model)
        * **LeNet5_tanh_model_noAugmentation_lr0.001.ipynb**: this contains LeNet5 model with Tanh and learning rate of 0.001
        * **LeNet5_tanh_model_noAugmentation_lr0.01.ipynb**: this contains LeNet5 model with Tanh and learning rate of 0.01
        * **LeNet5_tanh_model_noAugmentation_lr0.1.ipynb**: this contains LeNet5 model with Tanh and learning rate of 0.1
    * In **code/dahlia/olenet_models** directory:
        * **OLeNet_model_relu_noAugmentation_lr0.001.ipynb**: this contains OLeNet model with ReLU and learning rate of 0.001
        * **OLeNet_model_selu_noAugmentation_lr0.001.ipynb**: this contains OLeNet model with SeLU and learning rate of 0.001 (final model)
2. To run any of the selected IPYNB notebooks, update the following variables to your own file path:
    * **data_dir**: the file path of the directory that contains both positive (1) and negative (0) images from the original METU dataset -- in 4th cell
    * **path**: the file path of the directory to split and save the test set of the original data to -- in 5th cell with the comment “generate and save test set”
    * **path**: the file path of the directory to save the trained model at each epoch -- in 8th cell that trains the model
        * you can use "lenet5_lr0.001_cpu_epoch7.pth" in the **code/dahlia/lenet5_models** directory to load saved final model for LeNet5 with ReLU
        * you can use "olenet_lr0.001_cpu_epoch4.pth" in the **code/dahlia/olenet_models** directory to load saved final model for OLeNet with SeLU
    * **path**: the file path of the saved trained model (.PTH Pytorch file) at a specific epoch -- in 11th cell under “calculate test accuracy” section
        * you can use "lenet5_lr0.001_cpu_epoch7.pth" in the **code/dahlia/lenet5_models** directory to load saved final model for LeNet5 with ReLU
        * you can use "olenet_lr0.001_cpu_epoch4.pth" in the **code/dahlia/olenet_models** directory to load saved final model for OLeNet with SeLU
    * **testset_path**: the file path of the split and saved test set of original data (.PTH file) generated from the 5th cell in 2b -- in 11th cell under “calculate test accuracy” section; this variable is not applicable to OLeNet models, as Google Colab could not load the saved splitted test set and calculate the accuracy due to the limited 12 GB of RAM
    * In the last code cell in IPYNB notebook that evaluates test accuracy using augmented test set (if this code cell exists), update **“path”** and **“testset_path”** variables accordingly like in step 2e; except, update the file path for “testset_path” that leads to the “small_1000_set.pth” augmented test set
3. Run the IPYNB notebook to train model
    * If you are not running the code in Google Colab as in the original file, then do not run the first code cell that mounts to your Google Drive.
    * If you have already splitted, generated and saved the test set from original data from another Jupyter model training notebook, then no need to run the 5th code cell (“generate and save test set”) again.