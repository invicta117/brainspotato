# Applications of CNNs for the Detection of Brain Cancer in CT Scans

The following code is adapted from the project Applications of CNNs for the Detection of Brain Cancer in CT Scans. The aim of this project is to determine if a patient is healthy of suffering from brain cancer by training the model on a subset of the CT data and to then determining the accuracy of the model based on its predictions using the remainder of this data. The techniques discussed in the paper demonstrate CNN as a powerful technique to identify cancer in patients. Techniques such as this could undoubtedly contribute significantly to modern medicine.

Custom data augmentation techniques are provided in this code for 3D CT data, the techniques used are: left-right flip, random angle rotation and value augmentation. Files for unzipping the data, pre-processing the data and the final model are also provided. The final model in this code is composed of 3 individual CNNs which are combined to increase sensitivity and specificity.

## How to Implement

This project was built using Python 3.7 and using the following:

OpenCV 4.2.0.32

Tensorflow 1.15.0

Tqdm 4.26.0

Scipy 1.1.0

Skimage 0.16.2

Pydicom 1.4.2

Pandas 0.23.4

## Acknowledgements

Data Science Bowl 2017 Kaggle competition in particular, Sentdex for his notebook: Applying a 3D convolutional neural network to the data, as well as Ankasor for his notebook: Improved Lung Segmentation using Watershed
