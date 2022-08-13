# BraTS2020CapstoneProject
The dataset of my project, which i prepared as a graduation project of Bootcamp, was taken from the Kaggle site.
 

# Brain Tumor Segmentation
31.07.2022
─
Ahmet ÇAKMAK
 
# Overview
All BraTS multimodal scans are available as NIfTI files (.nii.gz) -> commonly used medical imaging format to store brain imagin data obtained using MRI and describe different MRI settings
T1: T1-weighted, native image, sagittal or axial 2D acquisitions, with 1–6 mm slice thickness.
T1c: T1-weighted, contrast-enhanced (Gadolinium) image, with 3D acquisition and 1 mm isotropic voxel size for most patients.
T2: T2-weighted image, axial 2D acquisition, with 2–6 mm slice thickness.
FLAIR: T2-weighted FLAIR image, axial, coronal, or sagittal 2D acquisitions, 2–6 mm slice thickness.
Data were acquired with different clinical protocols and various scanners from multiple (n=19) institutions.
All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1), as described both in the BraTS 2012-2013 TMI paper and in the latest BraTS summarizing paper. The provided data are distributed after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution (1 mm^3) and skull-stripped.

# Goals
Segmentation of gliomas in pre-operative MRI scans.
Each pixel on image must be labeled:
Pixel is part of a tumor area (1 or 2 or 3) -> can be one of multiple classes / sub-regions
Anything else -> pixel is not on a tumor region (0)
The sub-regions of tumor considered for evaluation are: 1) the "enhancing tumor" (ET), 2) the "tumor core" (TC), and 3) the "whole tumor" (WT) The provided segmentation labels have values of 1 for NCR & NET, 2 for ED, 4 for ET, and 0 for everything else.


# EXPLORATORY ANALYSIS

There are 369 rows in the dataset. In row 355, 1 file has a different name, first I fixed it manually. Then I did a first look by taking all the columns in 1 row. Then I looked at their shape. The size of the images is (240x240x155). It refers to 155 slices here, you can examine the picture below for better understanding.
![Slices] (C:\Users\4hmet\Desktop\CapstoneProject\one picture.png "Slices")

There is one more thing here. In the image that is our label (seg.nii), 5 classes are specified, but the 3rd class here does not correspond to anything. For this reason, I re-specified the classes. Here I make a view according to different slices.

I'm starting to think that Flair, t1ce and t2 images are more efficient at explaining our tag in my views.
![First Looking] (C:\Users\4hmet\Desktop\CapstoneProject\NewModelFirstLooking.png "First Looking")
 


# MODEL 


As a result of the literature review, i first wanted to build a Deep Learning model. In the studies on the same data set, i saw that the U-net architecture was used a lot, especially in segmentation studies, and i started by building a simple model on it (details will be given in the presentation). First, i built a model over all image formats, and then I reduced the number of variables to only flair and t1ce, as i mentioned above. You can find detailed information about the U-net architecture i use and the loss functions i use below.
“The u-net is convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. It has won the Grand Challenge for Computer-Automated Detection of Caries in Bitewing Radiography at ISBI 2015, and it has won the Cell Tracking Challenge at ISBI 2015 on the two most challenging transmitted light microscopy categories (Phase contrast and DIC microscopy) by a large margin (See also our annoucement).” [Check Here] (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
![U-net] (C:\Users\4hmet\Desktop\CapstoneProject\u-net-architecture.png)


You can find here more things about U-net. [here] (https://github.com/christianversloot/machine-learning-articles/blob/3995782892d6f34b70c139265acdfa1c7b9ee07e/u-net-a-step-by-step-introduction.md) 


Based on similar studies, 2 success metrics were used in this model. These are the dice coefficient and the Mean IOU. You can find detailed information about these metrics [here]. (https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c)

# MODEL STEPS
We look at the dataset for the first time and we get outputs like this.
![Images] (C:\Users\4hmet\Desktop\CapstoneProject\NewModel 50-50 slices.png "Images")
1[Masks] (C:\Users\4hmet\Desktop\CapstoneProject\NewModelOriginalSegmentation.png "Masks")
I define functions for our model that will make the dataset we want. Besides these functions, we also define our metrics. Here I will use the dice coefficient that I mentioned above. The dice coefficient is especially used in binary classifications, since we have multiple classes, we make it suitable for each class. We also define precision function as another metric we will use. In order to optimize the dataset, we define functions that combine the variables that we will use to get better performance and save resources. Then we apply min max scaling and transform our dataset in this way. We divide the new data set into train, test and validation in the form of distribution as follows.
![Train,Test,Validation] (C:\Users\4hmet\Desktop\CapstoneProject\NumberOfImages.png "Train,Test,Validation")

We use the callbacks functions in the keras library to keep our model under control while it is running.
I tried many different models and you can find the ones I tried on my github profile. I found this model as the best model and the success metrics of the model are as follows.
![Curves] (C:\Users\4hmet\Desktop\CapstoneProject\Model26-6 LAYERS.png "Curves")
![Predict] (C:\Users\4hmet\Desktop\CapstoneProject\PredictModel26-6layers.png "Predict")


Evaluate on test data 45/45 [==============================] - 61s 1s/step - loss: 0.0296 - accuracy: 0.9905 - mean_io_u_6: 0.8138 - dice_coef: 0.4243 - precision: 0.9938

[Here is the work I referenced] (https://www.kaggle.com/code/rastislav/3d-mri-brain-tumor-segmentation-u-net/notebook?scriptVersionId=61189746) 
[Also you can find his profile] (https://github.com/rastislavkopal)
