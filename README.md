# insight-face-erase
A deep learning pipeline for face anonymization that I developed during Autumn 2018 in the Insight Data Science program.

This is a first step toward an open-source pipeline for human anonymization in visual media. The ultimate goal is to develop a  tool that can anonymize human faces, license plates, and other sensitive objects in a broad variety of poses, lighting conditions, and scales.

To this end, I chose the <a href="http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/">WIDER FACE dataset</a>, which is a richly annotated collection of over 32,000 images of human faces with a 40/10/50 train-validation-test split for each of its 61 event classes. Since this data set is currently being used as part of the <a href="http://wider-challenge.org/">Wider Face and Pedestrian Challenge</a>, annotation for the test set is unavailable. Thus my data consisted only of the train and validation subsets.

The model is a PyTorch implementation of U-Net, a deep neural network that was originally designed for biomedical image segmentation.
