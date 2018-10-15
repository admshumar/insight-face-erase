# insight-face-erase
FaceErase is a deep learning pipeline for face anonymization that I developed while in the Insight Data Science program in Autumn 2018.

This is a first step toward an open-source pipeline for human anonymization in visual media. The ultimate goal is to develop a  tool that can anonymize human faces, license plates, and other sensitive objects in a broad variety of poses, lighting conditions, and scales.

To this end, I chose the <a href="http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/">WIDER FACE dataset</a>, which is a richly annotated collection of over 32,000 images of human faces with a 40/10/50 train-validation-test split for each of its 61 event classes. Since this data set is currently being used in the <a href="http://wider-challenge.org/">Wider Face and Pedestrian Challenge</a>, annotation for the test set is unavailable. Thus my data consisted only of the train and validation subsets.

The model is a PyTorch implementation of <a href="https://arxiv.org/abs/1505.04597">U-Net</a>, with a few modifications. U-Net is a fully-convolutional neural network that was originally designed for biomedical image segmentation. Since its introduction in 2015, it has found broader applications in computer vision, especially in image segmentation outside of a biomedical context. (For an example, see the <a href="https://www.kaggle.com/c/carvana-image-masking-challenge">Carvana Image Masking Challenge</a>.)

<p align="center">
  <img src="https://github.com/admshumar/insight-face-erase/blob/master/img/unet.png">
  </p>

The above image comes from the original 2015 paper and is a rough approximation of the architecture that I used. My architecture takes a one-channel 256x256 image as input, and it returns a one-channel 256x256 image mask. The network has 21 layers and incorporates batch normalization after each 3x3 convolution operation. The convolutions are preceded by same padding, and max-pooling is performed with a 2x2 kernel and a stride of 2. The effect of this is to halve both image dimensions when downsampling.

<p align="center">
<img src="https://github.com/admshumar/insight-face-erase/blob/master/img/example_masks.png" width="600" height="600">
  </p>
  
The top-left image is an input to U-Net. The top right is its mask label. The bottom-left image is an output. The reason for this appearance is that U-Net outputs a pixel-by-pixel probability for the presence of that pixel in the "face" class. One sees that pixels that are located within regions that correspond to the whitespaces of the mask label are given a non-zero value. The image on the bottom-right is a binary mask that is produced with a threshold value which depends on the distribution of pixel intensities in U-Net's output. This mask is then applied to the input image to occlude faces. Finally, the occlusions are inpainted with OpenCV to produce a smoother anonymization. To summarize:

<p align="center">
<img src="https://github.com/admshumar/insight-face-erase/blob/master/img/pipeline2.png" width="600" height="370">
  </p>

Potential use cases involve any company that would need to comply with privacy regulations, such as the European Union's <a href="https://en.wikipedia.org/wiki/General_Data_Protection_Regulation">GDPR</a>. Consider, for example, a photography/film company that intends to disseminate visual media featuring indivduals who did not consent to an appearance in its footage. Such a company would be bound by law to anonymize any such person prior to dissemination. There is a strong incentive to comply, as penalities for non-compliance can be considerable.
