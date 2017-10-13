#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The steps of this project are the following:

* Load the data set. Data sets are Python binary files (pickle) with traffic 
signs images;
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/histogram.png "Histogram"
[image2]: ./output/evolution_1FD32_L5_E50_B128_R0.0005_A958.png "Evolution for the selected model's hyperparameters"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

This project is available as jupyter notebook [here](./Notebook.ipynb). Also
in [PDF](./report.pdf).

###Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####Exploratory visualization

The data set can be plotted as a histogram, showing the number of images for each class.

![alt text][image1]

###Design and Test a Model Architecture

The model used in this project is based on LeNet-5. LeNet consists of 2 convolutional layers, 1 max pool and 2 fully connected layer.
The architecture proposed here adds another fully connected layer and uses a deeper convolutional filter.

Pre-processing was made in two steps:

- grayscaling with: Gray = 0.299 Red + 0.587 Green + 0.114 Blue
- normalizing by pixel intensity (pixel / 255) - 0.5

This steps were applied to training, validation and testing, so images had consistent treatment.
The data set used was the original provided.


My final model consisted of the following layers:

|:---------------------:|:---------------------------------------------:| 
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32	 				|
| Fully connected		| Flat input 800, outputs 400					|
| RELU					|												|
| Fully connected		| Input 400, outputs 129						|
| RELU					|												|
| Fully connected		| Input 129, outputs 86							|
| RELU					|												|
| Fully connected		| Input 86, outputs 43							|
|:---------------------:|:---------------------------------------------:| 
 


To train the model it was used the Adam Optimizer (which is based on Kingma and Ba's algorithm. Best results were found for 50 epochs, batch of 128 and learning rate 0.005.

![alt text][image2]

