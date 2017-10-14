# Traffic Sign Recognition


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
[image2]: ./output/train_evolution_1FD32_L5_E50_B128_R0.0005_A999.png "Evolution for the selected model's hyperparameters"
[image3]: ./output/treated.png "Processed images"
[image4]: ./output/train_loss_1FD32_L5_E50_B128_R0.0005_A999.png "Loss function"
[image5]: ./output/webimages.png "Images from web"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup

This project is available as jupyter notebook [here](./Notebook.ipynb). Also
in [PDF](./report.pdf).

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization

The data set can be plotted as a histogram, showing the number of images for each class.

![alt text][image1]

### Design and Test a Model Architecture

The model used in this project is based on LeNet-5. LeNet consists of 2 convolutional layers, 1 max pool and 2 fully connected layer.
The architecture proposed here adds another fully connected layer and uses a deeper convolutional filter.

Pre-processing was made in two steps:

- grayscaling with: Gray = 0.299 Red + 0.587 Green + 0.114 Blue
- normalizing by pixel intensity (pixel / 255) - 0.5

These steps were applied to training, validation and testing, so images had consistent treatment.
The data set used was the original provided.

![alt text][image3]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|-----------------------|-----------------------------------------------| 
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
 
This architecture was chosen to create a pyramid effect, reducing size of each
layer and increasing depth.

To train the model it was used the Adam Optimizer, which is based on Kingma and 
Ba's algorithm. Best results were found for 50 epochs, batch of 128 and 
learning rate 0.005.

![alt text][image2]

Loss function

![alt text][image4]

Jupyter input cell In[5] contains the python function. The function was coded 
with a factory pattern, a form of closure. This makes it reusable with different 
number of output classes.

Alternative combinations of hyperparameters were tried:

- batch size 128, 50 epochs and learning rate 0.01: validation accuracy 89%
- batch size 256, 100 epochs and learning rate 0.0001: validation accuracy 90%
- batch size 256, 100 epochs and learning rate 0.0005: validation accuracy 93%

My final model results were:

* training set accuracy of 99%
* validation set accuracy of 95% 
* test set accuracy of 92%

To develop the model, an iterative approach was chosen. The first architecture
(original LeNet) could reach 89%. Project target was 93% on validation. So, 
model needed improvements.

To improve the model it was chosen to increase the number of layers, adding
fully connected layers and increasing the convolutional filter's depth.
This approach allows the model to identify higher complexity connections in
data.

The learning rate of 0.01 of found to be too high, and 0.0001 too low, and final
version uses 0.0005. Batch size is more related to the computations then the
results.

The result of 99% in training, 95% on validation and 92% on testing indicates
that model is balanced, but tending to overfiting.

Using images in grayscale avoids problems with image's backgrounds and reduces
the number of layers. But there's a loss of information.

Considering the three accuracy results, the model is shown to behave as expected.


### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image5]

All five images are cleary reconizable by humans. The shape is the same as 
training images, 32x32x3 (RGB) JPEG.

| Image	|     Description	    | Class | Probability |
|-------|-----------------------|-------|-------------|
| 1   	| stop  				|  14 	| 100.% 	  |
| 2     | no vehicles 			|  15 	| 0.13% 	  |
| 3		| Pedestrians			|  27 	| 0.00% 	  |
| 4	    | beware of ice			|  30 	| 0.00% 	  |
| 5	    | Roundabout			|  40 	| 61.6%  	  |

#### Discussions

The model was able to predict 2 out of 5 new images, 40% accuracy. This result
is significantly lower than validation (95%) and previous test (92.8%). One
possible explanation for this difference is the way how the images were resized
from their original shape (width and height). 

For the first image, the model was very confident, 100%, but for image number 5
the confidence was only 62%.



