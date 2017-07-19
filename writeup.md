
# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./writeup_images/training_set_histogram.png "Training set"
[image2]: ./writeup_images/validation_set_histogram.png "Validation set"
[image3]: ./writeup_images/class_images.png "Images from all classes"
[image4]: ./writeup_images/training_set_image.png "Original image"
[image5]: ./writeup_images/grayscaled_training_set_image.png "Grayscaled image"
[image6]: ./writeup_images/hist_eq_training_set_image.png "Histogram equalized image"
[image7]: ./writeup_images/normalized_training_set_image.png "Normalized image"
[image8]: ./writeup_images/speed_limit_30.jpg "Traffic Sign 1"
[image9]: ./writeup_images/slippery_road.jpg "Traffic Sign 2"
[image10]: ./writeup_images/wild_animals_crossing.jpg "Traffic Sign 3"
[image11]: ./writeup_images/stop.jpg "Traffic Sign 4"
[image12]: ./writeup_images/road_work.jpg "Traffic Sign 5"
[image13]: ./writeup_images/lenet.png "Lenet architecture"
[image14]: ./writeup_images/softmax_class_1.png
[image15]: ./writeup_images/softmax_class_23.png
[image16]: ./writeup_images/softmax_class_31.png
[image17]: ./writeup_images/softmax_class_14.png
[image18]: ./writeup_images/softmax_class_25.png
[image19]: ./writeup_images/accuracy_curves.png "Accuracy curves"
[image20]: ./writeup_images/loss_curves.png "Loss curves"
[image21]: ./writeup_images/feature_maps_1.png "Convolution layer 1 feature maps"
[image22]: ./writeup_images/feature_maps_2.png "Convolution layer 2 feature maps"
[image23]: ./writeup_images/feature_map_image.png "Test image"

### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. 


### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/sheelabhadra/CarND-Traffic-Sign-Classifier-P2)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy library to calculate summary statistics of the [German traffic signs](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) data set:

* The size of training set is 34799.
* The size of the validation set is 12630.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing how the data is distributed by the labels in the training set.

![alt text][image1]

We can observe that some classes such as Class 1 and Class 2 have a lot of data samples (about 2000 each) while some classes such as Class 0 and Class 19 have relatively much fewer data samples (about 200 each). This difference in the number of samples in a particular class may lead to the neural network learning more from the data belonging to the class with more number of samples the class than other classes that have relatively lower number of data samples. This makes the network biased towards a few classes during testing.

We can also observe sample images form all the classes to get some familiarity with the data along with their corresponding labels.

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I converted the images to grayscale because grayscaling removes clutter from an image. Since the problem includes only only classifying images, grayscaling allows the feature maps to concentrate only on the subject under interest. Also, grayscaling converts a 3-channel RGB image to a single channel image which reduces the computation complexity. Grayscaling was achieved by using the OpenCV's cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]
![alt text][image5]

There were quite a few images that had low contrast and hence the signs were not clearly visible. This issue can be solved using histogram equalization which can be achieved using OpenCV's cv2.equalizeHist(image). This basicaly enhances tthe contrast of the images and makes the traffic signs more clear. More information about histogram equalization can be found [here](http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html). So, in the next step I applied histogram equalization on the grayscaled images.

Here is an example of a traffic sign image before and after histogram equalization.

![alt text][image6]

In the last step, I normalized the image data using the formula: (image/255) - 0.5. This step is necessary because normalization helps in making the neural network converge faster since the variation in data is restricted within a specific range (in this case between -0.5 and 0.5). An in-depth discussion on image data pre-processing techniques and their significance is provided in the CS231 [course notes](http://cs231n.github.io/neural-networks-2/).

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used a slightly modified version of the [LeNet-5](http://yann.lecun.com/exdb/lenet/) architecture which is a simple 5 layer Convolutional Neural Network with 2 convolution layers, 2 fully connected layers, and a Softmax layer. The diagram below shows LeNet's architecture.

![alt text][image13]


I added a few dropout layers in between to reduce overfitting. My final model consisted of the following layers:  

| Layer           | Kernel Size | Number of filters | Stride    |      Description               | Padding  |
|:---------------:|:-----------:|:-----------------:|:---------:|:------------------------------:|:--------:|
| Input           |     -       |       -           |       	|  32x32x1 pre-processed image   |    -     |   
| Convolution     |     5x5     |	    6           |   1x1     |  outputs 28x28x6 	             | Valid    | 
| RELU            |		-       | 		-			|    -	    |  Activation function           |    -     |
| Max pooling     |	    2x2     |   6               |   2x2     |  outputs 14x14x6 	             |	Valid	|
| Dropout         |	    -       |     -             |      -    |  keep probability (0.7) 	     |	   -	|
| Convolution     |     5x5     |	16              |   1x1     |  outputs 10x10x16              | Valid    | 
| RELU            |		-       | 		-			|    -	    |  Activation function           |    -     |
| Max pooling     |	    2x2     |   16              |   2x2     |  outputs 5x5x16 	             |	Valid	|
| Dropout         |	    -       |     -             |      -    |  keep probability (0.7) 	     |	   -	|
| Flattening      |	    -       |     -             |      -    |  outputs 400 	                 |	   -	|
| Fully Connected |	    -       |     -             |      -    |  400 input, 120 output 	     |	   -	|
| RELU            |	    -       |     -             |      -    |  Activation function 	         |	   -	|
| Dropout         |	    -       |     -             |      -    |  keep probability (0.7)  	     |	   -	|
| Fully Connected |	    -       |     -             |      -    |  120 input, 84 output 	     |	   -	|
| RELU            |	    -       |     -             |      -    |  Activation function 	         |	   -	|
| Dropout         |	    -       |     -             |      -    |  keep probability (0.7)  	     |	   -	|
| Softmax         |	    -       |     -             |      -    |  84 input, 43 output	         |	   -	|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters:  
optimizer: Adam  
batch size: 128  
number of epochs: 50

The other hyperparameters apart from the Kernel size and stride used were:  
learning rate: 0.001  
mean (for weight initialization): 0  
stddev (for weight initialization): 0.1  

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.4%.
* validation set accuracy of 97.3%.
* test set accuracy of 94.5%.

The following are my model's accuracy and loss curves:

![alt text][image19]

![alt text][image20]

I chose an iterative approach to modify the Le-Net architecture to get the results:  

**What was the first architecture that was tried and why was it chosen?**  
I chose the [**LeNet-5 architecture**]((http://yann.lecun.com/exdb/lenet/)) as a starting point since it works well on hand-written digits.   

**What were some problems with the initial architecture?**  
There was a lot of overfitting with the initial architecture after feeding the network with the pre-processed data. The training accuracy was about 98% and the validation set accuracy was about 93%.  

**How was the architecture adjusted and why was it adjusted?**   
To reduce overfitting, I used 2 dropout layers after the 2 fully connected layers. I also used dropout layers after the max-pooling layers after experimentation since it reduced overfitting further. I experimented with different values for the keep probability and 0.7 seemed to provide the best validation accuracy on my architecture. Due to time constraints, I couldn't experiment with different dropout rates for different layers which would have led to a more finely tuned network.

**Which parameters were tuned? How were they adjusted and why?**  
I increased the number of epochs because the images were not as simple as handwritten digits with 10 classes. The traffic sign dataset contained 43 classes and the complexity of the images was also higher. To encode the information in this training set into the CNN required more nuber of epochs of training. After 10 epochs the validation set accuracy obtained was 93%, after 20 epochs the validation set accuracy increased to 95%, and after 50 epochs the validation set accuracy increased to 96.2%. This shows that the more number of epochs we train, the better the network learns the training data.

**What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?**  
Convolution layers are useful if the data contains regional similarity i.e. the arrangement of data points gives us useful insight about the spatial information contained in the daata. Images particularly contain useful spatial information which is an important feature. Convolution layers help in extracting these features.  

Dropout turns off or sets the activation in some of the neurons in a layer to 0 randomly. This helps the network learn only the most important features in the images ignoring very minute details which might be noise and may change from image to image even belonging to the same class. This prevents the network from overfitting to the training data. It increases the accuracy on the validation set and hence the testing set. So, the network performs better on new images.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
Here are five German traffic signs that I found on the web:

![alt text][image8] 
![alt text][image9] 
![alt text][image10] 
![alt text][image11] 
![alt text][image12]

In general the images acquired from the web have a higher resolution than the images in the training dataset. These images have a lot of objects apart from just traffic signs. Since, my model does not work on newly seen images, I manually cropped the images so that it only contained the traffic sign. Some of the images collected from the ineternet have watermarks embedded on them which distorts the image and adds unwanted noise to them. Furthermore, my model only accepts inputs with an aspect ratio of 1:1 and of size 32x32. Hence, I resized them in order to fit them into my model which lead to a loss of detail in the images. These are a few issues during data pre-processing.  

The first image should not be difficult for the model to classify because the characteristics (the number "30") of the traffic sign are simple. But, it should also be kept in mind that there a lot of images in the training dataset that belong to the speed-limit class. So, the model might classify it incorrectly failing to distinguish it from other speed limit traffic signs. 

The second image might be challenging for the model to classify because it contains details. But, at the same time it is also very different from the rest of the class of images.

The third image should be difficult for the model to classify because the original image has details such as the deer's antlers and legs which might be lost while down-sampling to 32x32. This might make it look similar to the "Turn left ahead" trafiic sign.

The fourth image should be easy for the model to classify since the characteristics (the "STOP" letters) of the traffic sign are very distinct from other traffic signs.

The fifth image might be challenging for the model to classify because it contains details. But, at the same time it is also very different from the rest of the class of images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)  						| 
| Slippery road     	| Slippery road						            |
| Wild animals crossing	| Dangerous curve to the left					|
| Stop	      		    | Stop					 				        |
| Road work			    | Road work      							    |


The model was able to correctly guess 4 of these 5 traffic signs, which gives an accuracy of 80%. I actually downloaded 20 images from the internet and got correct prediction on 16 images which gives an overall accuracy of 80%. Although it is far lower than the accuracy on the test set, I feel that the major reason for the lower accuracy than the actual testing accuracy is that the images downloaded from the internet were originally of higher overall quality (resolution) than the training and testing images. On resizing the images to 32x32x3, some important pixels in traffic signs with detail e.g. road work, slippery road, wild animals crossing etc. were lost which might have led to lower accuracty on the internet images.  

But, I was pretty confused as to why sometimes the model was not very certain while identifying speed limit signs. Since, the Le-Net network used was primarily used for digit recognition, I hoped that it would perform very well on speed limit signs which was not the case.

#### 3 . Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is very certain that this is a **30 km/h speed limit** sign (probability of 0.99), and the image does actually contain a **30 km/h speed limit** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         		    | Speed limit (30km/h)   						| 
| 0.00     			    | Speed limit (70km/h) 							|
| 0.00					| Speed limit (20km/h)							|
| 0.00	      			| Speed limit (50km/h)					 		|
| 0.00				    | Speed limit (100km/h)      					|

![alt text][image14]
          
For the second image, the model is relatively certain that this is a **Slippery road** sign (probability of 0.89), and the image does contain a **Slippery road** sign. I did not expect the model to be so certain about this sign as after squeezing the image to a size of 32x32x3 some crucial pixels belonging to the *road* and the *car* in the image were lost that I felt were important in identifying the sign. The top five soft max probabilities were
        
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.54         		    | Slippery road 						        | 
| 0.29     			    | Dangerous curve to the right           		|
| 0.14					| Beware of ice/snow				            |
| 0.01	      			| Children crossing			                    |
| 0.00				    | Right-of-way at the next intersection         |

![alt text][image15]

For the third image, the model is certain that this is a **Dangerous curve to the right** sign (probability of 0.36), while the image actually contains a **Wild animals crossing** sign. I did not expect the model to be certain about this sign as after squeezing the image to a size of 32x32x3 some pixels belonging to the antlers of the deer were missing that I felt are crucial in identifying the sign. In fact, after reducing the size of the image the sign looks very similar to the class i.e. Dangerous curve to the left. But, the correct class for the image is the 2nd most probable sign as predicted by the model which is as expected. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.35         		    | Dangerous curve to the right  				| 
| 0.27     			    | Wild animals crossing							|
| 0.11					| Bumpy road	                                |
| 0.08	      			| Slippery road			                        |
| 0.05				    | Go straight or left                           |

![alt text][image16]

For the fourth image, the model is very certain that this is a **Stop** sign (probability of 0.99), and the image does actually contain a **Stop** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         		    | Stop  						                | 
| 0.00     			    | Turn right ahead	 							|
| 0.00					| Turn left ahead						        |
| 0.00	      			| No entry		                                |
| 0.00				    | Keep right                                    |

![alt text][image17]

For the fifth image, the model is relatively certain that this is a **Road work** sign (probability of 0.77), and the image does contain a **Road work** sign. I did not expect the model to be so certain about this sign as after squeezing the image to a size of 32x32x3 some pixels belonging to the *spade* that the person in the image is holding were missing that I felt were crucial in identifying the sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.42         		    | Road work  						            | 
| 0.18     			    | Beware of ice/snow 				            |
| 0.12					| Dangerous curve to the right                  |
| 0.11	      			| Right-of-way at the next intersection			|
| 0.03				    | Wild animals crossing                         |

![alt text][image18]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
I took an image from the test images downloaded from the internet and fed it to the network.

![alt text][image23]

The feature maps obtained in convolution layer 1 are

![alt text][image21]

It can be observed that the feature maps of the first convolution layer encode the shape of the traffic sign and the edges of the image details, in this example the edges around the letters.  

The feature maps obtained in convolution layer 2 are

![alt text][image22]

It seems that the feature maps of the second convolution layer encode more complex information and other minute details present in the traffic sign. It is difficult to comprehend what exactly the convolution filters are capturing in this layer. 

```python

```
