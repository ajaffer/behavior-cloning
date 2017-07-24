#**Behavioral Cloning** 

##Writeup

### The aim of this project is to train a CNN that can autonomously drive the car in the simulated evnironment. 

---

**Behavioral Cloning Project**

I did the following steps:

* Used the simulator to collect the following data points: Driving in the center, driving in the opposite direction, recovering from driving over the lane lines, driving on the track two.
* Built two convolution neural networks in Keras that predicts steering angles from images
* Trained and validated the model with a training and validation set
* Tested that the model can successfully drive around track one without leaving the road

[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./center.jpg "Example Center Image"
[image3]: ./recover-1.jpg "Recovery Image 1"
[image3]: ./recover-2.jpg "Recovery Image 2"
[image3]: ./recover-3.jpg "Recovery Image 3"
[image6]: ./orig.jpg "Normal Image"
[image7]: ./flipped.jpg "Flipped Image"
[image8]: ./history.png "History"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
Use this command to create and train the model: 
```sh 
python model.py --use_adv_measurements=false --use_lenet=false --model_name=model.h5 --use_side_images=true --data=data,data-turning,data-recovering,data-recover-3,data-recover-2,data-turning-2
```
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* this file: summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 use_adv_measurements=false
```

####3. Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
I researched using LeNet and nvidia CNNs. I chose the nvidia one (line #136)

My nvidia model consists of 5 convolution neural network, the first two have a 5x5 filter sizes and the next three have a 3x3 filter and depths between 50 and 100 (model.py lines 136-147) 

The model includes RELU layers to introduce nonlinearity (code lines 137, 139, 141, 142, 143), and the data is normalized in the model using a Keras lambda layer (code line 121). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
Following image shows how both the training and validtion errors were low, showing that the model is not overfitting.
![alt text][image8]


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 159).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, dirving smoothly during curves, driving in the opposite direction and driving on the track two. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to take small steps and fix the issues, e.g., car driving straight on a curve and falling into water.

My first step was to use a convolution neural network model similar to LeNet I thought this model might be appropriate because the lane lines can be thought to have features similar to the features of letters and numbers, i.e., lines and curves.

I tried many ideas on how to make the car drive naturally around the track. I experimented with using what I called 'advanced measurements' in addition to just using the steering angle, i.e., throttle, brake and speed. Using these 'advanced measurement' did not provide better results. Both the training and validation errors were higher than before. 

Then I used the CNN from the this paper: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes

    Convolution 24
    Convolution 36
    Convolution 48
    Convolution 64
    Convolution 64
    Fully Connected 100
    Fully Connected 50
    Fully Connected 1

Here is a visualization of the architecture
![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center in case it goes over the lane lines.  These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would give more data points and make the learning generalized. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 17944 number of data points. I then preprocessed this data by cropping 50/20 rows pixels from the top/bottom of the image, and then centered around zero with small standard deviation. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
