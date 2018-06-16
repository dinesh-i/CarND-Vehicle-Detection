# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, my goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4) and produce a detailed writeup of the project. 

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique/hog sub-sampling and use trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (initially with the test_video.mp4 and later on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/input_image890.jpg
[image2]: ./output_images/output_image_with_boxes_image890.jpg
[image3]: ./output_images/output_image890.jpg
[video1]: ./vehicles_tracked_on_project_video.mp4	

**Histogram of Oriented Gradients (HOG)
The method get_hog_features() gets the hog features of the image for the specified single channel or all the channels. It supports visualization of the resultant hog image and cell#5 has a sample hog image of a car. Other utility methods like bin_spatial() and color_hist() are used from the lessons to extract spatial and color histogram features.

Below are the different values of the parameters used:
* color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
* orient = 9  # HOG orientations  e.g.: 9
* pix_per_cell = 8 # HOG pixels per cell e.g: 8
* cell_per_block = 2 # HOG cells per block e.g: 2
* hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
* spatial_size = (16, 16) # Spatial binning dimensions e.g: (16,16)
* hist_bins = 16    # Number of histogram bins e.g: 16
* spatial_feat = True # Spatial features on or off
* hist_feat = True # Histogram features on or off
* hog_feat = True # HOG features on or off


Below are some of the other combinations of values I've tried and measured the execution time and validation accuracy for 2000 samples each from vehicles and not-vehicles category:

color_space | orient | pix_per_cell | cell_per_block | hog_channel | spatial_size | hist_bins | Execution Time | Validation Accuracy |
:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
YUV | 9 | 8 | 2 | ALL | (16, 16) | 16 | 56.69 | 0.9775 |
YUV | 12 | 8 | 2 | ALL | (16, 16) | 16 | 60.31 | 0.97 |
YUV | 12 | 16 | 4 | ALL | (16, 16) | 16 | 21.49 | 0.9762 |
YUV | 12 | 16 | 4 | 0 | (16, 16) | 16 | 12.03 | 0.9788 |
YUV | 12 | 16 | 4 | 1 | (16, 16) | 16 | 11.76 | 0.9538 |
YUV | 12 | 16 | 4 | 2 | (16, 16) | 16 | 11.92 | 0.9612 |
RGB | 12 | 16 | 4 | ALL | (16, 16) | 16 | 21.14 | 0.9838 |
RGB | 12 | 16 | 4 | 0 | (16, 16) | 16 | 11.93 | 0.9788 |
RGB | 12 | 16 | 4 | 1 | (16, 16) | 16 | 12.17 | 0.965 |
RGB | 12 | 16 | 4 | 2 | (16, 16) | 16 | 11.83 | 0.965 |
HSV | 12 | 16 | 4 | ALL | (16, 16) | 16 | 20.44 | 0.9775 |
HSV | 12 | 16 | 4 | 0 | (16, 16) | 16 | 12.05 | 0.955 |
HSV | 12 | 16 | 4 | 1 | (16, 16) | 16 | 11.61 | 0.975 |
HSV | 12 | 16 | 4 | 2 | (16, 16) | 16 | 11.86 | 0.9825 |
LUV | 12 | 16 | 4 | ALL | (16, 16) | 16 | 22.16 | 0.9825 |
LUV | 12 | 16 | 4 | 0 | (16, 16) | 16 | 12.95 | 0.9762 |
LUV | 12 | 16 | 4 | 1 | (16, 16) | 16 | 12.92 | 0.9562 |
LUV | 12 | 16 | 4 | 2 | (16, 16) | 16 | 12.94 | 0.9588 |
HLS | 12 | 16 | 4 | ALL | (16, 16) | 16 | 20.65 | 0.9838 |
HLS | 12 | 16 | 4 | 0 | (16, 16) | 16 | 11.67 | 0.9612 |
HLS | 12 | 16 | 4 | 1 | (16, 16) | 16 | 11.33 | 0.975 |
HLS | 12 | 16 | 4 | 2 | (16, 16) | 16 | 12.25 | 0.9762 |
YCrCb | 12 | 16 | 4 | ALL | (16, 16) | 16 | 20.71 | 0.9788 |
YCrCb | 12 | 16 | 4 | 0 | (16, 16) | 16 | 11.59 | 0.9688 |
YCrCb | 12 | 16 | 4 | 1 | (16, 16) | 16 | 12.49 | 0.96 |
YCrCb | 12 | 16 | 4 | 2 | (16, 16) | 16 | 11.98 | 0.955 |
HLS | 9 | 8 | 2 | 1 | (16, 16) | 16 | 21.79 | 0.9838 |

****Training:
The train() method has the implementation to train a Linear SVC classifier. The classifier uses the hog, bin spatial and color histogram features and trains the model based on the label set provided for the categories vehicles and non-vehicles.


**Sliding Window Search
The find_cars() method extracts features using hog sub-sampling and make predictions.
 The parameters y_start and y_stop decide the y-axis range of the image to be searched. This will help us to avoid searching unwanted areas of the image like sky, tree top etc. and focus only on the region where cars will potentially appear. 
This method extracts the hog, spatial and color histogram features. The rectangles where the presence of car is predicted is returned by this method.
The method pipeline() attempts to find cars using the following scales in the given y-pixel ranges. Note that each of the y-pixel ranges has an overlap.

Scale|y_start|y_stop|
:--:|:--:|:--:|
1 | 350 | 450|
2 | 350 | 550|
3 | 500 | 680|

Heatmap approach is followed to sum up the heat map values from each of the above results.


The pipeline() method has the logic to invoke find_cars() with different scale and y values. The result of each of the invocations are considered together to apply the heat map and a threshold filter.

I've introduced a flag called "snapshot_needed" which let me to take snapshot of the given image when it passes through various stages of the pipeline. This helped me to get insight into the processing within the pipeline. 

|Input Image| Potential Vehicles Highlighted | Output Image After Threshold is Applied |
:--:|:--:|:--:|
![alt text][image1] | ![alt text][image2] | ![alt text] [image3] |
![alt text][image1]        |  ![alt text][image2]
The last 5 cells in the notebook shows different stages of the image in the pipeline. 

To filter false positives, the image heatmap map was averaged over 4 consecutive frames. This idea was adapted from https://github.com/darienmt/CarND-Vehicle-Detection-P5 

Even after the above implementation, lot of false positives were reported. I tried out different values of threshold to balance between false positives and true negatives. A value of 8 almost removed all false positives and failed to detect car in many scenarios. A value of 3 resulted in false positives and identified cars in most of the frames. So I decided to stick to the threshold value of 3.

** Video Implementation
The video implementation uses the same pipeline() method described above. Identified vehicles are highlighted with a rectangular box. 

This is a link to the project video processed by the pipeline -> [video1]

**Open Challenges
* There are still false positives showing up in the video frame and vehicles are not identified in some cases too. Images with shadows are also detected as vehicles.
* The execution time for the project video processing was around 20m 55s. This time should be improved further.
