**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4
[n_image]: ./test_images/extra5756.png
[p_image]: ./test_images/18.png
[c_image1]: ./output_images/t1_results.png
[c_image2]: ./output_images/t3_results.png
[c_image3]: ./output_images/t4_results.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file `classifier.py` (line 87)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][p_image]
![alt text][n_image]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, by which I evaluate using 3-fold cross validation. There seems to be relatively little differences around the default values of (2x2 cells per block, 8x8 pixels per cell and 9 orientations). 

I eventually settled on 2x2 CPB, 16X16 PPC and 9 orientations for better computational time, and small feature size.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear `SVM sklearn.svm.LinearSVC` using the following features:

1. HOG features with 2 cells per block, 16 pixels per cell and 9 orientation bins.
2. Histogram of intensity/color (32 bins) on each of the 3 color channels
3. Vectorization of the 3 color channels by resizing the image to 32x32 pixels

These can be found on the function `get_features` (line 145) in the file `src/classifier.py`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Based on the assumption that detection is run on a flat road, highway(vehicles are sufficiently spaced) scenario, I began the sliding window search on the bottom half of the image.

To decide on the sizes of the detection windows, I randomly extract frames from the project video and recorded the dimensions in which the entire vehicle would fit into a window. I eventually settled on 64x64, 96x96 and 128x128 pixels. 

The parameter for overlapping windows is also decided on a similar basis. I began on a conservative 75% (on a 64x64 window this implies an overlap of 48 pixels), progressively scaled downwards and observed the detections/non-detections on random frames.

The scale and overlap actually work on tandem. Too many window scales and high overlap increases the number of false positives as well as computation time. On the other hand, lesser windows would mean vehicles are not effectively detected.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

There are ~5967 positive images (I only used the data from the KITTI)  and 8900 negative images for training initially. To balance out the classes and ensure equal representation of positive and negative features, I randomly sampled images from the GTI and Udacity dataset. I also used a learning curve to understand whether I need more data samples, or if I am overfitting/underfitting. 

Ultimately, extracting the HOG, histogram and spatially binned color features from the LUV instead of the "default" RGB channels proved to be especially fruitful. Here are some sample images. The bounding boxes are colored in such a way that the more confident predictions are brighter in color.

![alt text][c_image1]
![alt text][c_image2]
![alt text][c_image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://drive.google.com/open?id=0Bxtv1dvjqwk7Vk54dXNjVmM2UjQ)

##### UPDATE 22-8-2017

I have updated my implementation and uploaded a new [video](https://drive.google.com/open?id=0Bxtv1dvjqwk7U0RNZnN2M0VBSnc). There should be a lesser false positive and false negatives. I made the changes based on the suggestions 

1. using yCrCb color space instead of LUV.
2. Scrapping the use of "tracking" (item 3 in the following section) and using a temporal heatmap instead,


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

There are several measures I took in the pipeline to reduce the number of false positives.

1. Using only *confident* results: I used the function `svc.decision_function()` to use only detections whose data points are far away from the decision hyperplane. This can be found in line 220 inside the function `process_frame()`
2. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. This can be found in line 223.
3. I used a simplified version of object tracking. I treat the bounding boxes found in the step 2 as prospective candidate *tracks* and recorded their x,y positions. In subsequent frames, each of the newly detected boxes is associated with a track if the distance between the x,y center of the box and the tracks is less than a threshold (otherwise, initialize a new track for each unassociated box.) Any tracks unassociated for 4 consecutive frames is discarded. A track's xy position is updated by averaging with their associated boxes. A detected box in current frame is drawn if it is associated with a track and that track has been associated for at least 3 frames previously. This can be found in the function `update_centroids()` on line 152 inside the file `pipeline.py`   

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In my implementation, to reduce computational efforts, the sliding window search is limited to the bottom half of the image frame. For a flat road highway scenario (project video), this work well as vehicles are spaced amply from one another. This approach/heuristic might not work well in city driving conditions where cars might be right in front of the ego-vehicle, or where there are downhill/uphill driving.

Secondly, in various cases, the boxes found are entirely not entirely tightly bounded to the vehicle, or detected boxes only cover portions of the vehicle. Because we can associate the lower corners of the boxes to be touching the flat ground, we can calculate the current dimensions of the vehicle, resulting in noisy estimations.

Finally, in my implementation, I used a simplified version of tracking to remove false positives. However, my future would be to include proper track merging, track/detection association. This would be a more sensible, albeit complicated approach whereby kalman filters can be use for each track to model the x,y location and h,w window size. This will likely result in a more robust and smoother detection.   

