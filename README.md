# Lane-Line-Detection
![Lane detection output](https://user-images.githubusercontent.com/68200593/165641851-3e0d67db-66d9-4aff-aa86-96fb17bc7e6f.JPG)
## Overview
Detect lanes using computer vision techniques. This project is provided by Faculty of Engineering Ain Shams University at 4th Mechatronics Image Processing Course.

he following steps were performed for lane detection:

* Change the image to HSV.
* Apply a filter to detect only the yellow and white colors.
* Change the output of the filters to garyscale.
* Apply Canny operation to the grayscale images to detect boundaries.
* Select a specific region to apply the lane detection process.
* Determine the lane boundaries.
* Determine the vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of vehicle position.

The output video is 'Lane Detection Output.mp4' in this repo. The original video is 'project_video.mp4' which is found in the project_data.

## Dependencies
* Python 3.10
* Numpy
* OpenCV-Python

## How to run
* Run `Lane_Detection.py`.
* If you want to run the code on an image , Press 'P' or 'p', this will take the raw image file at 'Project_data/test_images/straight_lines1.jpg'
* if you want to run the code on a video , Press 'V' or 'v', this will take the raw video file at 'Project_data/project_video.mp4', and if you want to record the output Press 'Y' or 'y' to create an annotated output video at 'Lane Detection Output.mp4'.

## Note
To resume the code after processing on an image, you have to close all of the opened windows.

## Lane detection pipeline

# Change the image to HSV
![HSV](https://user-images.githubusercontent.com/68200593/165647444-579f462b-d9ac-435b-8871-b5f891cae8a2.JPG)

# Apply a filter to detect only the yellow colors
![yellow color detection](https://user-images.githubusercontent.com/68200593/165647487-7edebb8d-1677-4edc-87ec-edd6c69d14d3.JPG)

# Change the output of the filters to garyscale
![Gray image of yellow color detection](https://user-images.githubusercontent.com/68200593/165647584-574c66bc-eb58-40df-9d90-bd37426c72c3.JPG)

# Apply Canny operation to the grayscale images to detect boundaries
![Canny image of yellow color detection](https://user-images.githubusercontent.com/68200593/165647642-f285c7e3-afc8-4ae6-b427-6d3b5b44dddf.JPG)

# Apply a filter to detect only the white colors
![white color detection](https://user-images.githubusercontent.com/68200593/165647708-1134f7d2-202d-46a9-b4f7-82baa4c710e0.JPG)

# Change the output of the filters to garyscale
![Gray image of white color detection](https://user-images.githubusercontent.com/68200593/165647745-3ccbede6-09f1-4322-a2a6-13755ec0db3f.JPG)

# Apply Canny operation to the grayscale images to detect boundaries
![Canny image of white color detection](https://user-images.githubusercontent.com/68200593/165647767-cfc0da7c-eaa9-4229-9d1b-6135bd9e96b1.JPG)

# Select a specific region to apply the lane detection process
![Cropped sction of the main image](https://user-images.githubusercontent.com/68200593/165647806-d9826fc7-43c5-4964-928a-bf41aefe4df2.JPG)

# Determine the lane boundaries and vehicle position with respect to center
![Lane detection output](https://user-images.githubusercontent.com/68200593/165647885-20f24a70-9f71-4234-8c66-fad44ed82940.JPG)

## Discussion
This is an initial version of lane detector. There are multiple scenarios where this lane detector would not work. For example, the challenge video includes roads with cracks which could be mistaken as lane lines (see 'challenge_video.mp4'). Also, it is possible that other vehicles in front would trick the lane finder into thinking it was part of the lane. More work can be done to make the lane detector more robust.
