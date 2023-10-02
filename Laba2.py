import numpy as np 
import cv2 as cv 
from matplotlib import pyplot as plt 

MIN_MATCH_COUNT = 10 

img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE, encoding='cp1251') # queryImage 
img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE, encoding='cp1251') # trainImage 

# Initiate SIFT detector 
sift = cv.SIFT_create() 

# find the keypoints and descriptors with SIFT 
kp1, des1 = sift.detectAndCompute(img1, None) 
kp2, des2 = sift.detectAndCompute(img2, None) 

FLANN_INDEX_KDTREE = 1 
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) 
search_params = dict(checks=50) 
flann = cv.FlannBasedMatcher(index_params, search_params) 
matches = flann.knnMatch(des1, des2, k=2) 

# store all the good matches as per Lowe's ratio test. 
good = [] 
for m, n in matches: 
    if m.distance < 0.7 * n.distance: 
        good.append(m)

# Load and compare the new image 
test_img = cv.imread('test_image.png', cv.IMREAD_GRAYSCALE, encoding='cp1251') 
kp_test, des_test = sift.detectAndCompute(test_img, None) 

matches_test = flann.knnMatch(des1, des_test, k=2) 

good_test = [] 
for m, n in matches_test: 
    if m.distance < 0.7 * n.distance: 
        good_test.append(m)

# Check if the robot is present or not 
if len(good_test) >= MIN_MATCH_COUNT: 
    print("Robot nayden") 
else: 
    print("Robot ne nayden")