# --------------------------------------------------------------------------------------------------
# EE569 Homework Assignment #4
# Date: March 19, 2019
# Name: Suchismita Sahu
# ID: 7688176370
# email: suchisms@usc.edu
# --------------------------------------------------------------------------------------------------

import numpy as np
from matplotlib import pylab as plt
import cv2


# Read the Raw Images:
height = 1024
width= 768
raw_image = open('river1.raw','rb').read()
raw_image = np.frombuffer(raw_image, np.uint8)
river1_img = np.reshape(raw_image, (height, width,3))

raw_image = open('river2.raw','rb').read()
raw_image = np.frombuffer(raw_image, np.uint8)
river2_img = np.reshape(raw_image, (height, width,3))

#Calculate SIFT Features using OpenCV
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(river1_img,None)
kp2, des2 = sift.detectAndCompute(river2_img,None)

#Project the SIFT features on Input Images
img = cv2.drawKeypoints(river2_img, kp2,None,color=-1, flags=4)
plt.figure(figsize = (8,10))
plt.imshow(img)
plt.show()

img = cv2.drawKeypoints(river1_img, kp1,None,color=-1, flags=4)
plt.figure(figsize = (8,10))
plt.imshow(img)
plt.show()

# Get the Keypoints with Largest L2 norm
larg_kp_norm = np.linalg.norm(des1, axis=1)
larg_des = np.argmax(larg_kp_norm)

# Highlight the Largest Scale Keypoint
img = cv2.drawKeypoints(river1_img, [kp1[larg_des]],None,color=255, flags=4)
plt.figure(figsize = (8,10))
plt.imshow(img)
plt.show()

# Brute Force Image Matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(np.array([des1[larg_des]]),des2, k=1)

img3 = cv2.drawMatchesKnn(river1_img, [kp1[larg_des]], river2_img, kp2, matches, None, matchColor= (0,255,0), flags=2)
plt.figure(figsize = (8,10))
plt.imshow(img3)
plt.show()
# Note: cv2.drawMatchesKnn expects list of lists as matches.


# FLANN based Image Matching
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(np.array([des1[larg_des]]),des2,k=2)

# Create Mask to Draw only the Important Matches
matchesMask = [[0,0] for i in range(len(matches))]

# Ratio Test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    #if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   matchesMask = matchesMask,
                   flags = 2)

img3 = cv2.drawMatchesKnn(river1_img,[kp1[larg_des]],river2_img, kp2, matches,None,**draw_params)
plt.figure(figsize = (10,10))
plt.imshow(img3)
plt.show()

print('Orientation of key point in river image1', kp1[larg_des].angle)
print('orientation of matching key point in river image2', kp2[matches[0][0].trainIdx].angle)
