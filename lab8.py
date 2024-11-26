import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the original image
imgOrig = cv2.imread('ATU.jpg')  
# Convert to grayscale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Sobel edge detection
sobelHorizontal = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=5)  
sobelVertical = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=5)

# Sobel combined edges
sobelEdges = cv2.magnitude(sobelHorizontal, sobelVertical)

# Apply Canny edge detection
cannyThreshold = 100
cannyParam2 = 200
canny = cv2.Canny(imgGray, cannyThreshold, cannyParam2)

# Apply Gaussian Blur with different kernel sizes
imgOut1 = cv2.GaussianBlur(imgOrig, (15, 15), 0)  
imgOut2 = cv2.GaussianBlur(imgOrig, (25, 25), 0) 

# Second Image 
imgSecond = cv2.imread('ATU_gray.jpg')

# Sobel thresholding with nested loops
threshold_value = 50
sobelThresholded = np.zeros_like(sobelHorizontal)
for i in range(sobelHorizontal.shape[0]):
    for j in range(sobelHorizontal.shape[1]):
        if abs(sobelHorizontal[i, j]) > threshold_value:
            sobelThresholded[i, j] = 255
        else:
            sobelThresholded[i, j] = 0

# Plot the results
nrows = 3
ncols = 3

plt.figure(figsize=(10, 5))

plt.subplot(nrows, ncols, 1)
plt.imshow(sobelHorizontal, cmap='gray')
plt.title('Sobel Horizontal Edges')
plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 2)
plt.imshow(sobelVertical, cmap='gray')
plt.title('Sobel Vertical Edges')
plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 3)
plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB))  
plt.title('Original Image')
plt.xticks([]), plt.yticks([]) 

plt.subplot(nrows, ncols, 4)
plt.imshow(imgGray, cmap='gray') 
plt.title('Grayscale Image')
plt.xticks([]), plt.yticks([]) 

plt.subplot(nrows, ncols, 5)
plt.imshow(cv2.cvtColor(imgOut1, cv2.COLOR_BGR2RGB))  
plt.title('GaussianBlur (15x15)')
plt.xticks([]), plt.yticks([]) 

plt.subplot(nrows, ncols, 6)
plt.imshow(cv2.cvtColor(imgOut2, cv2.COLOR_BGR2RGB))  
plt.title('GaussianBlur (25x25)')
plt.xticks([]), plt.yticks([]) 

plt.subplot(nrows, ncols, 7)
plt.imshow(sobelEdges, cmap='gray')
plt.title('Sobel Edge Magnitude')
plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 8)
plt.imshow(canny, cmap='gray')
plt.title('Canny Edges')
plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 9)
plt.imshow(sobelThresholded, cmap='gray')
plt.title('Sobel Thresholded')
plt.xticks([]), plt.yticks([])

plt.tight_layout()

plt.show()
