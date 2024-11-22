import cv2
import numpy as np
from matplotlib import pyplot as plt

imgOrig = cv2.imread('ATU.jpg')  
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

sobelHorizontal = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=5)  
sobelVertical = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=5)

imgOut1 = cv2.GaussianBlur(imgOrig, (15, 15), 0)  
imgOut2 = cv2.GaussianBlur(imgOrig, (25, 25), 0) 

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

plt.tight_layout()

plt.show()
