import cv2
import numpy as np
from matplotlib import pyplot as plt

imgOrig = cv2.imread('ATU.jpg')  
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

imgOut1 = cv2.GaussianBlur(imgOrig, (15, 15), 0)  
imgOut2 = cv2.GaussianBlur(imgOrig, (25, 25), 0)  

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB))  
plt.title('Original Image')
plt.xticks([]), plt.yticks([]) 

plt.subplot(2, 2, 2)
plt.imshow(imgGray, cmap='gray') 
plt.title('Grayscale Image')
plt.xticks([]), plt.yticks([]) 

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(imgOut1, cv2.COLOR_BGR2RGB))  
plt.title('GaussianBlur (15x15)')
plt.xticks([]), plt.yticks([]) 

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(imgOut2, cv2.COLOR_BGR2RGB))  
plt.title('GaussianBlur (25x25)')
plt.xticks([]), plt.yticks([]) 

plt.tight_layout()

plt.show()
