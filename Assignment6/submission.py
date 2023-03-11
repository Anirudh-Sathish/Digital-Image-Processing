# Assignment 6 
# Author : Anirudh Sathish 
# Roll No : CS20B1125 

# importing nececsary libaries 
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import random 

# read the image 
lena_img = cv2.imread('Lena.png')

# convert lena to greyscale 
lena_gray = cv2.cvtColor(lena_img,cv2.COLOR_BGR2GRAY)
plt.imshow(lena_gray,cmap='gray')
#plt.show()


# add salt and pepper noises to the image with noise quantities from 0.1 to 1 , gnerating  10 images 

# function for salt and pepper nosie 
def saltAndPepper(img,noiseQuantity):
	# create an ooutput copy 
	resultant = np.copy(img)

	#run through all the pixels 
	for i in range(img.shape[0]): 
		for j in range(img.shape[1]):
			# random number to take decisions 
			randomness = random.random()
			if randomness < noiseQuantity: 
				resultant[i][j] = 0 
			elif randomness >( 1 - noiseQuantity):
				resultant[i][j] = 255 
	# return the resultant image 
	return resultant 

# test out the noise 
noisy_im = saltAndPepper(lena_gray,0.2)
plt.imshow(noisy_im,cmap = 'gray')
#plt.show()


# let us generate subplots for all noise images 
noisy_levels = [ x/10 for x in range(1,11)]

# for the above noise levels let us generate images and put it up on a subplot 
plt.figure(figsize = (20,10))

# loop through to generate all noisy images 
imageList = []  
for i in range(len(noisy_levels)):
	noisy_img = saltAndPepper(lena_gray,noisy_levels[i])
	imageList.append(noisy_img)

# now display all the images in a subplot 
for i in range(len(imageList)): 
	plt.subplot(5,2,i+1)
	plt.imshow(imageList[i],cmap = 'gray')
	title = "Salt and Pepper noise at " + str(noisy_levels[i])
	plt.title(title)
	plt.xticks([]), plt.yticks([])
# display the subplots 
plt.show()

# clear the plot 
plt.clf()

# Also do the same using  lib functions 

# create gaussian filters with variance = 1 , and sizes 3*3 , 5*5 , 7*7 
variance = 1

# lets create the required filter sizes 
kdims = [3,5,7]

# lets create the required filters 
gaussianKernels = []
for i in range(len(kdims)):
	kernel = cv2.getGaussianKernel(kdims[i],variance)
	gaussian_kernel = np.outer(kernel,kernel.transpose())
	gaussianKernels.append(gaussian_kernel)

#Applying the 3*3 gaussian filter on all noisy images , using built in function 
# -> performing correlation 
plt.figure(figsize= (20,10))
for i in range(len(imageList)):
	gaussian = cv2.filter2D(imageList[i],-1,gaussianKernels[0])
	plt.subplot(2,5,i+1)
	title = "Kernel : 3*3 , noise = " + str(noisy_levels[i])
	plt.title(title)
	plt.imshow(gaussian,cmap = 'gray')
plt.show()
plt.clf()

# for 5*5 
plt.figure(figsize= (20,10))
for i in range(len(imageList)):
        gaussian = cv2.filter2D(imageList[i],-1,gaussianKernels[1])
        plt.subplot(2,5,i+1)
        title = "Kernel : 5*5 , noise = " + str(noisy_levels[i])
        plt.title(title)
        plt.imshow(gaussian,cmap = 'gray')
plt.show()
plt.clf()

# for 7*7 
plt.figure(figsize= (20,10))
for i in range(len(imageList)):
        gaussian = cv2.filter2D(imageList[i],-1,gaussianKernels[2])
        plt.subplot(2,5,i+1)
        title = "Kernel : 7*7 , noise = " + str(noisy_levels[i])
        plt.title(title)
        plt.imshow(gaussian,cmap = 'gray')
plt.show()


#kdim = 3 
#kernel = cv2.getGaussianKernel(kdim, variance)
#gaussian_kernel = np.outer(kernel,kernel.transpose())
#gaussian = cv2.filter2D(imageList[1],-1,gaussian_kernel)
#plt.imshow(gaussian,cmap = 'gray')
#print(gaussian)
#plt.show()
