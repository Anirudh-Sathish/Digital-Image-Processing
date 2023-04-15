# Anirudh Sathish 
# CS20B1125 
# Assignment 7 

# Load libs 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

# Load the dog and lena images 
dog = cv2.imread("dog.jpg")
lena = cv2.imread("lena.png")

# convert to gray 
dog_gray = cv2.cvtColor(dog,cv2.COLOR_BGR2GRAY)
lena_gray = cv2.cvtColor(lena,cv2.COLOR_BGR2GRAY)

# convert both to grayscale 
freq_dog = np.fft.fft2(dog_gray)


# Shift the zero-frequency component to the center of the spectrum
fft_shifted = np.fft.fftshift(freq_dog)

# Compute the magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fft_shifted))

# Normalize the magnitude spectrum to the range [0, 255]
magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display the magnitude spectrum
cv2.imshow('Magnitude Spectrum', magnitude_spectrum_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

