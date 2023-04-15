import numpy as np
import matplotlib.pyplot as plt

# Define the size of the image
size = 100

# Create an empty image with all zeros
image = np.zeros((size, size))

# Set the intensity values of the inner square
inner_size = 50
inner_start = (size - inner_size) // 2
inner_end = inner_start + inner_size
image[inner_start:inner_end, inner_start:inner_end] = 125

# Define the projection functions
def row_projection(image):
    return np.sum(image, axis=1)

def column_projection(image):
    return np.sum(image, axis=0)

def diagonal_projection(image, angle):
    if angle == 45:
        return np.sum(np.diagonal(image))
    elif angle == 135:
        return np.sum(np.diagonal(np.fliplr(image)))

# Define the back projection function
def back_projection(projections, axis):
    size = len(projections)
    image = np.zeros((size, size))
    for i in range(size):
        if axis == 'row':
            image[i, :] = projections[i]
        elif axis == 'column':
            image[:, i] = projections[i]
        elif axis == 'diagonal_45':
            image += np.diag(projections, k=i-1)
        elif axis == 'diagonal_135':
            image += np.diag(projections, k=-(i-size))
    return image

# Calculate the projections
row_proj = row_projection(image)
col_proj = column_projection(image)
diag45_proj = diagonal_projection(image, 45)
diag135_proj = diagonal_projection(image, 135)

# Reconstruct the image using different projections
row_recon = back_projection(row_proj, 'row')
row_col_recon = back_projection([row_proj, col_proj], 'row')
row_col_diag45_recon = back_projection([row_proj, col_proj, diag45_proj], 'row')
row_col_diag_recon = back_projection([row_proj, col_proj, diag45_proj, diag135_proj], 'row')

# Plot the original and reconstructed images
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 1].imshow(row_recon, cmap='gray')
axs[0, 1].set_title('Row Projection')
axs[1, 0].imshow(row_col_recon, cmap='gray')
axs[1, 0].set_title('Row and Column Projections')
axs[1, 1].imshow(row_col_diag_recon, cmap='gray')
axs[1, 1].set_title('Row, Column, and Diagonal Projections')
plt.show()