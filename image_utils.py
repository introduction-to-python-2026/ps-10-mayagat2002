from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    """
    Loads a color image from the given file path and returns it as a NumPy array.
    """
    img = Image.open(path).convert("RGB")  # ensure color image
    img_array = np.array(img)
    return img_array

def edge_detection(image):
    """
    Performs edge detection on a color image array.
    Returns the edge magnitude image.
    """

    # 1. Convert RGB image to grayscale by averaging channels
    gray = np.mean(image, axis=2)

    # 2. Define filters
    kernelY = np.array([
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
    ])

    kernelX = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    # 3. Apply convolution with zero padding and same output size
    edgeY = convolve2d(gray, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray, kernelX, mode='same', boundary='fill', fillvalue=0)

    # 4. Compute edge magnitude
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
