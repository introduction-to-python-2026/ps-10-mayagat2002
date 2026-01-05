import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def load_image(path):
    """
    Load image and return numpy array (do NOT force RGB).
    """
    img = Image.open(path)
    return np.array(img)


def edge_detection(image):
    """
    Edge detection using given kernels and zero padding.
    """

    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # Kernels EXACTLY as defined in the assignment
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

    edgeY = convolve2d(gray, kernelY, mode='same',
                       boundary='fill', fillvalue=0)

    edgeX = convolve2d(gray, kernelX, mode='same',
                       boundary='fill', fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
