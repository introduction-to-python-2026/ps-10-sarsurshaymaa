#from PIL import Image
# import numpy as np
# from scipy.signal import convolve2d

# def load_image(path):
#     pass # Replace the `pass` with your code

# def edge_detection(image):
#     pass # Replace the `pass` with your code


import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(file_path):
  """
  Loads a color image from the given file path and converts it to a NumPy array.

  Args:
    file_path (str): The path to the image file.

  Returns:
    np.array: The image as a NumPy array.
  """
  img = Image.open(file_path)
  return np.array(img)

def edge_detection(image_array):
  """
  Performs edge detection on a color image array.

  Args:
    image_array (np.array): The input color image as a NumPy array (H, W, 3).

  Returns:
    np.array: The edge magnitude array.
  """
  # Convert to grayscale by averaging the three color channels
  grayscale_image = np.mean(image_array, axis=2)

  # Define kernelY (vertical change detection)
  kernelY = np.array([
      [1, 2, 1],
      [0, 0, 0],
      [-1, -2, -1]
  ])

  # Define kernelX (horizontal change detection)
  kernelX = np.array([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
  ])

  # Apply convolution with zero padding
  edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
  edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)

  # Compute edge magnitude
  edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

  return edgeMAG
