!wget https://raw.githubusercontent.com/yotam-biu/ps12/main/image_utils.py -O /content/image_utils.py
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit


from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

import numpy as np
from PIL import Image

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

%%writefile /content/image_utils.py
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


# Step 4: Main Code

# 1. Load an image using the functions from image_utils.py
# First, let's download a sample image to work with.
# The previous URL was broken. Using a new valid URL for a sample image.
!wget -O sample_image.jpg https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png

from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = 'sample_image.jpg'
try:
  original_image = load_image(image_path)
  print(f"Image loaded successfully with shape: {original_image.shape}, dtype: {original_image.dtype}")
except FileNotFoundError:
  print(f"Error: Image file not found at {image_path}. Please provide a valid path.")
  original_image = None
except Exception as e:
  print(f"An error occurred: {e}")
  original_image = None

if original_image is not None:
  # 2. Suppress noise using a median filter
  # Note: skimage.filters.median expects grayscale or single-channel image, or will apply to each channel.
  # For consistency with edge_detection, we'll convert to grayscale first for median filtering.
  # Or, apply median filter on the color image and then convert to grayscale for edge detection.
  # Let's apply it to the color image first, as per the instruction context.

  # Convert to uint8 for skimage if it's not already.
  image_for_median = original_image.astype(np.uint8)
  clean_image = median(image_for_median, ball(3))
  print(f"Noise suppressed. Clean image shape: {clean_image.shape}, dtype: {clean_image.dtype}")

  # 3. Run the noise-free image through the edge_detection function
  edge_magnitude_image = edge_detection(clean_image)
  print(f"Edge detection applied. Resulting shape: {edge_magnitude_image.shape}, dtype: {edge_magnitude_image.dtype}")

  # 4. Convert the resulting edgeMAG array into a binary array
  # Let's visualize the histogram to choose a threshold. For now, a common threshold is used.
  # plt.hist(edge_magnitude_image.ravel(), bins=256, range=(0.0, edge_magnitude_image.max()))
  # plt.title('Histogram of Edge Magnitude Image')
  # plt.show()

  threshold = 50 # This value might need adjustment based on the image
  edge_binary = (edge_magnitude_image > threshold).astype(np.uint8) * 255 # Convert boolean to 0/255
  print(f"Binary edge image created. Shape: {edge_binary.shape}, dtype: {edge_binary.dtype}")

  # 5. Display the binary image and save it as .png file
  plt.imshow(edge_binary, cmap='gray')
  plt.title('Binary Edge Detected Image')
  plt.axis('off')
  plt.show()

  edge_image_pil = Image.fromarray(edge_binary)
  output_filename = 'my_edges.png'
  edge_image_pil.save(output_filename)
  print(f"Binary edge image saved as {output_filename}")


import numpy as np
from scipy.signal import convolve2d

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

%%writefile /content/main.py
import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import matplotlib.pyplot as plt

# Import functions from our custom image_utils.py
from image_utils import load_image, edge_detection

def main():
    print("Starting image processing...")

    # 1. Load an image
    # First, let's ensure a sample image is available. This part assumes 'sample_image.jpg' is already downloaded.
    # In a real GitHub scenario, you might have the image alongside main.py or provide a path.
    image_path = 'sample_image.jpg'
    try:
        original_image = load_image(image_path)
        print(f"Image loaded successfully with shape: {original_image.shape}, dtype: {original_image.dtype}")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}. Please ensure it exists.")
        return
    except Exception as e:
        print(f"An error occurred during image loading: {e}")
        return

    # 2. Suppress noise using a median filter
    # Convert to uint8 for skimage if it's not already, as median expects specific types.
    image_for_median = original_image.astype(np.uint8)
    clean_image = median(image_for_median, ball(3))
    print(f"Noise suppressed. Clean image shape: {clean_image.shape}, dtype: {clean_image.dtype}")

    # 3. Run the noise-free image through the edge_detection function
    edge_magnitude_image = edge_detection(clean_image)
    print(f"Edge detection applied. Resulting shape: {edge_magnitude_image.shape}, dtype: {edge_magnitude_image.dtype}")

    # 4. Convert the resulting edgeMAG array into a binary array
    # A fixed threshold is used here. For more advanced use, this could be dynamic.
    threshold = 50 # This value might need adjustment based on the image
    # Ensure the output is 0 or 255 for proper image saving and display
    edge_binary = (edge_magnitude_image > threshold).astype(np.uint8) * 255
    print(f"Binary edge image created. Shape: {edge_binary.shape}, dtype: {edge_binary.dtype}")

    # 5. Display the binary image and save it as .png file
    output_filename = 'my_edges.png'
    plt.imshow(edge_binary, cmap='gray')
    plt.title('Binary Edge Detected Image')
    plt.axis('off')
    plt.savefig(output_filename)
    print(f"Binary edge image saved as {output_filename}")
    print("Image processing complete.")

if __name__ == '__main__':
    main()

!cat /content/image_utils.py

