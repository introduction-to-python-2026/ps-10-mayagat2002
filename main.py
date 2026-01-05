from PIL import Image 
from image_utils import load_image, edge_detection 
import numpy as np 
from skimage.filters import median 
from skimage.morphology import ball
my_image = load_image('dogphoto-jpg' )
image_gray_temp = np.mean(my_image, axis=2)
image_cleaned = median (image_gray_temp, ball(3))
final_edges = edge_detection(my_image)
threshold_value - 50
edge_binary = final_edges › threshold_value
binary_image_data = edge_binary-astype(np.uint8) * 255
edge_image = Image. fromarray (binary_image_data )
edge_image. save( 'Newphoto-jpg*)
