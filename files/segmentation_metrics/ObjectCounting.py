import cv2
import numpy as np
from path_names import PathNamesSegmented as pns
from skimage import morphology
from matplotlib import pyplot as plt


image = cv2.imread(pns.GAUSSIAN_MODERATE, 0)
image = image.astype(bool)
image = morphology.remove_small_objects(image,15)
image = image.astype(np.uint8)
ret, con = cv2.connectedComponents(image)
plt.imsave("objects.png",con)
print(ret - 1)


