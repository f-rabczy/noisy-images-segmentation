import cv2
import numpy as np
from path_names import PathNamesSegmentation as pns
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import morphology

OUTPUT_FOLDER_MASK = pns.SEGMENTED + "k-means\\mask\\image4"
OUTPUT_FOLDER_SEGMENTED = pns.SEGMENTED + "k-means\\segmented\\image4"

def do_Kmeans(file_path, output_path):

    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    k = 4
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    ret, label, center = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape(image.shape)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2GRAY)
    list = np.unique(result_image)

    new_image = np.zeros((512, 512, 3), dtype=np.uint8)
    new_image[result_image == list[3]] = [255, 255, 255]
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)

    new_image = cv2.morphologyEx(new_image, cv2.MORPH_ERODE, np.ones((2, 2)))
    new_image = new_image.astype(bool)
    new_image = ndi.binary_fill_holes(new_image, np.ones((3, 3)))
    new_image = morphology.remove_small_objects(new_image, 50)
    new_image = new_image.astype(np.uint8)

    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    mask[new_image == 0] = [255, 255, 255]
    mask = cv2.bitwise_not(mask)
    segmented = cv2.bitwise_and(image, mask)
    plt.imsave(OUTPUT_FOLDER_SEGMENTED + output_path, segmented)
    plt.imsave(OUTPUT_FOLDER_MASK + output_path, mask)


do_Kmeans(pns.DEFAULT_IMAGE, "\\default_image4.png")

do_Kmeans(pns.GAUSSIAN_LOW, "\\gaussian\\gaussian_low.png")
do_Kmeans(pns.GAUSSIAN_MODERATE, "\\gaussian\\gaussian_moderate.png")
do_Kmeans(pns.GAUSSIAN_HIGH, "\\gaussian\\gaussian_high.png")

do_Kmeans(pns.LAPLACIAN_LOW, "\\laplacian\\laplacian_low.png")
do_Kmeans(pns.LAPLACIAN_MODERATE, "\\laplacian\\laplacian_moderate.png")
do_Kmeans(pns.LAPLACIAN_HIGH, "\\laplacian\\laplacian_high.png")

do_Kmeans(pns.POISSON_LOW, "\\poisson\\poisson_low.png")
do_Kmeans(pns.POISSON_MODERATE, "\\poisson\\poisson_moderate.png")
do_Kmeans(pns.POISSON_HIGH, "\\poisson\\poisson_high.png")

do_Kmeans(pns.SPECKLE_LOW, "\\speckle\\speckle_low.png")
do_Kmeans(pns.SPECKLE_MODERATE, "\\speckle\\speckle_moderate.png")
do_Kmeans(pns.SPECKLE_HIGH, "\\speckle\\speckle_high.png")

do_Kmeans(pns.UNIFORM_LOW, "\\uniform\\uniform_low.png")
do_Kmeans(pns.UNIFORM_MODERATE, "\\uniform\\uniform_moderate.png")
do_Kmeans(pns.UNIFORM_HIGH, "\\uniform\\uniform_high.png")

do_Kmeans(pns.PEPPER_LOW, "\\pepper\\pepper_low.png")
do_Kmeans(pns.PEPPER_MODERATE, "\\pepper\\pepper_moderate.png")
do_Kmeans(pns.PEPPER_HIGH, "\\pepper\\pepper_high.png")

do_Kmeans(pns.SALT_LOW, "\\salt\\saltr_low.png")
do_Kmeans(pns.SALT_MODERATE, "\\salt\\salt_moderate.png")
do_Kmeans(pns.SALT_HIGH, "\\salt\\saltr_high.png")

do_Kmeans(pns.SP_LOW, "\\salt&pepper\\salt&pepper_low.png")
do_Kmeans(pns.SP_MODERATE, "\\salt&pepper\\salt&pepper_moderate.png")
do_Kmeans(pns.SP_HIGH, "\\salt&pepper\\salt&pepper_high.png")


plt.show()