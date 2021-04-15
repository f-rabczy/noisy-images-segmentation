import cv2
from path_names import PathNamesSegmentation as pns
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
import matplotlib.pyplot as plt

OUTPUT_FOLDER_MASK = pns.SEGMENTED + "otsu\\mask\\image1"
OUTPUT_FOLDER_SEGMENTED = pns.SEGMENTED + "otsu\\segmented\\image1"


def do_Otsu(file_path, output_path):
    image = cv2.imread(file_path, 0)
    ret1, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = ndi.binary_fill_holes(thresh, np.ones((3,3)))
    thresh = morphology.remove_small_objects(thresh, 3)
    thresh = thresh.astype(np.uint8)

    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    mask[thresh == 0] = [255, 255, 255]
    mask = cv2.bitwise_not(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((2, 2)))
    segmented = cv2.bitwise_and(image, mask)
    plt.imsave(OUTPUT_FOLDER_SEGMENTED + output_path, segmented)
    plt.imsave(OUTPUT_FOLDER_MASK + output_path, mask)


do_Otsu(pns.DEFAULT_IMAGE,"\\default_image1.png")

do_Otsu(pns.GAUSSIAN_LOW, "\\gaussian\\gaussian_low.png")
do_Otsu(pns.GAUSSIAN_MODERATE, "\\gaussian\\gaussian_moderate.png")
do_Otsu(pns.GAUSSIAN_HIGH, "\\gaussian\\gaussian_high.png")

do_Otsu(pns.LAPLACIAN_LOW, "\\laplacian\\laplacian_low.png")
do_Otsu(pns.LAPLACIAN_MODERATE, "\\laplacian\\laplacian_moderate.png")
do_Otsu(pns.LAPLACIAN_HIGH, "\\laplacian\\laplacian_high.png")

do_Otsu(pns.POISSON_LOW, "\\poisson\\poisson_low.png")
do_Otsu(pns.POISSON_MODERATE, "\\poisson\\poisson_moderate.png")
do_Otsu(pns.POISSON_HIGH, "\\poisson\\poisson_high.png")

do_Otsu(pns.SPECKLE_LOW, "\\speckle\\speckle_low.png")
do_Otsu(pns.SPECKLE_MODERATE, "\\speckle\\speckle_moderate.png")
do_Otsu(pns.SPECKLE_HIGH, "\\speckle\\speckle_high.png")

do_Otsu(pns.UNIFORM_LOW, "\\uniform\\uniform_low.png")
do_Otsu(pns.UNIFORM_MODERATE, "\\uniform\\uniform_moderate.png")
do_Otsu(pns.UNIFORM_HIGH, "\\uniform\\uniform_high.png")

do_Otsu(pns.PEPPER_LOW, "\\pepper\\pepper_low.png")
do_Otsu(pns.PEPPER_MODERATE, "\\pepper\\pepper_moderate.png")
do_Otsu(pns.PEPPER_HIGH, "\\pepper\\pepper_high.png")

do_Otsu(pns.SALT_LOW, "\\salt\\salt_low.png")
do_Otsu(pns.SALT_MODERATE, "\\salt\\salt_moderate.png")
do_Otsu(pns.SALT_HIGH, "\\salt\\salt_high.png")

do_Otsu(pns.SP_LOW, "\\salt&pepper\\salt&pepper_low.png")
do_Otsu(pns.SP_MODERATE, "\\salt&pepper\\salt&pepper_moderate.png")
do_Otsu(pns.SP_HIGH, "\\salt&pepper\\salt&pepper_high.png")
