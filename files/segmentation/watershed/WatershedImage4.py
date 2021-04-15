import cv2
import numpy as np
from matplotlib import pyplot as plt
from path_names import PathNamesSegmentation as pns


OUTPUT_FOLDER_MASK = pns.SEGMENTED + "watershed\\mask\\image4"
OUTPUT_FOLDER_SEGMENTED = pns.SEGMENTED + "watershed\\segmented\\image4"

def do_Watershed(path_name,output_path):
    image_color = cv2.imread(path_name)
    gray = image_color[:, :, 0]

    ret1, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel1 = np.ones((1, 1), np.uint8)
    kernel2 = np.ones((2, 2), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations=1)

    sure_background = cv2.dilate(closing, kernel2, iterations=3)
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)

    ret, sure_forground = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_forground = np.uint8(sure_forground)

    unknown_area = cv2.subtract(sure_background, sure_forground)
    ret, markers = cv2.connectedComponents(sure_forground)
    markers = markers + 1
    markers[unknown_area == 255] = 0

    markers = cv2.watershed(image_color, markers)
    image_color[markers == -1] = [255, 0, 0]
    list = np.unique(markers)
    mask = np.zeros((image_color.shape[0], image_color.shape[1], 3), dtype=np.uint8)

    mask[markers == 1] = [255, 255, 255]
    mask[markers == list[0]] = [255, 255, 255]
    mask = cv2.bitwise_not(mask)
    plt.imshow(mask)
    image_color = cv2.cvtColor(image_color , cv2.COLOR_BGR2RGB)
    segmented = cv2.bitwise_and(image_color, mask)

    plt.imsave(OUTPUT_FOLDER_SEGMENTED + output_path, segmented)
    plt.imsave(OUTPUT_FOLDER_MASK + output_path, mask)


do_Watershed(pns.DEFAULT_IMAGE,"\\default_image4.png")

do_Watershed(pns.GAUSSIAN_LOW, "\\gaussian\\gaussian_low_watershed.png")
do_Watershed(pns.GAUSSIAN_MODERATE, "\\gaussian\\gaussian_moderate_watershed.png")
do_Watershed(pns.GAUSSIAN_HIGH, "\\gaussian\\gaussian_high_watershed.png")

do_Watershed(pns.LAPLACIAN_LOW, "\\laplacian\\laplacian_low_watershed.png")
do_Watershed(pns.LAPLACIAN_MODERATE, "\\laplacian\\laplacian_moderate_watershed.png")
do_Watershed(pns.LAPLACIAN_HIGH, "\\laplacian\\laplacian_high_watershed.png")

do_Watershed(pns.POISSON_LOW, "\\poisson\\poisson_low_watershed.png")
do_Watershed(pns.POISSON_MODERATE, "\\poisson\\poisson_moderate_watershed.png")
do_Watershed(pns.POISSON_HIGH, "\\poisson\\poisson_high_watershed.png")

do_Watershed(pns.SPECKLE_LOW, "\\speckle\\speckle_low_watershed.png")
do_Watershed(pns.SPECKLE_MODERATE, "\\speckle\\speckle_moderate_watershed.png")
do_Watershed(pns.SPECKLE_HIGH, "\\speckle\\speckle_high_watershed.png")

do_Watershed(pns.UNIFORM_LOW, "\\uniform\\uniform_low_watershed.png")
do_Watershed(pns.UNIFORM_MODERATE, "\\uniform\\uniform_moderate_watershed.png")
do_Watershed(pns.UNIFORM_HIGH, "\\uniform\\uniform_high_watershed.png")

do_Watershed(pns.PEPPER_LOW, "\\pepper\\pepper_low_watershed.png")
do_Watershed(pns.PEPPER_MODERATE, "\\pepper\\pepper_moderate_watershed.png")
do_Watershed(pns.PEPPER_HIGH, "\\pepper\\pepper_high_watershed.png")

do_Watershed(pns.SALT_LOW, "\\salt\\saltr_low_watershed.png")
do_Watershed(pns.SALT_MODERATE, "\\salt\\salt_moderate_watershed.png")
do_Watershed(pns.SALT_HIGH, "\\salt\\saltr_high_watershed.png")

do_Watershed(pns.SP_LOW, "\\salt&pepper\\salt&pepper_low_watershed.png")
do_Watershed(pns.SP_MODERATE, "\\salt&pepper\\salt&pepper_moderate_watershed.png")
do_Watershed(pns.SP_HIGH, "\\salt&pepper\\salt&pepper_high_watershed.png")



