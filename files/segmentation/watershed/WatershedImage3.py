import cv2
import numpy as np
from matplotlib import pyplot as plt
from path_names import PathNamesSegmentation as pns

OUTPUT_FOLDER_MASK = pns.SEGMENTED + "watershed\\mask\\image3"
OUTPUT_FOLDER_SEGMENTED = pns.SEGMENTED + "watershed\\segmented\\image3"


def do_Watershed(path_name,output_path):
    image_color = cv2.imread(path_name)

    image = image_color[:, :, 0]
    ret1, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((6, 6), np.uint8)
    kernel3 = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1, iterations=10)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations=4)

    sure_background = cv2.dilate(closing, kernel3, iterations=7)

    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)

    ret2, sure_forground = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)

    sure_forground = np.uint8(sure_forground)
    unknown_area = cv2.subtract(sure_background, sure_forground)
    ret3, markers = cv2.connectedComponents(sure_forground)

    markers = markers + 10
    markers[unknown_area == 255] = 0
    plt.imsave("essa.png", markers)
    markers = cv2.watershed(image_color, markers)

    list = np.unique(markers)
    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)




    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, np.ones((7,7)))
    plt.imshow(markers)

    image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
    segmented = cv2.bitwise_and(image_color,mask)
    plt.imsave(OUTPUT_FOLDER_SEGMENTED + output_path, segmented)
    plt.imsave(OUTPUT_FOLDER_MASK + output_path, mask)





# do_Watershed(pns.DEFAULT_IMAGE,"\\default_image3.png")
#
# do_Watershed(pns.GAUSSIAN_LOW, "\\gaussian\\gaussian_low_watershed.png")
# do_Watershed(pns.GAUSSIAN_MODERATE, "\\gaussian\\gaussian_moderate_watershed.png")
# do_Watershed(pns.GAUSSIAN_HIGH, "\\gaussian\\gaussian_high_watershed.png")
#
# do_Watershed(pns.LAPLACIAN_LOW, "\\laplacian\\laplacian_low_watershed.png")
# do_Watershed(pns.LAPLACIAN_MODERATE, "\\laplacian\\laplacian_moderate_watershed.png")
do_Watershed(pns.UNIFORM_HIGH, "test.png")


# do_Watershed(pns.POISSON_LOW, "\\poisson\\poisson_low_watershed.png")
# do_Watershed(pns.POISSON_MODERATE, "\\poisson\\poisson_moderate_watershed.png")
# do_Watershed(pns.POISSON_HIGH, "\\poisson\\poisson_high_watershed.png")
#
# do_Watershed(pns.SPECKLE_LOW, "\\speckle\\speckle_low_watershed.png")
# do_Watershed(pns.SPECKLE_MODERATE, "\\speckle\\speckle_moderate_watershed.png")
# do_Watershed(pns.SPECKLE_HIGH, "\\speckle\\speckle_high_watershed.png")
#
# do_Watershed(pns.UNIFORM_LOW, "\\uniform\\uniform_low_watershed.png")
# do_Watershed(pns.UNIFORM_MODERATE, "\\uniform\\uniform_moderate_watershed.png")


# do_Watershed(pns.PEPPER_LOW, "\\pepper\\pepper_low_watershed.png")
# do_Watershed(pns.PEPPER_MODERATE, "\\pepper\\pepper_moderate_watershed.png")
# do_Watershed(pns.PEPPER_HIGH, "\\pepper\\pepper_high_watershed.png")
# #
# do_Watershed(pns.SALT_LOW, "\\salt\\saltr_low_watershed.png")
# do_Watershed(pns.SALT_MODERATE, "\\salt\\salt_moderate_watershed.png")
# do_Watershed(pns.SALT_HIGH, "\\salt\\saltr_high_watershed.png")
# #
# do_Watershed(pns.SP_LOW, "\\salt&pepper\\salt&pepper_low_watershed.png")
# do_Watershed(pns.SP_MODERATE, "\\salt&pepper\\salt&pepper_moderate_watershed.png")
# do_Watershed(pns.SP_HIGH, "\\salt&pepper\\salt&pepper_high_watershed.png")









