import cv2
from path_names import PathNamesSegmentation as pns
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import morphology

OUTPUT_FOLDER_MASK = pns.SEGMENTED + "otsu\\mask\\image2"
OUTPUT_FOLDER_SEGMENTED = pns.SEGMENTED + "otsu\\segmented\\image2"


def do_Otsu(file_path, output_path):
    image_color = cv2.imread(pns.DEFAULT_IMAGE)
    image_color = cv2.cvtColor(image_color,cv2.COLOR_BGR2RGB)
    image = cv2.imread(file_path, 0)
    ret1, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_ERODE,np.ones((8,8)))
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,np.ones((3,3)))

    thresh = ndi.binary_fill_holes(thresh,np.ones((5,5)))
    thresh = morphology.remove_small_objects(thresh, 950,connectivity=8)
    thresh = thresh.astype(np.uint8)

    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    mask[thresh == 0] = [255, 255, 255]
    mask = cv2.bitwise_not(mask)
    edged = cv2.Canny(mask, 100, 210)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, (255, 0, 0), 1)

    segmented = cv2.bitwise_and(image_color, mask)

    plt.imsave(OUTPUT_FOLDER_SEGMENTED + output_path, segmented)
    plt.imsave(OUTPUT_FOLDER_MASK + output_path, mask)


do_Otsu(pns.DEFAULT_IMAGE,"\\default_image2.png")
#
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

do_Otsu(pns.SALT_LOW, "\\salt\\saltr_low.png")
do_Otsu(pns.SALT_MODERATE, "\\salt\\salt_moderate.png")
do_Otsu(pns.SALT_HIGH, "\\salt\\saltr_high.png")

do_Otsu(pns.SP_LOW, "\\salt&pepper\\salt&pepper_low.png")
do_Otsu(pns.SP_MODERATE, "\\salt&pepper\\salt&pepper_moderate.png")
do_Otsu(pns.SP_HIGH, "\\salt&pepper\\salt&pepper_high.png")







# image = cv2.imread(pns.GAUSSIAN_HIGH,0)
# image_color = cv2.imread(pns.GAUSSIAN_HIGH)
# image_color = cv2.cvtColor(image_color,cv2.COLOR_BGR2RGB)
# new_image = np.zeros((image.shape[0],image.shape[1], 3), dtype=np.uint8)
# ret1, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# thresh = ndi.binary_fill_holes(thresh)
# thresh = thresh.astype(np.uint8)
# new_image[thresh==1] = [255,255,255]
#
# result = cv2.bitwise_and(image_color,new_image)
# plt.imshow(result)
# plt.show()


# POMARANCZE
# image = cv2.imread(pns.ORIGINAL2,0)
# ret1, segmented = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# segmented = cv2.morphologyEx(segmented,cv2.MORPH_OPEN,np.ones((7,7)))

# NOWOTWÓR
# image = cv2.imread(pns.ORIGINAL1)
# image = image[:,:,0]
# ret1, segmented = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# segmented = cv2.morphologyEx(segmented,cv2.MORPH_OPEN,np.ones((4,4)))

# def imshow_components(labels):
#     # Map component labels to hue val
#     label_hue = np.uint8(179*labels/np.max(labels))
#     blank_ch = 255*np.ones_like(label_hue)
#     labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#
#     # cvt to BGR for display
#     labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#
#     # set bg label to black
#     labeled_img[label_hue==0] = 0
#
#     cv2.imshow('labeled.png', labeled_img)
#     cv2.waitKey()
