from math import sqrt, exp

import cv2

import numpy as np

import matplotlib.pyplot as plt
from path_names import PathNamesNoise as pn

OUTPUT_FOLDER_NAME = pn.DENOISED_IMAGES + "denoised images" + pn.IMAGE_NUMBER


def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def ideal_filter_lp(D0, img_shape):
    base = np.zeros(img_shape[:2])
    rows, cols = img_shape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < D0:
                base[y, x] = 1
    return base


def butterworth_lp(D0, img_shape, n):
    base = np.zeros(img_shape[:2])
    rows, cols = img_shape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))
    return base


def gaussian_lp(D0, img_shape):
    base = np.zeros(img_shape[:2])
    rows, cols = img_shape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
    return base


def do_gaussian_lp(file_path, output_name):
    img = cv2.imread(file_path)
    b, g, r = cv2.split(img)
    D0 = 90
    original_b = np.fft.fft2(b)
    center_b = np.fft.fftshift(original_b)
    low_pass_center_b = center_b * gaussian_lp(D0, b.shape)
    low_pass_b = np.fft.ifftshift(low_pass_center_b)
    inverse_low_pass_b = np.fft.ifft2(low_pass_b)

    original_g = np.fft.fft2(g)
    center_g = np.fft.fftshift(original_g)
    low_pass_center_g = center_g * gaussian_lp(D0, g.shape)
    low_pass_g = np.fft.ifftshift(low_pass_center_g)
    inverse_low_pass_g = np.fft.ifft2(low_pass_g)

    original_r = np.fft.fft2(r)
    center_r = np.fft.fftshift(original_r)
    low_pass_center_r = center_r * gaussian_lp(D0, r.shape)
    low_pass_r = np.fft.ifftshift(low_pass_center_r)
    inverse_low_pass_r = np.fft.ifft2(low_pass_r)

    absolute_b = np.abs(inverse_low_pass_b)
    absolute_g = np.abs(inverse_low_pass_g)
    absolute_r = np.abs(inverse_low_pass_r)

    output = cv2.merge((absolute_b, absolute_g, absolute_r))

    output_folder_path = OUTPUT_FOLDER_NAME + "\\fourier_gaussianLP_filter\\"
    plt.imsave(output_folder_path + output_name, cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2RGB))


def do_ideal_filter_lp(file_path, output_name):
    img = cv2.imread(file_path)
    b, g, r = cv2.split(img)
    D0 = 90
    original_b = np.fft.fft2(b)
    center_b = np.fft.fftshift(original_b)
    low_pass_center_b = center_b * ideal_filter_lp(D0, b.shape)
    low_pass_b = np.fft.ifftshift(low_pass_center_b)
    inverse_low_pass_b = np.fft.ifft2(low_pass_b)

    original_g = np.fft.fft2(g)
    center_g = np.fft.fftshift(original_g)
    low_pass_center_g = center_g * ideal_filter_lp(D0, g.shape)
    low_pass_g = np.fft.ifftshift(low_pass_center_g)
    inverse_low_pass_g = np.fft.ifft2(low_pass_g)

    original_r = np.fft.fft2(r)
    center_r = np.fft.fftshift(original_r)
    low_pass_center_r = center_r * ideal_filter_lp(D0, r.shape)
    low_pass_r = np.fft.ifftshift(low_pass_center_r)
    inverse_low_pass_r = np.fft.ifft2(low_pass_r)

    absolute_b = np.abs(inverse_low_pass_b)
    absolute_g = np.abs(inverse_low_pass_g)
    absolute_r = np.abs(inverse_low_pass_r)

    output = cv2.merge((absolute_b, absolute_g, absolute_r))
    output_folder_path = OUTPUT_FOLDER_NAME + "\\fourier_idealLP_filter\\"
    plt.imsave(output_folder_path + output_name, cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2RGB))


def do_butterworth_lp(file_path, output_name, n):
    img = cv2.imread(file_path)
    b, g, r = cv2.split(img)
    D0 = 120
    original_b = np.fft.fft2(b)
    center_b = np.fft.fftshift(original_b)
    low_pass_center_b = center_b * butterworth_lp(D0, b.shape, n)
    low_pass_b = np.fft.ifftshift(low_pass_center_b)
    inverse_low_pass_b = np.fft.ifft2(low_pass_b)

    original_g = np.fft.fft2(g)
    center_g = np.fft.fftshift(original_g)
    low_pass_center_g = center_g * butterworth_lp(D0, b.shape, n)
    low_pass_g = np.fft.ifftshift(low_pass_center_g)
    inverse_LowPass_g = np.fft.ifft2(low_pass_g)

    original_r = np.fft.fft2(r)
    center_r = np.fft.fftshift(original_r)
    low_pass_center_r = center_r * butterworth_lp(D0, b.shape, n)
    low_pass_r = np.fft.ifftshift(low_pass_center_r)
    inverse_LowPass_r = np.fft.ifft2(low_pass_r)

    absolute_b = np.abs(inverse_low_pass_b)
    absolute_g = np.abs(inverse_LowPass_g)
    absolute_r = np.abs(inverse_LowPass_r)

    output = cv2.merge((absolute_b, absolute_g, absolute_r))
    output_folder_path = OUTPUT_FOLDER_NAME + "\\fourier_butterworthLP_filter\\"
    plt.imsave(output_folder_path + output_name, cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2RGB))


#
do_gaussian_lp(pn.GAUSSIAN_LOW, "gaussian\\gaussian_low_gaussianLP.png")
do_gaussian_lp(pn.GAUSSIAN_MODERATE, "gaussian\\gaussian_moderate_gaussianLP.png")
do_gaussian_lp(pn.GAUSSIAN_HIGH, "gaussian\\gaussian_high_gaussianLP.png")

do_gaussian_lp(pn.LAPLACIAN_LOW, "laplacian\\laplacian_low_gaussianLP.png")
do_gaussian_lp(pn.LAPLACIAN_MODERATE, "laplacian\\laplacian_moderate_gaussianLP.png")
do_gaussian_lp(pn.LAPLACIAN_HIGH, "laplacian\\laplacian_high_gaussianLP.png")

do_gaussian_lp(pn.POISSON_LOW, "poisson\\poisson_low_gaussianLP.png")
do_gaussian_lp(pn.POISSON_MODERATE, "poisson\\poisson_moderate_gaussianLP.png")
do_gaussian_lp(pn.POISSON_HIGH, "poisson\\poisson_high_gaussianLP.png")

do_gaussian_lp(pn.GAUSSIAN_LOW, "gaussian\\gaussian_low_gaussianLP.png")
do_gaussian_lp(pn.GAUSSIAN_MODERATE, "gaussian\\gaussian_moderate_gaussianLP.png")
do_gaussian_lp(pn.GAUSSIAN_HIGH, "gaussian\\gaussian_high_gaussianLP.png")
#
do_gaussian_lp(pn.SPECKLE_LOW, "speckle\\speckle_low_gaussianLP.png")
do_gaussian_lp(pn.SPECKLE_MODERATE, "speckle\\speckle_moderate_gaussianLP.png")
do_gaussian_lp(pn.SPECKLE_HIGH, "speckle\\speckle_high_gaussianLP.png")
#
do_gaussian_lp(pn.UNIFORM_LOW, "uniform\\uniform_low_gaussianLP.png")
do_gaussian_lp(pn.UNIFORM_MODERATE, "uniform\\uniform_moderate_gaussianLP.png")
do_gaussian_lp(pn.UNIFORM_HIGH, "uniform\\uniform_high_gaussianLP.png")

do_gaussian_lp(pn.PEPPER_LOW, "pepper\\pepper_low_gaussianLP.png")
do_gaussian_lp(pn.PEPPER_MODERATE, "pepper\\pepper_moderate_gaussianLP.png")
do_gaussian_lp(pn.PEPPER_HIGH, "pepper\\pepper_high_gaussianLP.png")

do_gaussian_lp(pn.SALT_LOW, "salt\\salt_low_gaussianLP.png")
do_gaussian_lp(pn.SALT_MODERATE, "salt\\salt_moderate_gaussianLP.png")
do_gaussian_lp(pn.SALT_HIGH, "salt\\salt_high_gaussianLP.png")

do_gaussian_lp(pn.SP_LOW, "salt&pepper\\salt&pepper_low_gaussianLP.png")
do_gaussian_lp(pn.SP_MODERATE, "salt&pepper\\salt&pepper_moderate_gaussianLP.png")
do_gaussian_lp(pn.SP_HIGH, "salt&pepper\\salt&pepper_high_gaussianLP.png")

###
do_ideal_filter_lp(pn.GAUSSIAN_LOW, "gaussian\\gaussian_low_idealLP.png")
do_ideal_filter_lp(pn.GAUSSIAN_MODERATE, "gaussian\\gaussian_moderate_idealLP.png")
do_ideal_filter_lp(pn.GAUSSIAN_HIGH, "gaussian\\gaussian_high_idealLP.png")

do_ideal_filter_lp(pn.LAPLACIAN_LOW, "laplacian\\laplacian_low_idealLP.png")
do_ideal_filter_lp(pn.LAPLACIAN_MODERATE, "laplacian\\laplacian_moderate_idealLP.png")
do_ideal_filter_lp(pn.LAPLACIAN_HIGH, "laplacian\\laplacian_high_idealLP.png")

do_ideal_filter_lp(pn.POISSON_LOW, "poisson\\poisson_low_idealLP.png")
do_ideal_filter_lp(pn.POISSON_MODERATE, "poisson\\poisson_moderate_idealLP.png")
do_ideal_filter_lp(pn.POISSON_HIGH, "poisson\\poisson_high_idealLP.png")
#
do_ideal_filter_lp(pn.SPECKLE_LOW, "speckle\\speckle_low_idealLP.png")
do_ideal_filter_lp(pn.SPECKLE_MODERATE, "speckle\\speckle_moderate_idealLP.png")
do_ideal_filter_lp(pn.SPECKLE_HIGH, "speckle\\speckle_high_idealLP.png")
#
do_ideal_filter_lp(pn.UNIFORM_LOW, "uniform\\uniform_low_idealLP.png")
do_ideal_filter_lp(pn.UNIFORM_MODERATE, "uniform\\uniform_moderate_idealLP.png")
do_ideal_filter_lp(pn.UNIFORM_HIGH, "uniform\\uniform_high_idealLP.png")

do_ideal_filter_lp(pn.PEPPER_LOW, "pepper\\pepper_low_idealLP.png")
do_ideal_filter_lp(pn.PEPPER_MODERATE, "pepper\\pepper_moderate_idealLP.png")
do_ideal_filter_lp(pn.PEPPER_HIGH, "pepper\\pepper_high_idealLP.png")

do_ideal_filter_lp(pn.SALT_LOW, "salt\\salt_low_idealLP.png")
do_ideal_filter_lp(pn.SALT_MODERATE, "salt\\salt_moderate_idealLP.png")
do_ideal_filter_lp(pn.SALT_HIGH, "salt\\salt_high_idealLP.png")

do_ideal_filter_lp(pn.SP_LOW, "salt&pepper\\salt&pepper_low_idealLP.png")
do_ideal_filter_lp(pn.SP_MODERATE, "salt&pepper\\salt&pepper_moderate_idealLP.png")
do_ideal_filter_lp(pn.SP_HIGH, "salt&pepper\\salt&pepper_high_idealLP.png")

##
do_butterworth_lp(pn.GAUSSIAN_LOW, "gaussian\\gaussian_low_butterworthLP.png", 1)
do_butterworth_lp(pn.GAUSSIAN_MODERATE, "gaussian\\gaussian_moderate_butterworthLP.png", 1)
do_butterworth_lp(pn.GAUSSIAN_HIGH, "gaussian\\gaussian_high_butterworthLP.png", 1)
#
do_butterworth_lp(pn.LAPLACIAN_LOW, "laplacian\\laplacian_low_butterworthLP.png", 3)
do_butterworth_lp(pn.LAPLACIAN_MODERATE, "laplacian\\laplacian_moderate_butterworthLP.png", 3)
do_butterworth_lp(pn.LAPLACIAN_HIGH, "laplacian\\laplacian_high_butterworthLP.png", 3)

do_butterworth_lp(pn.POISSON_LOW, "poisson\\poisson_low_butterworthLP.png", 3)
do_butterworth_lp(pn.POISSON_MODERATE, "poisson\\poisson_moderate_butterworthLP.png", 3)
do_butterworth_lp(pn.POISSON_HIGH, "poisson\\poisson_high_butterworthLP.png", 3)

do_butterworth_lp(pn.SPECKLE_LOW, "speckle\\speckle_low_butterworthLP.png", 3)
do_butterworth_lp(pn.SPECKLE_MODERATE, "speckle\\speckle_moderate_butterworthLP.png", 3)
do_butterworth_lp(pn.SPECKLE_HIGH, "speckle\\speckle_high_butterworthLP.png", 3)
#
do_butterworth_lp(pn.UNIFORM_LOW, "uniform\\uniform_low_butterworthLP.png", 3)
do_butterworth_lp(pn.UNIFORM_MODERATE, "uniform\\uniform_moderate_butterworthLP.png", 3)
do_butterworth_lp(pn.UNIFORM_HIGH, "uniform\\uniform_high_butterworthLP.png", 3)

do_butterworth_lp(pn.PEPPER_LOW, "pepper\\pepper_low_butterworthLP.png", 3)
do_butterworth_lp(pn.PEPPER_MODERATE, "pepper\\pepper_moderate_butterworthLP.png", 3)
do_butterworth_lp(pn.PEPPER_HIGH, "pepper\\pepper_high_butterworthLP.png", 3)

do_butterworth_lp(pn.SALT_LOW, "salt\\salt_low_butterworthLP.png", 3)
do_butterworth_lp(pn.SALT_MODERATE, "salt\\salt_moderate_butterworthLP.png", 3)
do_butterworth_lp(pn.SALT_HIGH, "salt\\salt_high_butterworthLP.png", 3)

do_butterworth_lp(pn.SP_LOW, "salt&pepper\\salt&pepper_low_butterworthLP.png", 3)
do_butterworth_lp(pn.SP_MODERATE, "salt&pepper\\salt&pepper_moderate_butterworthLP.png", 3)
do_butterworth_lp(pn.SP_HIGH, "salt&pepper\\salt&pepper_high_butterworthLP.png", 3)

# plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)
# img = cv2.imread("E:\\PycharmProjects\\segmentation\\noisy images\\gaussian\\gaussianHighNoise.png", 1)
# b, g, r = cv2.split(img)
#
# ###### B
# original_b = np.fft.fft2(b)
# center_b = np.fft.fftshift(original_b)
#
# LowPassCenter_b = center_b * gaussianLP(50, b.shape)
# LowPass_b = np.fft.ifftshift(LowPassCenter_b)
# inverse_LowPass_b = np.fft.ifft2(LowPass_b)
#
# ###### G
# original_g = np.fft.fft2(g)
# center_g = np.fft.fftshift(original_g)
#
# LowPassCenter_g = center_g * gaussianLP(50, g.shape)
# LowPass_g = np.fft.ifftshift(LowPassCenter_g)
# inverse_LowPass_g = np.fft.ifft2(LowPass_g)
#
# ###### R
# original_r = np.fft.fft2(r)
# center_r = np.fft.fftshift(original_r)
#
# LowPassCenter_r = center_r * gaussianLP(50, r.shape)
# LowPass_r = np.fft.ifftshift(LowPassCenter_r)
# inverse_LowPass_r = np.fft.ifft2(LowPass_r)
#
# absolute_b = np.abs(inverse_LowPass_b)
# absolute_g = np.abs(inverse_LowPass_g)
# absolute_r = np.abs(inverse_LowPass_r)
#
# output_low = cv2.merge((absolute_b, absolute_g, absolute_r))
# # plt.subplot(132), plt.imshow( cv2.cvtColor((essa).astype(np.uint8), cv2.COLOR_BGR2RGB)), plt.title("gauss_low")
# plt.imsave("gaussian_high_gaussianLP.png", cv2.cvtColor(output_low.astype(np.uint8), cv2.COLOR_BGR2RGB))


# plt.subplot(133), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Gaussian Low Pass")
# LowPassCenter = center * idealFilterLP(50,img.shape)
# LowPass = np.fft.ifftshift(LowPassCenter)
# inverse_LowPass = np.fft.ifft2(LowPass)
# plt.subplot(131), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Ideal Low Pass")
#
# LowPassCenter = center * butterworthLP(50,img.shape,10)
# LowPass = np.fft.ifftshift(LowPassCenter)
# inverse_LowPass = np.fft.ifft2(LowPass)
# plt.subplot(132), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Butterworth Low Pass (n=10)")
#
# LowPassCenter = center_b * gaussianLP(50, img.shape)
# LowPass = np.fft.ifftshift(LowPassCenter)
# inverse_LowPass_b = np.fft.ifft2(LowPass)
# plt.subplot(133), plt.imshow(np.abs(inverse_LowPass_b), "gray"), plt.title("Gaussian Low Pass")

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.show()
