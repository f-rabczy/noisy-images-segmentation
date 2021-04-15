import cv2
import skimage
from skimage import io
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle, denoise_wavelet
import numpy as np
from skimage import img_as_float, io, img_as_ubyte
import matplotlib.pyplot as plt
from path_names import PathNamesNoise as pn
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

OUTPUT_FOLDER_NAME = pn.DENOISED_IMAGES + "denoised images" + pn.IMAGE_NUMBER
default_image = skimage.io.imread(pn.DEFAULT_IMAGE)


def nl_means_filtering(file_path, output_name, h_value):
    img = img_as_float(io.imread(file_path))
    sigma_est = np.mean(estimate_sigma(img, multichannel=True))
    patch_kw = dict(patch_size=15,
                    patch_distance = 10,
                    multichannel=True)
    img_denoised = denoise_nl_means(img, h=h_value * sigma_est, sigma=sigma_est, fast_mode=True,
                                    **patch_kw)
    img_denoised_ubyte = img_as_ubyte(img_denoised)
    if len(img_denoised_ubyte.shape) > 2 and img_denoised_ubyte.shape[2] == 4:
        img_denoised_ubyte = cv2.cvtColor(img_denoised_ubyte, cv2.COLOR_BGRA2BGR)

    output_folder_path = OUTPUT_FOLDER_NAME + "\\nl_means_filter\\"
    plt.imsave(output_folder_path + output_name, img_denoised_ubyte)


def total_variation_filtering(file_path, output_name, weight):
    img = img_as_float(io.imread(file_path))
    img_denoised = denoise_tv_chambolle(img, weight, multichannel=True)
    img_denoised_ubyte = img_as_ubyte(img_denoised)

    if len(img_denoised_ubyte.shape) > 2 and img_denoised_ubyte.shape[2] == 4:
        img_denoised_ubyte = cv2.cvtColor(img_denoised_ubyte, cv2.COLOR_BGRA2BGR)

    output_folder_path = OUTPUT_FOLDER_NAME + "\\tv_filter\\"
    plt.imsave(output_folder_path + output_name, img_denoised_ubyte)


def bilateral_filtering(file_path, output_name, d, sigmaC, sigmaS):
    img = cv2.imread(file_path)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_denoised = cv2.bilateralFilter(img, d, sigmaC, sigmaS, borderType=cv2.BORDER_CONSTANT)
    if len(img_denoised.shape) > 2 and img_denoised.shape[2] == 4:
        img_denoised = cv2.cvtColor(img_denoised, cv2.COLOR_BGRA2BGR)

    output_folder_path = OUTPUT_FOLDER_NAME + "\\bilateral_filter\\"
    plt.imsave(output_folder_path + output_name, img_denoised)


def gaussian_filtering(file_path, output_name, kernel):
    img = cv2.imread(file_path)
    img_denoised = cv2.GaussianBlur(img, (kernel, kernel), 0, cv2.BORDER_CONSTANT)

    output_folder_path = OUTPUT_FOLDER_NAME + "\\gaussian_filter\\"
    img_denoised_ubyte = img_as_ubyte(img_denoised)

    if len(img_denoised_ubyte.shape) > 2 and img_denoised_ubyte.shape[2] == 4:
        img_denoised_ubyte = cv2.cvtColor(img_denoised_ubyte, cv2.COLOR_BGRA2BGR)

    plt.imsave(output_folder_path + output_name, cv2.cvtColor(img_denoised_ubyte, cv2.COLOR_BGR2RGB))


def median_filtering(file_path, output_name, kernel):
    img = cv2.imread(file_path)
    img_denoised = cv2.medianBlur(img, kernel)
    img_denoised_ubyte = img_as_ubyte(img_denoised)

    if len(img_denoised.shape) > 2 and img_denoised.shape[2] == 4:
        img_denoised_ubyte = cv2.cvtColor(img_denoised, cv2.COLOR_BGRA2BGR)

    output_folder_path = OUTPUT_FOLDER_NAME + "\\median_filter\\"
    plt.imsave(output_folder_path + output_name, cv2.cvtColor(img_denoised_ubyte, cv2.COLOR_BGR2RGB))


def wavelet_filtering(file_path, output_name, method, mode, wavelet_levels, wavelet):
    img = skimage.io.imread(file_path)
    if len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img_denoised = denoise_wavelet(img, method=method, wavelet_levels=wavelet_levels, mode=mode, wavelet=wavelet,
                                   multichannel=True, convert2ycbcr=True, rescale_sigma=True)
    img_denoised_ubyte = skimage.img_as_ubyte(img_denoised)
    output_folder_path = OUTPUT_FOLDER_NAME + "\\wavelet_filter\\"
    skimage.io.imsave(output_folder_path + output_name, img_denoised_ubyte)


# NON LOCAL MEANS
H_VALUE = 0.5
nl_means_filtering(pn.GAUSSIAN_LOW, "gaussian\\gaussian_low_nl_means.png", H_VALUE)
nl_means_filtering(pn.GAUSSIAN_MODERATE, "gaussian\\gaussian_moderate_nl_means.png", H_VALUE)
nl_means_filtering(pn.GAUSSIAN_HIGH, "gaussian\\gaussian_high_nl_means.png", H_VALUE)

nl_means_filtering(pn.LAPLACIAN_LOW, "laplacian\\laplacian_low_nl_means.png", H_VALUE)
nl_means_filtering(pn.LAPLACIAN_MODERATE, "laplacian\\laplacian_moderate_nl_means.png", H_VALUE)
nl_means_filtering(pn.LAPLACIAN_HIGH, "laplacian\\laplacian_high_nl_means.png", H_VALUE)

nl_means_filtering(pn.POISSON_LOW, "poisson\\poisson_low_nl_means.png", H_VALUE)
nl_means_filtering(pn.POISSON_MODERATE, "poisson\\poisson_moderate_nl_means.png", H_VALUE)
nl_means_filtering(pn.POISSON_HIGH, "poisson\\poisson_high_nl_means.png", H_VALUE)

nl_means_filtering(pn.SPECKLE_LOW, "speckle\\speckle_low_nl_means.png", H_VALUE)
nl_means_filtering(pn.SPECKLE_MODERATE, "speckle\\speckle_moderate_nl_means.png", H_VALUE)
nl_means_filtering(pn.SPECKLE_HIGH, "speckle\\speckle_high_nl_means.png", H_VALUE)

nl_means_filtering(pn.UNIFORM_LOW, "uniform\\uniform_low_nl_means.png", H_VALUE)
nl_means_filtering(pn.UNIFORM_MODERATE, "uniform\\uniform_moderate_nl_means.png", H_VALUE)
nl_means_filtering(pn.UNIFORM_HIGH, "uniform\\uniform_high_nl_means.png", H_VALUE)

nl_means_filtering(pn.PEPPER_LOW, "pepper\\pepper_low_nl_means.png", H_VALUE)
nl_means_filtering(pn.PEPPER_MODERATE, "pepper\\pepper_moderate_nl_means.png", H_VALUE)
nl_means_filtering(pn.PEPPER_HIGH, "pepper\\pepper_high_nl_means.png", H_VALUE)

nl_means_filtering(pn.SALT_LOW, "salt\\salt_low_nl_means.png", H_VALUE)
nl_means_filtering(pn.SALT_MODERATE, "salt\\salt_moderate_nl_means.png", H_VALUE)
nl_means_filtering(pn.SALT_HIGH, "salt\\salt_high_nl_means.png", H_VALUE)

nl_means_filtering(pn.SP_LOW, "salt&pepper\\salt&pepper_low_nl_means.png", H_VALUE)
nl_means_filtering(pn.SP_MODERATE, "salt&pepper\\salt&pepper_moderate_nl_means.png", H_VALUE)
nl_means_filtering(pn.SP_HIGH, "salt&pepper\\salt&pepper_high_nl_means.png", H_VALUE)

#
D_VALUE = 30
S1_VALUE = 75
S2_VALUE = 75
bilateral_filtering(pn.GAUSSIAN_LOW, "gaussian\\gaussian_low_bilateral.png", D_VALUE, S1_VALUE, S2_VALUE)
bilateral_filtering(pn.GAUSSIAN_MODERATE, "gaussian\\gaussian_moderate_bilateral.png", D_VALUE, S1_VALUE, S2_VALUE)
bilateral_filtering(pn.GAUSSIAN_HIGH, "gaussian\\gaussian_high_bilateral.png", D_VALUE, S1_VALUE, S2_VALUE)

bilateral_filtering(pn.LAPLACIAN_LOW, "laplacian\\laplacian_low_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.LAPLACIAN_MODERATE, "laplacian\\laplacian_moderate_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.LAPLACIAN_HIGH, "laplacian\\laplacian_high_bilateral.png", 15, 75, 75)

bilateral_filtering(pn.POISSON_LOW, "poisson\\poisson_low_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.POISSON_MODERATE, "poisson\\poisson_moderate_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.POISSON_HIGH, "poisson\\poisson_high_bilateral.png", 15, 75, 75)

bilateral_filtering(pn.SPECKLE_LOW, "speckle\\speckle_low_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.SPECKLE_MODERATE, "speckle\\speckle_moderate_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.SPECKLE_HIGH, "speckle\\speckle_high_bilateral.png", 15, 75, 75)

bilateral_filtering(pn.UNIFORM_LOW, "uniform\\uniform_low_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.UNIFORM_MODERATE, "uniform\\uniform_moderate_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.UNIFORM_HIGH, "uniform\\uniform_high_bilateral.png", 15, 75, 75)

bilateral_filtering(pn.PEPPER_LOW, "pepper\\pepper_low_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.PEPPER_MODERATE, "pepper\\pepper_moderate_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.PEPPER_HIGH, "pepper\\pepper_high_bilateral.png", 15, 75, 75)

bilateral_filtering(pn.SALT_LOW, "salt\\salt_low_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.SALT_MODERATE, "salt\\salt_moderate_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.SALT_HIGH, "salt\\salt_high_bilateral.png", 15, 75, 75)

bilateral_filtering(pn.SP_LOW, "salt&pepper\\salt&pepper_low_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.SP_MODERATE, "salt&pepper\\salt&pepper_moderate_bilateral.png", 15, 75, 75)
bilateral_filtering(pn.SP_HIGH, "salt&pepper\\salt&pepper_high_bilateral.png", 15, 75, 75)
print("BILATERAL")

# GAUSSIAN BLUR
gaussian_filtering(pn.GAUSSIAN_LOW, "gaussian\\gaussian_low_gaussian.png", 3)
gaussian_filtering(pn.GAUSSIAN_MODERATE, "gaussian\\gaussian_moderate_gaussian.png", 3)
gaussian_filtering(pn.GAUSSIAN_HIGH, "gaussian\\gaussian_high_gaussian.png", 3)

gaussian_filtering(pn.LAPLACIAN_LOW, "laplacian\\laplacian_low_gaussian.png", 3)
gaussian_filtering(pn.LAPLACIAN_MODERATE, "laplacian\\laplacian_moderate_gaussian.png", 3)
gaussian_filtering(pn.LAPLACIAN_HIGH, "laplacian\\laplacian_high_gaussian.png", 3)

gaussian_filtering(pn.POISSON_LOW, "poisson\\poisson_low_gaussian.png", 3)
gaussian_filtering(pn.POISSON_MODERATE, "poisson\\poisson_moderate_gaussian.png", 3)
gaussian_filtering(pn.POISSON_HIGH, "poisson\\poisson_high_gaussian.png", 3)

gaussian_filtering(pn.SPECKLE_LOW, "speckle\\speckle_low_gaussian.png", 3)
gaussian_filtering(pn.SPECKLE_MODERATE, "speckle\\speckle_moderate_gaussian.png", 3)
gaussian_filtering(pn.SPECKLE_HIGH, "speckle\\speckle_high_gaussian.png", 3)

gaussian_filtering(pn.UNIFORM_LOW, "uniform\\uniform_low_gaussian.png", 3)
gaussian_filtering(pn.UNIFORM_MODERATE, "uniform\\uniform_moderate_gaussian.png", 3)
gaussian_filtering(pn.UNIFORM_HIGH, "uniform\\uniform_high_gaussian.png", 3)

gaussian_filtering(pn.PEPPER_LOW, "pepper\\pepper_low_gaussian.png", 3)
gaussian_filtering(pn.PEPPER_MODERATE, "pepper\\pepper_moderate_gaussian.png", 3)
gaussian_filtering(pn.PEPPER_HIGH, "pepper\\pepper_high_gaussian.png", 3)

gaussian_filtering(pn.SALT_LOW, "salt\\salt_low_gaussian.png", 3)
gaussian_filtering(pn.SALT_MODERATE, "salt\\salt_moderate_gaussian.png", 3)
gaussian_filtering(pn.SALT_HIGH, "salt\\salt_high_gaussian.png", 3)

gaussian_filtering(pn.SP_LOW, "salt&pepper\\salt&pepper_low_gaussian.png", 3)
gaussian_filtering(pn.SP_MODERATE, "salt&pepper\\salt&pepper_moderate_gaussian.png", 3)
gaussian_filtering(pn.SP_HIGH, "salt&pepper\\salt&pepper_high_gaussian.png", 3)
print("GAUSSIAN BLUR")

# # TV FILTER
total_variation_filtering(pn.GAUSSIAN_LOW, "gaussian\\gaussian_low_tv.png", 0.1)
total_variation_filtering(pn.GAUSSIAN_MODERATE, "gaussian\\gaussian_moderate_tv.png", 0.1)
total_variation_filtering(pn.GAUSSIAN_HIGH, "gaussian\\gaussian_high_tv.png", 0.1)

total_variation_filtering(pn.LAPLACIAN_LOW, "laplacian\\laplacian_low_tv.png", 0.1)
total_variation_filtering(pn.LAPLACIAN_MODERATE, "laplacian\\laplacian_moderate_tv.png", 0.1)
total_variation_filtering(pn.LAPLACIAN_HIGH, "laplacian\\laplacian_high_tv.png", 0.1)

total_variation_filtering(pn.POISSON_LOW, "poisson\\poisson_low_tv.png", 0.1)
total_variation_filtering(pn.POISSON_MODERATE, "poisson\\poisson_moderate_tv.png", 0.1)
total_variation_filtering(pn.POISSON_HIGH, "poisson\\poisson_high_tv.png", 0.1)

total_variation_filtering(pn.SPECKLE_LOW, "speckle\\speckle_low_tv.png", 0.1)
total_variation_filtering(pn.SPECKLE_MODERATE, "speckle\\speckle_moderate_tv.png", 0.1)
total_variation_filtering(pn.SPECKLE_HIGH, "speckle\\speckle_high_tv.png", 0.1)

total_variation_filtering(pn.UNIFORM_LOW, "uniform\\uniform_low_tv.png", 0.1)
total_variation_filtering(pn.UNIFORM_MODERATE, "uniform\\uniform_moderate_tv.png", 0.1)
total_variation_filtering(pn.UNIFORM_HIGH, "uniform\\uniform_high_tv.png", 0.1)

total_variation_filtering(pn.PEPPER_LOW, "pepper\\pepper_low_tv.png", 0.1)
total_variation_filtering(pn.PEPPER_MODERATE, "pepper\\pepper_moderate_tv.png", 0.1)
total_variation_filtering(pn.PEPPER_HIGH, "pepper\\pepper_high_tv.png", 0.1)

total_variation_filtering(pn.SALT_LOW, "salt\\salt_low_tv.png", 0.1)
total_variation_filtering(pn.SALT_MODERATE, "salt\\salt_moderate_tv.png", 0.1)
total_variation_filtering(pn.SALT_HIGH, "salt\\salt_high_tv.png", 0.1)

total_variation_filtering(pn.SP_LOW, "salt&pepper\\salt&pepper_low_tv.png", 0.1)
total_variation_filtering(pn.SP_MODERATE, "salt&pepper\\salt&pepper_moderate_tv.png", 0.1)
total_variation_filtering(pn.SP_HIGH, "salt&pepper\\salt&pepper_high_tv.png", 0.1)
print("TV")

# MEDIAN
median_filtering(pn.GAUSSIAN_LOW, "gaussian\\gaussian_low_median_median.png", 3)
median_filtering(pn.GAUSSIAN_MODERATE, "gaussian\\gaussian_moderate_median.png", 3)
median_filtering(pn.GAUSSIAN_HIGH, "gaussian\\gaussian_high_median.png", 3)

median_filtering(pn.LAPLACIAN_LOW, "laplacian\\laplacian_low_median.png", 3)
median_filtering(pn.LAPLACIAN_MODERATE, "laplacian\\laplacian_moderate_median.png", 3)
median_filtering(pn.LAPLACIAN_HIGH, "laplacian\\laplacian_high_median.png", 3)

median_filtering(pn.POISSON_LOW, "poisson\\poisson_low_median.png", 3)
median_filtering(pn.POISSON_MODERATE, "poisson\\poisson_moderate_median.png", 3)
median_filtering(pn.POISSON_HIGH, "poisson\\poisson_high_median.png", 3)

median_filtering(pn.SPECKLE_LOW, "speckle\\speckle_low_median.png", 3)
median_filtering(pn.SPECKLE_MODERATE, "speckle\\speckle_moderate_median.png", 3)
median_filtering(pn.SPECKLE_HIGH, "speckle\\speckle_high_median.png", 3)

median_filtering(pn.UNIFORM_LOW, "uniform\\uniform_low_median.png", 3)
median_filtering(pn.UNIFORM_MODERATE, "uniform\\uniform_moderate_median.png", 3)
median_filtering(pn.UNIFORM_HIGH, "uniform\\uniform_high_median.png", 3)

median_filtering(pn.SALT_LOW, "salt\\salt_low_median.png", 3)
median_filtering(pn.SALT_MODERATE, "salt\\salt_moderate_median.png", 3)
median_filtering(pn.SALT_HIGH, "salt\\salt_high_median.png", 3)

median_filtering(pn.PEPPER_LOW, "pepper\\pepper_low_median.png", 3)
median_filtering(pn.PEPPER_MODERATE, "pepper\\pepper_moderate_median.png", 3)
median_filtering(pn.PEPPER_HIGH, "pepper\\pepper_high_median.png", 3)

median_filtering(pn.SP_LOW, "salt&pepper\\s&p_low_median.png", 3)
median_filtering(pn.SP_MODERATE, "salt&pepper\\s&p_moderate_median.png", 3)
median_filtering(pn.SP_HIGH, "salt&pepper\\s&p_high_median.png", 3)
print("MEDIAN")
#
# # WAVELET
wavelet_filtering(pn.GAUSSIAN_LOW, "gaussian\\gaussian_low_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.GAUSSIAN_MODERATE, "gaussian\\gaussian_moderate_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.GAUSSIAN_HIGH, "gaussian\\gaussian_high_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")

wavelet_filtering(pn.LAPLACIAN_LOW, "laplacian\\laplacian_low_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.LAPLACIAN_MODERATE, "laplacian\\laplacian_moderate_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.LAPLACIAN_HIGH, "laplacian\\laplacian_high_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")

wavelet_filtering(pn.POISSON_LOW, "poisson\\poisson_low_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.POISSON_MODERATE, "poisson\\poisson_moderate_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.POISSON_HIGH, "poisson\\poisson_high_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")

wavelet_filtering(pn.SPECKLE_LOW, "speckle\\speckle_low_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.SPECKLE_MODERATE, "speckle\\speckle_moderate_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.SPECKLE_HIGH, "speckle\\speckle_high_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")

wavelet_filtering(pn.UNIFORM_LOW, "uniform\\uniform_low_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.UNIFORM_MODERATE, "uniform\\uniform_moderate_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.UNIFORM_HIGH, "uniform\\uniform_high_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")

wavelet_filtering(pn.PEPPER_LOW, "pepper\\pepper_low_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.PEPPER_MODERATE, "pepper\\pepper_moderate_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.PEPPER_HIGH, "pepper\\pepper_high_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")

wavelet_filtering(pn.SALT_LOW, "salt\\salt_low_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.SALT_MODERATE, "salt\\salt_moderate_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.SALT_HIGH, "salt\\salt_high_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")

wavelet_filtering(pn.SP_LOW, "salt&pepper\\salt&pepper_low_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.SP_MODERATE, "salt&pepper\\salt&pepper_moderate_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
wavelet_filtering(pn.SP_HIGH, "salt&pepper\\salt&pepper_high_wavelet.png", method="BayesShrink", mode='soft',
                  wavelet_levels=3, wavelet="coif5")
print("WAVELET")
