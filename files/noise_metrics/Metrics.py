from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

from path_names import PathNamesDenoised as pn
import cv2

print(pn.GAUSSIAN_LOW)
print("\n")

print(pn.DEFAULT_IMAGE)

default_image = cv2.imread(pn.DEFAULT_IMAGE)

gaussian_low = cv2.imread(pn.GAUSSIAN_LOW)
gaussian_moderate = cv2.imread(pn.GAUSSIAN_MODERATE)
gaussian_high = cv2.imread(pn.GAUSSIAN_HIGH)

psnr_noisy_gaussian_l = peak_signal_noise_ratio(default_image, gaussian_low)
psnr_noisy_gaussian_m = peak_signal_noise_ratio(default_image, gaussian_moderate)
psnr_noisy_gaussian_h = peak_signal_noise_ratio(default_image, gaussian_high)

ssim_noisy_gaussian_l = ssim(default_image, gaussian_low, multichannel=True,
                             data_range=default_image.max() - default_image.min())
ssim_noisy_gaussian_m = ssim(default_image, gaussian_moderate, multichannel=True,
                             data_range=default_image.max() - default_image.min())
ssim_noisy_gaussian_h = ssim(default_image, gaussian_high, multichannel=True,
                             data_range=default_image.max() - default_image.min())




speckle_low = cv2.imread(pn.SPECKLE_LOW)
speckle_moderate = cv2.imread(pn.SPECKLE_MODERATE)
speckle_high = cv2.imread(pn.SPECKLE_HIGH)


laplacian_low = cv2.imread(pn.LAPLACIAN_LOW)
laplacian_moderate = cv2.imread(pn.LAPLACIAN_MODERATE)
laplacian_high = cv2.imread(pn.LAPLACIAN_HIGH)

uniform_low = cv2.imread(pn.UNIFORM_LOW)
uniform_moderate = cv2.imread(pn.UNIFORM_MODERATE)
uniform_high = cv2.imread(pn.UNIFORM_HIGH)

poisson_low = cv2.imread(pn.POISSON_LOW)
poisson_moderate = cv2.imread(pn.POISSON_MODERATE)
poisson_high = cv2.imread(pn.POISSON_HIGH)

pepper_low = cv2.imread(pn.PEPPER_LOW)
pepper_moderate = cv2.imread(pn.PEPPER_MODERATE)
pepper_high = cv2.imread(pn.PEPPER_HIGH)

salt_low = cv2.imread(pn.SALT_LOW)
salt_moderate = cv2.imread(pn.SALT_MODERATE)
salt_high = cv2.imread(pn.SALT_HIGH)

sp_low = cv2.imread(pn.SP_LOW)
sp_moderate = cv2.imread(pn.SP_MODERATE)
sp_high = cv2.imread(pn.SP_HIGH)

# PSNR
psnr_noisy_gaussian_l = peak_signal_noise_ratio(default_image, gaussian_low)
psnr_noisy_gaussian_m = peak_signal_noise_ratio(default_image, gaussian_moderate)
psnr_noisy_gaussian_h = peak_signal_noise_ratio(default_image, gaussian_high)

psnr_noisy_speckle_l = peak_signal_noise_ratio(default_image, speckle_low)
psnr_noisy_speckle_m = peak_signal_noise_ratio(default_image, speckle_moderate)
psnr_noisy_speckle_h = peak_signal_noise_ratio(default_image, speckle_high)

psnr_noisy_laplacian_l = peak_signal_noise_ratio(default_image, laplacian_low)
psnr_noisy_laplacian_m = peak_signal_noise_ratio(default_image, laplacian_moderate)
psnr_noisy_laplacian_h = peak_signal_noise_ratio(default_image, laplacian_high)

psnr_noisy_uniform_l = peak_signal_noise_ratio(default_image, uniform_low)
psnr_noisy_uniform_m = peak_signal_noise_ratio(default_image, uniform_moderate)
psnr_noisy_uniform_h = peak_signal_noise_ratio(default_image, uniform_high)

psnr_noisy_poisson_l = peak_signal_noise_ratio(default_image, poisson_low)
psnr_noisy_poisson_m = peak_signal_noise_ratio(default_image, poisson_moderate)
psnr_noisy_poisson_h = peak_signal_noise_ratio(default_image, poisson_high)

psnr_pepper_low = peak_signal_noise_ratio(default_image, pepper_low)
psnr_pepper_moderate = peak_signal_noise_ratio(default_image, pepper_moderate)
psnr_pepper_high = peak_signal_noise_ratio(default_image, pepper_high)

psnr_salt_low = peak_signal_noise_ratio(default_image, salt_low)
psnr_salt_moderate = peak_signal_noise_ratio(default_image, salt_moderate)
psnr_salt_high = peak_signal_noise_ratio(default_image, salt_high)

psnr_sp_low = peak_signal_noise_ratio(default_image, sp_low)
psnr_sp_moderate = peak_signal_noise_ratio(default_image, sp_moderate)
psnr_sp_high = peak_signal_noise_ratio(default_image, sp_high)



ssim_noisy_speckle_l = ssim(default_image, speckle_low, multichannel=True,
                            data_range=default_image.max() - default_image.min())
ssim_noisy_speckle_m = ssim(default_image, speckle_moderate, multichannel=True,
                            data_range=default_image.max() - default_image.min())
ssim_noisy_speckle_h = ssim(default_image, speckle_high, multichannel=True,
                            data_range=default_image.max() - default_image.min())

ssim_noisy_laplacian_l = ssim(default_image, laplacian_low, multichannel=True,
                              data_range=default_image.max() - default_image.min())
ssim_noisy_laplacian_m = ssim(default_image, laplacian_moderate, multichannel=True,
                              data_range=default_image.max() - default_image.min())
ssim_noisy_laplacian_h = ssim(default_image, laplacian_high, multichannel=True,
                              data_range=default_image.max() - default_image.min())

ssim_noisy_uniform_l = ssim(default_image, uniform_low, multichannel=True,
                            data_range=default_image.max() - default_image.min())
ssim_noisy_uniform_m = ssim(default_image, uniform_moderate, multichannel=True,
                            data_range=default_image.max() - default_image.min())
ssim_noisy_uniform_h = ssim(default_image, uniform_high, multichannel=True,
                            data_range=default_image.max() - default_image.min())

ssim_noisy_poisson_l = ssim(default_image, poisson_low, multichannel=True,
                            data_range=default_image.max() - default_image.min())
ssim_noisy_poisson_m = ssim(default_image, poisson_moderate, multichannel=True,
                            data_range=default_image.max() - default_image.min())
ssim_noisy_poisson_h = ssim(default_image, poisson_high, multichannel=True,
                            data_range=default_image.max() - default_image.min())

ssim_pepper_low = ssim(default_image, pepper_low, multichannel=True, data_range=default_image.max() - default_image.min())
ssim_pepper_moderate = ssim(default_image, pepper_moderate, multichannel=True,
                            data_range=default_image.max() - default_image.min())
ssim_pepper_high = ssim(default_image, pepper_high, multichannel=True,
                        data_range=default_image.max() - default_image.min())

ssim_salt_low = ssim(default_image, salt_low, multichannel=True, data_range=default_image.max() - default_image.min())
ssim_salt_moderate = ssim(default_image, salt_moderate, multichannel=True,
                          data_range=default_image.max() - default_image.min())
ssim_salt_high = ssim(default_image, salt_high, multichannel=True, data_range=default_image.max() - default_image.min())

ssim_sp_low = ssim(default_image, sp_low, multichannel=True, data_range=default_image.max() - default_image.min())
ssim_sp_moderate = ssim(default_image, sp_moderate, multichannel=True,
                        data_range=default_image.max() - default_image.min())
ssim_sp_high = ssim(default_image, sp_high, multichannel=True, data_range=default_image.max() - default_image.min())

# PSNR
print("Gaussian")
print("PSNR ORIGINAL VS GAUSSIAN LOW", psnr_noisy_gaussian_l)
print("PSNR ORIGINAL VS GAUSSIAN MODERATE", psnr_noisy_gaussian_m)
print("PSNR ORIGINAL VS GAUSIAN HIGH", psnr_noisy_gaussian_h)

print("\n")

print("Laplacian")
print("PSNR ORIGINAL VS  LAPLACIAN LOW", psnr_noisy_laplacian_l)
print("PSNR ORIGINAL VS  LAPLACIAN MODERATE", psnr_noisy_laplacian_m)
print("PSNR ORIGINAL VS  LAPLACIAN HIGH", psnr_noisy_laplacian_h)
print("\n")

#
print("Poisson")
print("PSNR ORIGINAL VS POISSON LOW", psnr_noisy_poisson_l)
print("PSNR ORIGINAL VS POISSON MODERATE", psnr_noisy_poisson_m)
print("PSNR ORIGINAL VS POISSON HIGH", psnr_noisy_poisson_h)
print("\n")

print("Speckle")
print("PSNR ORIGINAL VS SPECKLE LOW", psnr_noisy_speckle_l)
print("PSNR ORIGINAL VS SPECKLE MODERATE", psnr_noisy_speckle_m)
print("PSNR ORIGINAL VS SPECKLE HIGH", psnr_noisy_speckle_h)
# print("\n")
print("\n")

print("Uniform")
print("PSNR ORIGINAL VS UNIFORM LOW", psnr_noisy_uniform_l)
print("PSNR ORIGINAL VS  UNIFORM MODERATE", psnr_noisy_uniform_m)
print("PSNR ORIGINAL VS UNIFORM HIGH", psnr_noisy_uniform_h)
print("\n")

print("\n")
print("Pepper")
print("PSNR ORIGINAL VS PEPPER LOW", psnr_pepper_low)
print("PSNR ORIGINAL VS PEPPER MODERATE", psnr_pepper_moderate)
print("PSNR ORIGINAL VS PEPPER HIGH", psnr_pepper_high)

print("\n")

print("Salt")
print("PSNR ORIGINAL VS SALT LOW", psnr_salt_low)
print("PSNR ORIGINAL VS SALT MODERATE", psnr_salt_moderate)
print("PSNR ORIGINAL VS SALT HIGH", psnr_salt_high)

print("\n")

print("Salt&Pepper")
print("PSNR ORIGINAL VS SALT&PEPPER LOW", psnr_sp_low)
print("PSNR ORIGINAL VS SALT&PEPPER MODERATE", psnr_sp_moderate)
print("PSNR ORIGINAL VS SALT&PEPPER HIGH", psnr_sp_high)

print("\n")

# SSIM
print("Gaussian")

print("\n")
print("SSIM ORIGINAL VS GAUSSIAN LOW", ssim_noisy_gaussian_l)
print("SSIM ORIGINAL VS GAUSIAN MODERATE", ssim_noisy_gaussian_m)
print("SSIM ORIGINAL VS GAUSSIAN HIGH", ssim_noisy_gaussian_h)

print("Laplacian")

print("SSIM ORIGINAL VS LAPLACIAN LOW", ssim_noisy_laplacian_l)
print("SSIM ORIGINAL VS LAPLACIAN MODERATE", ssim_noisy_laplacian_m)
print("SSIM ORIGINAL VS LAPLACIAN HIGH", ssim_noisy_laplacian_h)
#
print("Poisson")

print("\n")
print("SSIM ORIGINAL VS POISSON LOW", ssim_noisy_poisson_l)
print("SSIM ORIGINAL VS POISSON MODERATE", ssim_noisy_poisson_m)
print("SSIM ORIGINAL VS POISSON HIGH", ssim_noisy_poisson_h)
print("Speckle")

print("\n")
print("SSIM ORIGINAL VS SPECKLE LOW", ssim_noisy_speckle_l)
print("SSIM ORIGINAL VS SPECKLE MODERATE", ssim_noisy_speckle_m)
print("SSIM ORIGINAL VS SPECKLE HIGH", ssim_noisy_speckle_h)
print("\n")

print("Uniform")

print("\n")
print("SSIM ORIGINAL VS UNIFORM LOW", ssim_noisy_uniform_l)
print("SSIM ORIGINAL VS UNIFORM MODERATE", ssim_noisy_uniform_m)
print("SSIM ORIGINAL VS UNIFORM HIGH", ssim_noisy_uniform_h)

print("\n")
print("Pepper")

print("SSIM ORIGINAL VS PEPPER LOW", ssim_pepper_low)
print("SSIM ORIGINAL VS PEPPER MODERATE", ssim_pepper_moderate)
print("SSIM ORIGINAL VS PEPPER HIGH", ssim_pepper_high)
print("\n")

print("Salt")

print("SSIM ORIGINAL VS SALT LOW", ssim_salt_low)
print("SSIM ORIGINAL VS SALT MODERATE", ssim_salt_moderate)
print("SSIM ORIGINAL VS SALT HIGH", ssim_salt_high)
print("\n")

print("Salt&Pepper")

print("SSIM ORIGINAL VS SALT&PEPPER LOW", ssim_sp_low)
print("SSIM ORIGINAL VS SALT&PEPPER MODERATE", ssim_sp_moderate)
print("SSIM ORIGINAL VS SALT&PEPPER HIGH", ssim_sp_high)
print("\n")
