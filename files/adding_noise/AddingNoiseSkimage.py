import skimage
from skimage import io
from path_names import PathNamesNoise as pns

IMAGE_PATH = pns.DEFAULT_IMAGE
OUTPUT_FOLDER_PATH = pns.PROJECT_PATH + "\\images\\noise\\noisy_image" + pns.IMAGE_NUMBER + "\\"
image = skimage.io.imread(IMAGE_PATH)


def add_low_noise_and_save(img, mode):
    if mode == 'gaussian' or mode == 'speckle':
        noisy_image = skimage.util.random_noise(img, mode=mode, var=0.02)
        eight_bit = skimage.img_as_ubyte(noisy_image)
        skimage.io.imsave(OUTPUT_FOLDER_PATH + mode + "\\" + mode + "LowNoise.png", eight_bit)
    if mode == 'salt' or mode == 'pepper':
        noisy_image = skimage.util.random_noise(img, mode=mode, amount=0.025)
        eight_bit = skimage.img_as_ubyte(noisy_image)
        skimage.io.imsave(OUTPUT_FOLDER_PATH + mode + "\\" + mode + "LowNoise.png", eight_bit)
    if mode == 's&p':
        noisy_image = skimage.util.random_noise(img, mode=mode, amount=0.025)
        eight_bit = skimage.img_as_ubyte(noisy_image)
        skimage.io.imsave(OUTPUT_FOLDER_PATH + 'salt&pepper' + "\\" + 'salt&pepper' + "LowNoise.png", eight_bit)


def add_moderate_noise_and_save(img, mode):
    if mode == 'gaussian' or mode == 'speckle':
        noisy_image = skimage.util.random_noise(img, mode=mode, var=0.04)
        eight_bit = skimage.img_as_ubyte(noisy_image)
        skimage.io.imsave(OUTPUT_FOLDER_PATH + mode + "\\" + mode + "ModerateNoise.png", eight_bit)
    if mode == 'salt' or mode == 'pepper':
        noisy_image = skimage.util.random_noise(img, mode=mode, amount=0.05)
        eight_bit = skimage.img_as_ubyte(noisy_image)
        skimage.io.imsave(OUTPUT_FOLDER_PATH + mode + "\\" + mode + "ModerateNoise.png", eight_bit)
    if mode == 's&p':
        noisy_image = skimage.util.random_noise(img, mode=mode, amount=0.05)
        eight_bit = skimage.img_as_ubyte(noisy_image)
        skimage.io.imsave(OUTPUT_FOLDER_PATH + 'salt&pepper' + "\\" + 'salt&pepper' + "ModerateNoise.png", eight_bit)


def add_high_noise_and_save(img, mode):
    if mode == 'gaussian' or mode == 'speckle':
        noisy_image = skimage.util.random_noise(img, mode=mode, var=0.08)
        eight_bit = skimage.img_as_ubyte(noisy_image)
        skimage.io.imsave(OUTPUT_FOLDER_PATH + mode + "\\" + mode + "HighNoise.png", eight_bit)
    if mode == 'salt' or mode == 'pepper':
        noisy_image = skimage.util.random_noise(img, mode=mode, amount=0.1)
        eight_bit = skimage.img_as_ubyte(noisy_image)
        skimage.io.imsave(OUTPUT_FOLDER_PATH + mode + "\\" + mode + "HighNoise.png", eight_bit)
    if mode == 's&p':
        noisy_image = skimage.util.random_noise(img, mode=mode, amount=0.1)
        eight_bit = skimage.img_as_ubyte(noisy_image)
        skimage.io.imsave(OUTPUT_FOLDER_PATH + 'salt&pepper' + "\\" + 'salt&pepper' + "HighNoise.png", eight_bit)


add_low_noise_and_save(image, "gaussian")
add_low_noise_and_save(image, "salt")
add_low_noise_and_save(image, "pepper")
add_low_noise_and_save(image, "s&p")
add_low_noise_and_save(image, "speckle")

add_moderate_noise_and_save(image, "gaussian")
add_moderate_noise_and_save(image, "salt")
add_moderate_noise_and_save(image, "pepper")
add_moderate_noise_and_save(image, "s&p")
add_moderate_noise_and_save(image, "speckle")

add_high_noise_and_save(image, "gaussian")
add_high_noise_and_save(image, "salt")
add_high_noise_and_save(image, "pepper")
add_high_noise_and_save(image, "s&p")
add_high_noise_and_save(image, "speckle")
