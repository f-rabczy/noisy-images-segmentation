from wand.image import Image
from path_names import PathNamesNoise as pns

IMAGE_PATH = pns.DEFAULT_IMAGE
OUTPUT_FOLDER_PATH = pns.PROJECT_PATH + "\\images\\noise\\noisy_image" + pns.IMAGE_NUMBER + "\\"


def add_noise_and_save(path_name, mode, attenuate, output_name):
    with Image(filename=path_name) as img:
        img.noise(mode, attenuate)
        img.save(filename=OUTPUT_FOLDER_PATH + mode + "\\" + output_name)


add_noise_and_save(IMAGE_PATH, mode="laplacian", attenuate=1, output_name="\\laplacianLowNoise.png")
add_noise_and_save(IMAGE_PATH, mode="laplacian", attenuate=2, output_name="\\laplacianModerateNoise.png")
add_noise_and_save(IMAGE_PATH, mode="laplacian", attenuate=4, output_name="\\laplacianHighNoise.png")

add_noise_and_save(IMAGE_PATH, mode="uniform", attenuate=10, output_name="\\uniformLowNoise.png")
add_noise_and_save(IMAGE_PATH, mode="uniform", attenuate=20, output_name="\\uniformModerateNoise.png")
add_noise_and_save(IMAGE_PATH, mode="uniform", attenuate=40, output_name="\\uniformHighNoise.png")

add_noise_and_save(IMAGE_PATH, mode="poisson", attenuate=10, output_name="\\poissonLowNoise.png")
add_noise_and_save(IMAGE_PATH, mode="poisson", attenuate=5, output_name="\\poissonModerateNoise.png")
add_noise_and_save(IMAGE_PATH, mode="poisson", attenuate=2.5, output_name="\\poissonHighNoise.png")
