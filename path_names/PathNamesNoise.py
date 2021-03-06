from util import PROJECT_PATH, DENOISED_IMAGES, NOISY_IMAGES

IMAGE_NUMBER = "1"  # 1, 2, 3, 4

NOISE_FOLDER = "\\images\\noise\\noisy_image" + IMAGE_NUMBER
FILE_EXTENSION = ".png"

DEFAULT_IMAGE = PROJECT_PATH + "\\images\\default_image" + IMAGE_NUMBER + FILE_EXTENSION

GAUSSIAN_LOW = PROJECT_PATH + NOISE_FOLDER + "\\gaussian\\gaussianLowNoise" + FILE_EXTENSION
GAUSSIAN_MODERATE = PROJECT_PATH + NOISE_FOLDER + "\\gaussian\\gaussianModerateNoise" + FILE_EXTENSION
GAUSSIAN_HIGH = PROJECT_PATH + NOISE_FOLDER + "\\gaussian\\gaussianHighNoise" + FILE_EXTENSION

LAPLACIAN_LOW = PROJECT_PATH + NOISE_FOLDER + "\\laplacian\\laplacianLowNoise" + FILE_EXTENSION
LAPLACIAN_MODERATE = PROJECT_PATH + NOISE_FOLDER + "\\laplacian\\laplacianModerateNoise" + FILE_EXTENSION
LAPLACIAN_HIGH = PROJECT_PATH + NOISE_FOLDER + "\\laplacian\\laplacianHighNoise" + FILE_EXTENSION

POISSON_LOW = PROJECT_PATH + NOISE_FOLDER + "\\poisson\\poissonLowNoise" + FILE_EXTENSION
POISSON_MODERATE = PROJECT_PATH + NOISE_FOLDER + "\\poisson\\poissonModerateNoise" + FILE_EXTENSION
POISSON_HIGH = PROJECT_PATH + NOISE_FOLDER + "\\poisson\\poissonHighNoise" + FILE_EXTENSION

SPECKLE_LOW = PROJECT_PATH + NOISE_FOLDER + "\\speckle\\speckleLowNoise" + FILE_EXTENSION
SPECKLE_MODERATE = PROJECT_PATH + NOISE_FOLDER + "\\speckle\\speckleModerateNoise" + FILE_EXTENSION
SPECKLE_HIGH = PROJECT_PATH + NOISE_FOLDER + "\\speckle\\speckleHighNoise" + FILE_EXTENSION

UNIFORM_LOW = PROJECT_PATH + NOISE_FOLDER + "\\uniform\\uniformLowNoise" + FILE_EXTENSION
UNIFORM_MODERATE = PROJECT_PATH + NOISE_FOLDER + "\\uniform\\uniformModerateNoise" + FILE_EXTENSION
UNIFORM_HIGH = PROJECT_PATH + NOISE_FOLDER + "\\uniform\\uniformHighNoise" + FILE_EXTENSION

PEPPER_LOW = PROJECT_PATH + NOISE_FOLDER + "\\pepper\\pepperLowNoise" + FILE_EXTENSION
PEPPER_MODERATE = PROJECT_PATH + NOISE_FOLDER + "\\pepper\\pepperModerateNoise" + FILE_EXTENSION
PEPPER_HIGH = PROJECT_PATH + NOISE_FOLDER + "\\pepper\\pepperHighNoise" + FILE_EXTENSION

SALT_LOW = PROJECT_PATH + NOISE_FOLDER + "\\salt\\saltLowNoise" + FILE_EXTENSION
SALT_MODERATE = PROJECT_PATH + NOISE_FOLDER + "\\salt\\saltModerateNoise" + FILE_EXTENSION
SALT_HIGH = PROJECT_PATH + NOISE_FOLDER + "\\salt\\saltHighNoise" + FILE_EXTENSION

SP_LOW = PROJECT_PATH + NOISE_FOLDER + "\\salt&pepper\\salt&pepperLowNoise" + FILE_EXTENSION
SP_MODERATE = PROJECT_PATH + NOISE_FOLDER + "\\salt&pepper\\salt&pepperModerateNoise" + FILE_EXTENSION
SP_HIGH = PROJECT_PATH + NOISE_FOLDER + "\\salt&pepper\\salt&pepperHighNoise" + FILE_EXTENSION
