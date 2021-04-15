from util import PROJECT_PATH,DEFAULT_IMAGE,SEGMENTED

IMAGE_NUMBER = "4"    # 1, 2, 3, 4
FILE_EXTENSION = ".png"
DEFAULT_IMAGE = DEFAULT_IMAGE + IMAGE_NUMBER + FILE_EXTENSION
FOLDER = "\\images\\chosen_denoised\\chosen_denoised_image" + IMAGE_NUMBER


GAUSSIAN_LOW = PROJECT_PATH + FOLDER + "\\gaussian\\gaussian_low" + FILE_EXTENSION
GAUSSIAN_MODERATE = PROJECT_PATH + FOLDER + "\\gaussian\\gaussian_moderate" + FILE_EXTENSION
GAUSSIAN_HIGH = PROJECT_PATH + FOLDER + "\\gaussian\\gaussian_high" + FILE_EXTENSION

LAPLACIAN_LOW = PROJECT_PATH + FOLDER + "\\laplacian\\laplacian_low" + FILE_EXTENSION
LAPLACIAN_MODERATE = PROJECT_PATH + FOLDER + "\\laplacian\\laplacian_moderate" + FILE_EXTENSION
LAPLACIAN_HIGH = PROJECT_PATH + FOLDER + "\\laplacian\\laplacian_high" + FILE_EXTENSION

POISSON_LOW = PROJECT_PATH + FOLDER + "\\poisson\\poisson_low" + FILE_EXTENSION
POISSON_MODERATE = PROJECT_PATH + FOLDER + "\\poisson\\poisson_moderate" + FILE_EXTENSION
POISSON_HIGH = PROJECT_PATH + FOLDER + "\\poisson\\poisson_high" + FILE_EXTENSION

SPECKLE_LOW = PROJECT_PATH + FOLDER + "\\speckle\\speckle_low" + FILE_EXTENSION
SPECKLE_MODERATE = PROJECT_PATH + FOLDER + "\\speckle\\speckle_moderate" + FILE_EXTENSION
SPECKLE_HIGH = PROJECT_PATH + FOLDER + "\\speckle\\speckle_high" + FILE_EXTENSION

UNIFORM_LOW = PROJECT_PATH + FOLDER + "\\uniform\\uniform_low" + FILE_EXTENSION
UNIFORM_MODERATE = PROJECT_PATH + FOLDER + "\\uniform\\uniform_moderate" + FILE_EXTENSION
UNIFORM_HIGH = PROJECT_PATH + FOLDER + "\\uniform\\uniform_high" + FILE_EXTENSION

PEPPER_LOW = PROJECT_PATH + FOLDER + "\\pepper\\pepper_low" + FILE_EXTENSION
PEPPER_MODERATE = PROJECT_PATH + FOLDER + "\\pepper\\pepper_moderate" + FILE_EXTENSION
PEPPER_HIGH = PROJECT_PATH + FOLDER + "\\pepper\\pepper_high" + FILE_EXTENSION

SALT_LOW = PROJECT_PATH + FOLDER + "\\salt\\salt_low" + FILE_EXTENSION
SALT_MODERATE = PROJECT_PATH + FOLDER + "\\salt\\salt_moderate" + FILE_EXTENSION
SALT_HIGH = PROJECT_PATH + FOLDER + "\\salt\\salt_high" + FILE_EXTENSION

SP_LOW = PROJECT_PATH + FOLDER + "\\salt&pepper\\salt&pepper_low" + FILE_EXTENSION
SP_MODERATE = PROJECT_PATH + FOLDER + "\\salt&pepper\\salt&pepper_moderate" + FILE_EXTENSION
SP_HIGH = PROJECT_PATH + FOLDER + "\\salt&pepper\\salt&pepper_high" + FILE_EXTENSION