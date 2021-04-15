import cv2
from path_names import PathNamesSegmented as pns

a = cv2.imread(pns.GAUSSIAN_LOW, 0)
a_pixels = cv2.countNonZero(a)
b = cv2.imread(pns.IDEAL_MASK1, 0)
b_pixels = cv2.countNonZero(b)

result = cv2.bitwise_and(a, b)
result_pixels = cv2.countNonZero(result)

print("A pixels:", a_pixels)
print("B pixels:", b_pixels)
print("RESULT:", result_pixels)
