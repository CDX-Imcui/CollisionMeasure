# import opencv
import cv2
# import hyperlpr3
import hyperlpr3 as lpr3
import time
# Instantiate object
start_time = time.time()
catcher = lpr3.LicensePlateCatcher()
# load image
image = cv2.imread("image_005.jpg")
# print result
print(catcher(image))
end_time = time.time()
print(f"耗时: {end_time - start_time:.4f} 秒")
