import cv2

MODEL_PATH = "ESPCN_x4.pb"
IMAGE_PATH = "test_car3.png"

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(MODEL_PATH)
sr.setModel("espcn", 4)

img = cv2.imread(IMAGE_PATH)
upscaled = sr.upsample(img)

cv2.imwrite("test_car3_HD.png", upscaled)
print("HD image saved as test_car3_HD.png")
