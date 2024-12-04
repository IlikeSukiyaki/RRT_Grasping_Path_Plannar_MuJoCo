import cv2

# 读取一个RGB图像
image = cv2.imread('/home/da/project2_deliver(1)/project2_deliver/models/011_banana/texture_map.png')

# OpenCV默认读取图像为BGR格式，因此不需要转换，直接保存即可
img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite('/home/da/project2_deliver(1)/project2_deliver/models/011_banana/texture_map_bgr.png', img_bgr)