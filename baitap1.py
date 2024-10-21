import cv2
import numpy as np
from matplotlib import pyplot as plt

# Đọc ảnh đầu vào
image = cv2.imread('c:\Users\anhtr\Downloads\anhtrung.jpg', 0)  # Ảnh đầu vào, chế độ grayscale

# 1. Ảnh âm tính
negative_image = 255 - image

# 2. Tăng độ tương phản (bằng cách dùng histogram equalization)
equalized_image = cv2.equalizeHist(image)

# 3. Biến đổi log (thường dùng để làm sáng các vùng tối)
c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(1 + image))
log_image = np.array(log_image, dtype=np.uint8)

# 4. Cân bằng histogram
hist_equalized_image = cv2.equalizeHist(image)

# Hiển thị kết quả
titles = ['Original Image', 'Negative Image', 'Equalized Image', 'Log Transformed', 'Histogram Equalized']
images = [image, negative_image, equalized_image, log_image, hist_equalized_image]

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
