import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào
img = cv2.imread('anhtrung.jpg', 0)  # Đọc ảnh ở dạng grayscale

# Áp dụng toán tử Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Gradient theo hướng x
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Gradient theo hướng y
sobel = cv2.magnitude(sobelx, sobely)  # Tính tổng của hai hướng

# Áp dụng toán tử Laplace Gaussian
laplacian_gaussian = cv2.Laplacian(cv2.GaussianBlur(img, (3,3), 0), cv2.CV_64F)

# Hiển thị kết quả
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Ảnh gốc')

plt.subplot(1, 3, 2)
plt.imshow(sobel, cmap='gray')
plt.title('Toán tử Sobel')

plt.subplot(1, 3, 3)
plt.imshow(laplacian_gaussian, cmap='gray')
plt.title('Toán tử Laplace Gaussian')

plt.show()
