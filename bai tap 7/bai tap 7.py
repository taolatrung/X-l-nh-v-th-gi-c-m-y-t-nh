# Import các thư viện cần thiết
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đường dẫn đến ảnh đầu vào
image_path = 'image.png'

# Bước 1: Đọc ảnh
# Chuyển ảnh sang dạng ảnh xám (grayscale) để dễ dàng xử lý
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Bước 2: Làm mờ ảnh bằng Gaussian Blur
# Sử dụng Gaussian Blur để giảm nhiễu trước khi áp dụng các thuật toán phát hiện cạnh
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Bước 3: Phát hiện cạnh bằng thuật toán Sobel
# Áp dụng Sobel theo cả hai hướng x và y, sau đó tính độ lớn vector để kết hợp cả hai kết quả
sobelx = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)  # Sobel theo hướng x
sobely = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)  # Sobel theo hướng y
sobel = cv2.magnitude(sobelx, sobely)  # Tính độ lớn của vector để lấy kết quả cuối cùng của Sobel

# Bước 4: Phát hiện cạnh bằng thuật toán Prewitt
# Áp dụng kernel Prewitt theo hướng x và y, sau đó tính độ lớn vector
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewittx = cv2.filter2D(gaussian_blur, -1, kernelx)  # Prewitt theo hướng x
prewitty = cv2.filter2D(gaussian_blur, -1, kernely)  # Prewitt theo hướng y
prewitt = cv2.magnitude(prewittx.astype(float), prewitty.astype(float))  # Kết quả cuối cùng của Prewitt

# Bước 5: Phát hiện cạnh bằng thuật toán Roberts
# Sử dụng kernel Roberts để phát hiện cạnh theo hướng chéo
roberts_cross_v = np.array([[1, 0], [0, -1]])
roberts_cross_h = np.array([[0, 1], [-1, 0]])
robertsx = cv2.filter2D(gaussian_blur, -1, roberts_cross_v)  # Roberts theo hướng chéo
robertsy = cv2.filter2D(gaussian_blur, -1, roberts_cross_h)  # Roberts theo hướng chéo ngược lại
roberts = cv2.magnitude(robertsx.astype(float), robertsy.astype(float))  # Kết quả cuối cùng của Roberts

# Bước 6: Phát hiện cạnh bằng thuật toán Canny
# Thiết lập ngưỡng dưới và trên để xác định các cạnh
canny = cv2.Canny(gaussian_blur, 100, 200)

# Bước 7: Hiển thị kết quả của từng thuật toán
plt.figure(figsize=(15, 10))

# Ảnh gốc
plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray'), plt.title("Original Image")

# Kết quả của Sobel Edge Detection
plt.subplot(2, 3, 2), plt.imshow(sobel, cmap='gray'), plt.title("Sobel Edge Detection")

# Kết quả của Prewitt Edge Detection
plt.subplot(2, 3, 3), plt.imshow(prewitt, cmap='gray'), plt.title("Prewitt Edge Detection")

# Kết quả của Roberts Edge Detection
plt.subplot(2, 3, 4), plt.imshow(roberts, cmap='gray'), plt.title("Roberts Edge Detection")

# Kết quả của Canny Edge Detection
plt.subplot(2, 3, 5), plt.imshow(canny, cmap='gray'), plt.title("Canny Edge Detection")

# Ảnh Gaussian Blurred (ảnh đã được làm mờ để chuẩn bị cho các thuật toán phát hiện cạnh)
plt.subplot(2, 3, 6), plt.imshow(gaussian_blur, cmap='gray'), plt.title("Gaussian Blurred Image")

# Tùy chỉnh bố cục và hiển thị
plt.tight_layout()
plt.show()
