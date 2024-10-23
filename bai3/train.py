import os
import cv2
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Hàm đọc và resize ảnh
def load_images_from_folder(folder, img_size=(64, 64)):
    images = []
    labels = []
    total_files = sum([len(files) for _, _, files in os.walk(folder)])  # Đếm tổng số tệp
    processed_files = 0  # Biến đếm tệp đã xử lý

    for label, sub_folder in enumerate(os.listdir(folder)):
        sub_folder_path = os.path.join(folder, sub_folder)
        for filename in os.listdir(sub_folder_path):
            img_path = os.path.join(sub_folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
                processed_files += 1  # Tăng biến đếm
                if processed_files % 100 == 0:  # Thông báo mỗi 100 tệp
                    print(f"Đã xử lý {processed_files}/{total_files} tệp hình ảnh.")
    print(f"Hoàn tất! Đã xử lý tổng cộng {processed_files} tệp hình ảnh.")
    return np.array(images), np.array(labels)

# Đường dẫn đến thư mục ảnh
flowers_data_dir = r'C:\Users\anhtr\Downloads\bai3\archive\train'  # Đường dẫn đến bộ dữ liệu hoa

# Tải và xử lý dữ liệu
print("Bắt đầu tải dữ liệu...")
flowers_images, flowers_labels = load_images_from_folder(flowers_data_dir)
print("Dữ liệu đã được tải thành công.")

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(flowers_images, flowers_labels, test_size=0.2, random_state=42)

# Flatten ảnh (64x64x3 thành 1D vector)
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

# Huấn luyện và đánh giá SVM
start_time = time.time()
svm_model = SVC()
svm_model.fit(X_train_flatten, y_train)
svm_train_time = time.time() - start_time
y_pred_svm = svm_model.predict(X_test_flatten)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')

# Huấn luyện và đánh giá KNN
start_time = time.time()
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_flatten, y_train)
knn_train_time = time.time() - start_time
y_pred_knn = knn_model.predict(X_test_flatten)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average='macro')
recall_knn = recall_score(y_test, y_pred_knn, average='macro')

# Huấn luyện và đánh giá Decision Tree
start_time = time.time()
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_flatten, y_train)
dt_train_time = time.time() - start_time
y_pred_dt = dt_model.predict(X_test_flatten)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='macro')
recall_dt = recall_score(y_test, y_pred_dt, average='macro')

# In kết quả
print(f"SVM - Thời gian: {svm_train_time:.4f}s, Độ chính xác: {accuracy_svm:.4f}, Precision: {precision_svm:.4f}, Recall: {recall_svm:.4f}")
print(f"KNN - Thời gian: {knn_train_time:.4f}s, Độ chính xác: {accuracy_knn:.4f}, Precision: {precision_knn:.4f}, Recall: {recall_knn:.4f}")
print(f"Cây Quyết Định - Thời gian: {dt_train_time:.4f}s, Độ chính xác: {accuracy_dt:.4f}, Precision: {precision_dt:.4f}, Recall: {recall_dt:.4f}")
