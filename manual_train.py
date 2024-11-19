import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

# Giả sử nhãn của ảnh mới là "cat" (nhãn = 3 trong CIFAR-10)
new_image_label = 3  # "cat" trong CIFAR-10 có nhãn là 3

# Bước 1: Chuẩn bị ảnh mới
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))  # Điều chỉnh ảnh về kích thước 32x32
    img_array = image.img_to_array(img)  # Chuyển ảnh thành mảng
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch (1, 32, 32, 3)
    img_array = img_array / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    return img_array

# Đường dẫn tới ảnh mới
img_path = 'new_image.jpg'  # Thay đường dẫn thành ảnh của bạn
new_img_array = prepare_image(img_path)

# Bước 2: Thêm ảnh và nhãn vào dữ liệu huấn luyện
# Thêm ảnh mới vào tập huấn luyện
train_images = np.append(train_images, new_img_array, axis=0)
train_labels = np.append(train_labels, np.array([[new_image_label]]), axis=0)

# Chuyển nhãn thành dạng one-hot encoding
train_labels = to_categorical(train_labels, 10)  # 10 lớp cho CIFAR-10

# Bước 3: Huấn luyện lại mô hình với dữ liệu mới
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Bước 4: Đánh giá mô hình sau khi huấn luyện
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Độ chính xác trên tập kiểm tra sau khi huấn luyện lại: {test_acc}')

# Lưu lại mô hình đã huấn luyện
model.save('cifar10_cnn_model_updated.h5')
