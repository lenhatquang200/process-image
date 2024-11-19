import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Bước 1: Tải dữ liệu CIFAR-10 và chuẩn bị dữ liệu
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Chuẩn hóa dữ liệu về khoảng [0, 1] để giảm độ lệch số liệu
train_images, test_images = train_images / 255.0, test_images / 255.0

# Bước 2: Xây dựng mô hình CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # 10 nhãn cho CIFAR-10
])

# Bước 3: Biên dịch mô hình
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Bước 4: Huấn luyện mô hình
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Bước 5: Đánh giá mô hình trên tập kiểm tra
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Độ chính xác trên tập kiểm tra: {test_acc}')

# Bước 6: Lưu mô hình sau khi huấn luyện (tuỳ chọn)
# Bạn có thể lưu mô hình để sử dụng sau
model.save('cifar10_cnn_model.h5')

# Bước 7: Hàm chuẩn bị ảnh để dự đoán
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))  # Điều chỉnh ảnh về kích thước 32x32
    img_array = image.img_to_array(img)  # Chuyển ảnh thành mảng
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch (1, 32, 32, 3)
    img_array = img_array / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    return img_array

# Bước 8: Đường dẫn tới ảnh bạn muốn dự đoán
img_path = 'images.jpg'  # Thay đường dẫn thành ảnh của bạn
img_array = prepare_image(img_path)

# Hiển thị ảnh để kiểm tra
plt.imshow(image.load_img(img_path))
plt.title("Ảnh để dự đoán")
plt.show()

# Bước 9: Dự đoán ảnh sử dụng mô hình đã huấn luyện
predictions = model.predict(img_array)

# Lấy nhãn dự đoán
predicted_label = tf.argmax(predictions[0])
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Lấy nhãn dự đoán

# Hiển thị tên của nhãn
predicted_class = class_names[predicted_label]
print(f'Nhãn dự đoán của ảnh: {predicted_class}')