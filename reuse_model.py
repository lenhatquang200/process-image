import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('cifar10_cnn_model.h5')

# Danh sách các nhãn tương ứng với các lớp của mô hình CIFAR-10
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))  # Kích thước hình ảnh phải phù hợp
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Chuẩn hóa hình ảnh
    return img_array

def predict(img_path):
    img = prepare_image(img_path)
    predictions = model.predict(img)
    return predictions

if __name__ == '__main__':
    predictions = predict('images.jpg')
    predicted_label = labels[np.argmax(predictions)]  # Lấy nhãn dự đoán
    confidence = np.max(predictions)  # Lấy độ tin cậy của dự đoán
    print(f"Dự đoán: {predicted_label} với độ tin cậy: {confidence:.2f}")
