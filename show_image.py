import matplotlib.pyplot as plt
from tensorflow.keras import datasets

# Tải dữ liệu CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Danh sách tên lớp
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Hiển thị toàn bộ ảnh trong tập huấn luyện (hoặc một phần nếu quá nhiều)
def show_images(images, labels, class_names, num_rows=10, num_cols=10):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    axes = axes.flatten()
    for i in range(num_rows * num_cols):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        axes[i].set_title(class_names[labels[i][0]])
    plt.tight_layout()
    plt.show()

# Gọi hàm hiển thị ảnh
show_images(train_images, train_labels, class_names)