# Dataset

The MNIST dataset used in this project is **automatically downloaded** by TensorFlow/Keras when the script is first run. No manual download is needed.

```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

## Dataset Details

| Property         | Value                          |
|------------------|-------------------------------|
| Name             | MNIST                         |
| Training samples | 60,000                        |
| Testing samples  | 10,000                        |
| Image size       | 28 × 28 pixels (grayscale)   |
| Classes          | 10 (digits 0–9)               |
| Source           | `tensorflow.keras.datasets`   |

The downloaded data is cached locally at `~/.keras/datasets/mnist.npz` and is not tracked by Git.
