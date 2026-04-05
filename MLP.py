import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a):
    return a * (1.0 - a)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def tanh(z):
    return np.tanh(z)

def tanh_deriv(a):
    return 1.0 - a ** 2

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


class NeuralNetwork:
    def __init__(self, layer_sizes, activations, learning_rate=0.01):
        """
        layer_sizes: 列表, 如 [2, 4, 3, 1] 表示输入2, 两个隐藏层, 输出1
        activations: 每层的激活函数名称列表, 如 ['relu', 'relu', 'sigmoid']
        """

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activations = activations
        self.lr = learning_rate

        # He 初始化权重
        self.W = {}
        self.b = {}
        for l in range(1, self.num_layers + 1):
            self.W[l] = np.random.randn(layer_sizes[l], layer_sizes[l - 1]) * np.sqrt(2.0 / layer_sizes[l - 1])
            self.b[l] = np.zeros((layer_sizes[l], 1))


    def _activate(self, z, name):
        if name == "sigmoid":
            return sigmoid(z)
        if name == "relu":
            return relu(z)
        if name == "tanh":
            return tanh(z)
        if name == "softmax":
            return softmax(z)
        raise ValueError(f"Unknown activation: {name}")
    

    def _activate_deriv(self, z, a, name):
        if name == "sigmoid":
            return sigmoid_deriv(z)
        if name == "relu":
            return relu_deriv(z)
        if name == "tanh":
            return tanh_deriv(z)
        if name == "softmax":
            return np.ones_like(z)  # softmax 的梯度在 backward 中特殊处理
        raise ValueError(f"Unknown activation: {name}")
    


    def forward(self, X):
        """
        X: (n_features, m), m 为样本数
        返回缓存字典, 保存每层的 z 和 a
        """

        cache = {"a0": X}

        a = X
        for l in range(1, self.num_layers + 1):
            z = self.W[l] @ a + self.b[l]
            a = self._activate(z, self.activations[l - 1])
            cache[f"z{l}"] = z
            cache[f"a{l}"] = a

        return a, cache
    
    def compute_loss(self, y_pred, y_true):
        """交叉熵损失 (支持二分类和多分类)"""
        m = y_true.shape[1]
        eps = 1E-8
        if self.activations[-1] == "softmax":
            l = -(1.0 / m) * np.sum(y_true * np.log(y_pred + eps))
        else:
            l = -(1.0 / m) * np.sum(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        return l

    def backward(self, cache, y_true):
        m = y_true.shape[1]
        grads = {}

        # 计算输出层的 delta
        L = self.num_layers
        aL = cache[f"a{L}"]

        # 对于 sigmoid/softmax + 交叉熵, delta = aL - y
        if self.activations[-1] in ("sigmoid", "softmax"):
            delta = aL - y_true
        else:
            dL_da = aL - y_true
            g_prime = self._activate_deriv(cache[f"z{L}"], aL, self.activations[-1])
            delta = dL_da * g_prime

        # 逐层反向传播
        for l in range(L, 0, -1):
            a_prev = cache[f"a{l - 1}"]

            # 参数梯度
            grads[f"dW{l}"] = (1.0 / m) * (delta @ a_prev.T)
            grads[f"db{l}"] = (1.0 / m) * np.sum(delta, axis=1, keepdims=True)

            if l > 1:
                # 将 delta 传播到上一层
                z_prev = cache[f"z{l - 1}"]
                g_prime = self._activate_deriv(z_prev, a_prev, self.activations[l - 2])
                delta = (self.W[l].T @ delta) * g_prime

        return grads

    
    def update_params(self, grads):
        for l in range(1, self.num_layers + 1):
            self.W[l] -= self.lr * grads[f"dW{l}"]
            self.b[l] -= self.lr * grads[f"db{l}"]

    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        X: (n_features, m)
        y: (1, m)
        """

        for epoch in range(epochs):
            # 前向传播
            y_pred, cache = self.forward(X)

            # 计算损失
            loss = self.compute_loss(y_pred, y)

            # 反向传播
            grads = self.backward(cache, y)

            # 更新参数
            self.update_params(grads)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

    def predict(self, X, threshold=0.5):
        y_pred, _ = self.forward(X)
        if self.activations[-1] == "softmax":
            return np.argmax(y_pred, axis=0).reshape(1, -1)
        return (y_pred > threshold).astype(float)


def gradient_check(nn, X, y, epsilon=1e-7):
    """对比数值梯度与反向传播计算的梯度"""
    _, cache = nn.forward(X)
    grads = nn.backward(cache, y)

    for l in range(1, nn.num_layers + 1):
        for param, name in [(nn.W, 'W'), (nn.b, 'b')]:
            analytical = grads[f'd{name}{l}']
            numerical = np.zeros_like(param[l])

            it = np.nditer(param[l], flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                original = param[l][idx]

                param[l][idx] = original + epsilon
                loss_plus, _ = nn.forward(X)
                loss_plus = nn.compute_loss(loss_plus, y)

                param[l][idx] = original - epsilon
                loss_minus, _ = nn.forward(X)
                loss_minus = nn.compute_loss(loss_minus, y)

                numerical[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                param[l][idx] = original
                it.iternext()

            diff = np.linalg.norm(analytical - numerical) / \
                    (np.linalg.norm(analytical) + np.linalg.norm(numerical) + 1e-8)
            status = "✓" if diff < 1e-5 else "✗"
            print(f"Layer {l} {name}: relative diff = {diff:.2e} {status}")




def test_xor():
    np.random.seed(42)

    X = np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ])
    y = np.array([
        [0, 1, 1, 0]
    ])

    nn = NeuralNetwork(
        layer_sizes=[2, 8, 4, 1],
        activations=['relu', 'relu', 'sigmoid'],
        learning_rate=0.1
    )
    nn.train(X, y, epochs=2000)

    # 预测
    preds = nn.predict(X)
    print("\nPredictions:")
    print(preds)
    print("Accuracy:", np.mean(preds == y))

    gradient_check(nn, X, y)


def test_mnist():
    """使用手写神经网络在 MNIST 数据集上进行训练和测试"""
    import gzip
    import os
    from urllib import request

    # --- 下载并加载 MNIST 数据 ---
    mnist_dir = os.path.join(os.path.dirname(__file__), "mnist_data")
    os.makedirs(mnist_dir, exist_ok=True)

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }

    local_paths = {}
    for key, fname in files.items():
        path = os.path.join(mnist_dir, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            request.urlretrieve(base_url + fname, path)
        local_paths[key] = path

    def load_images(path):
        with gzip.open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return data.reshape(-1, 28 * 28).astype(np.float64) / 255.0

    def load_labels(path, num_classes=10):
        with gzip.open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        one_hot = np.zeros((num_classes, len(data)), dtype=np.float64)
        one_hot[data, np.arange(len(data))] = 1.0
        return one_hot, data

    print("Loading MNIST data...")
    X_train = load_images(local_paths["train_images"]).T   # (784, 60000)
    y_train_oh, y_train_raw = load_labels(local_paths["train_labels"])
    X_test = load_images(local_paths["test_images"]).T     # (784, 10000)
    y_test_oh, y_test_raw = load_labels(local_paths["test_labels"])

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # --- 构建网络 ---
    np.random.seed(42)
    nn = NeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        activations=["relu", "relu", "softmax"],
        learning_rate=0.5,
    )

    # --- 训练 (使用 mini-batch) ---
    epochs = 50
    batch_size = 64
    m = X_train.shape[1]

    for epoch in range(epochs):
        # 打乱数据
        perm = np.random.permutation(m)
        X_shuffled = X_train[:, perm]
        y_shuffled = y_train_oh[:, perm]

        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            X_batch = X_shuffled[:, start:end]
            y_batch = y_shuffled[:, start:end]

            y_pred, cache = nn.forward(X_batch)
            epoch_loss += nn.compute_loss(y_pred, y_batch)
            num_batches += 1

            grads = nn.backward(cache, y_batch)
            nn.update_params(grads)

        epoch_loss /= num_batches

        # 每 5 个 epoch 评估一次测试集准确率
        if (epoch + 1) % 5 == 0 or epoch == 0:
            preds = nn.predict(X_test)
            acc = np.mean(preds.flatten() == y_test_raw)
            print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {epoch_loss:.4f} | Test Acc: {acc:.4f}")
        else:
            print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {epoch_loss:.4f}")

    # 最终评估
    preds = nn.predict(X_test)
    acc = np.mean(preds.flatten() == y_test_raw)
    print(f"\nFinal Test Accuracy: {acc:.4f}")



if __name__ == "__main__":
    test_mnist()