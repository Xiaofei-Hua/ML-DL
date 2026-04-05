import numpy as np

# ============================================================
#  激活函数
# ============================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a):
    return a * (1.0 - a)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def tanh_fn(z):
    return np.tanh(z)

def tanh_deriv(a):
    return 1.0 - a ** 2

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


# ============================================================
#  优化器
# ============================================================

class SGD:
    """普通 SGD + Momentum + Nesterov"""

    def __init__(self, lr=0.01, momentum=0.0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.v = None  # 速度 (velocity)

    def step(self, params, grads):
        """
        params: dict  {l: Wl} 或 {l: bl}
        grads:  dict  {f"dW{l}": ...}  或 {f"db{l}": ...}
        key_prefix: "W" 或 "b"
        """
        if self.v is None:
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        for k in params:
            g = grads[k]
            self.v[k] = self.momentum * self.v[k] + g

            if self.nesterov:
                # Nesterov: 先"往前走一步"，再算梯度方向的修正
                params[k] -= self.lr * (self.momentum * self.v[k] + g)
            else:
                params[k] -= self.lr * self.v[k]


class Adam:
    """Adam 优化器 (Adaptive Moment Estimation)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # 一阶矩 (均值)
        self.v = None  # 二阶矩 (方差)
        self.t = 0     # 时间步

    def step(self, params, grads):
        self.t += 1

        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2

            # 偏差修正
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================
#  MLP 网络主体
# ============================================================

class MLP:
    def __init__(self, layer_sizes, activations,
                 optimizer="adam", learning_rate=0.001,
                 l2_lambda=0.0, dropout_rate=0.0, batchnorm=False):
        """
        参数
        ----------
        layer_sizes : list   如 [784, 128, 64, 10]
        activations : list   每层激活函数, 如 ['relu', 'relu', 'softmax']
        optimizer   : str    'sgd', 'sgd_momentum', 'nesterov', 'adam'
        learning_rate : float
        l2_lambda   : float  L2 正则化系数 (0 = 不使用)
        dropout_rate: float  Dropout 概率 (0 = 不使用)
        batchnorm   : bool   是否使用 Batch Normalization
        """
        assert len(activations) == len(layer_sizes) - 1

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activations = activations
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.batchnorm = batchnorm
        self.training = True  # 训练/推理模式标志

        # --- 权重初始化 (He) ---
        self.W = {}
        self.b = {}
        for l in range(1, self.num_layers + 1):
            self.W[l] = np.random.randn(layer_sizes[l], layer_sizes[l - 1]) * np.sqrt(2.0 / layer_sizes[l - 1])
            self.b[l] = np.zeros((layer_sizes[l], 1))

        # --- BatchNorm 参数 ---
        self.bn_gamma = {}
        self.bn_beta = {}
        self.bn_running_mean = {}
        self.bn_running_var = {}
        for l in range(1, self.num_layers + 1):
            # 只对非输出层启用 BN (输出层通常不需要)
            if batchnorm and l < self.num_layers:
                self.bn_gamma[l] = np.ones((layer_sizes[l], 1))
                self.bn_beta[l] = np.zeros((layer_sizes[l], 1))
                self.bn_running_mean[l] = np.zeros((layer_sizes[l], 1))
                self.bn_running_var[l] = np.ones((layer_sizes[l], 1))

        # --- 创建优化器 ---
        opt_map = {
            "sgd":           SGD(lr=learning_rate, momentum=0.0),
            "sgd_momentum":  SGD(lr=learning_rate, momentum=0.9),
            "nesterov":      SGD(lr=learning_rate, momentum=0.9, nesterov=True),
            "adam":          Adam(lr=learning_rate),
        }
        if optimizer not in opt_map:
            raise ValueError(f"Unknown optimizer: {optimizer}, choose from {list(opt_map.keys())}")

        # W 和 b 使用同一个优化器类的不同实例 (各自维护独立的动量/矩)
        self.opt_W = opt_map[optimizer]
        self.opt_b = opt_map[optimizer]

        # BN 参数也有独立的优化器
        if batchnorm:
            self.opt_bn_gamma = Adam(lr=learning_rate)
            self.opt_bn_beta = Adam(lr=learning_rate)

    # ---------- 激活函数 ----------

    def _activate(self, z, name):
        if name == "sigmoid": return sigmoid(z)
        if name == "relu":    return relu(z)
        if name == "tanh":    return tanh_fn(z)
        if name == "softmax": return softmax(z)
        raise ValueError(f"Unknown activation: {name}")

    def _activate_deriv(self, z, a, name):
        if name == "sigmoid": return sigmoid_deriv(a)
        if name == "relu":    return relu_deriv(z)
        if name == "tanh":    return tanh_deriv(a)
        if name == "softmax": return np.ones_like(z)
        raise ValueError(f"Unknown activation: {name}")

    # ---------- 前向传播 ----------

    def forward(self, X):
        cache = {"a0": X}
        a = X

        for l in range(1, self.num_layers + 1):
            z = self.W[l] @ a + self.b[l]

            # Batch Normalization (激活函数之前)
            if self._has_bn(l):
                z = self._batchnorm_forward(z, l, cache)

            a = self._activate(z, self.activations[l - 1])

            # Dropout (训练时才启用)
            if self.dropout_rate > 0 and self.training and l < self.num_layers:
                mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(np.float64)
                a = a * mask / (1.0 - self.dropout_rate)  # inverted dropout
                cache[f"drop_mask{l}"] = mask

            cache[f"z{l}"] = z
            cache[f"a{l}"] = a

        return a, cache

    def _has_bn(self, l):
        return self.batchnorm and l in self.bn_gamma

    def _batchnorm_forward(self, z, l, cache):
        if self.training:
            mu = np.mean(z, axis=1, keepdims=True)
            var = np.var(z, axis=1, keepdims=True)
            # 指数移动平均更新 running stats
            self.bn_running_mean[l] = 0.9 * self.bn_running_mean[l] + 0.1 * mu
            self.bn_running_var[l] = 0.9 * self.bn_running_var[l] + 0.1 * var
        else:
            mu = self.bn_running_mean[l]
            var = self.bn_running_var[l]

        z_norm = (z - mu) / np.sqrt(var + 1e-8)
        out = self.bn_gamma[l] * z_norm + self.bn_beta[l]

        cache[f"bn_mu{l}"] = mu
        cache[f"bn_var{l}"] = var
        cache[f"bn_z_norm{l}"] = z_norm
        return out

    # ---------- 损失函数 ----------

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[1]
        eps = 1e-8
        if self.activations[-1] == "softmax":
            data_loss = -(1.0 / m) * np.sum(y_true * np.log(y_pred + eps))
        else:
            data_loss = -(1.0 / m) * np.sum(
                y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
            )

        # L2 正则化项
        if self.l2_lambda > 0:
            l2_term = 0.0
            for l in range(1, self.num_layers + 1):
                l2_term += np.sum(self.W[l] ** 2)
            data_loss += (self.l2_lambda / (2 * m)) * l2_term

        return data_loss

    # ---------- 反向传播 ----------

    def backward(self, cache, y_true):
        m = y_true.shape[1]
        grads = {}

        L = self.num_layers
        aL = cache[f"a{L}"]

        # 输出层 delta
        if self.activations[-1] in ("sigmoid", "softmax"):
            delta = aL - y_true
        else:
            g_prime = self._activate_deriv(cache[f"z{L}"], aL, self.activations[-1])
            delta = (aL - y_true) * g_prime

        for l in range(L, 0, -1):
            a_prev = cache[f"a{l - 1}"]

            # BatchNorm 反向传播
            if self._has_bn(l):
                d_bn_gamma = np.sum(delta * cache[f"bn_z_norm{l}"], axis=1, keepdims=True)
                d_bn_beta = np.sum(delta, axis=1, keepdims=True)
                grads[f"dgamma{l}"] = d_bn_gamma
                grads[f"dbeta{l}"] = d_bn_beta

                z_norm = cache[f"bn_z_norm{l}"]
                var = cache[f"bn_var{l}"]
                # 对 z 的梯度
                d_z_norm = delta * self.bn_gamma[l]
                inv_std = 1.0 / np.sqrt(var + 1e-8)
                d_var = np.sum(d_z_norm * (z_norm * (-0.5) * inv_std ** 3), axis=1, keepdims=True)
                d_mu = np.sum(d_z_norm * (-inv_std), axis=1, keepdims=True)
                delta = d_z_norm * inv_std + d_var * 2.0 * z_norm / m + d_mu / m

            # 梯度 (加入 L2 正则化)
            grads[f"dW{l}"] = (1.0 / m) * (delta @ a_prev.T)
            if self.l2_lambda > 0:
                grads[f"dW{l}"] += (self.l2_lambda / m) * self.W[l]
            grads[f"db{l}"] = (1.0 / m) * np.sum(delta, axis=1, keepdims=True)

            if l > 1:
                z_prev = cache[f"z{l - 1}"]
                g_prime = self._activate_deriv(z_prev, a_prev, self.activations[l - 2])
                delta = (self.W[l].T @ delta) * g_prime

                # Dropout 反向传播: 把被丢弃的神经元梯度也置零
                if self.dropout_rate > 0 and f"drop_mask{l - 1}" in cache:
                    delta = delta * cache[f"drop_mask{l - 1}"] / (1.0 - self.dropout_rate)

        return grads

    # ---------- 参数更新 ----------

    def update_params(self, grads):
        # W 和 b
        W_grads = {l: grads[f"dW{l}"] for l in range(1, self.num_layers + 1)}
        b_grads = {l: grads[f"db{l}"] for l in range(1, self.num_layers + 1)}
        self.opt_W.step(self.W, W_grads)
        self.opt_b.step(self.b, b_grads)

        # BN 参数
        if self.batchnorm:
            gamma_grads = {}
            beta_grads = {}
            for l in range(1, self.num_layers):
                if f"dgamma{l}" in grads:
                    gamma_grads[l] = grads[f"dgamma{l}"]
                    beta_grads[l] = grads[f"dbeta{l}"]
            self.opt_bn_gamma.step(self.bn_gamma, gamma_grads)
            self.opt_bn_beta.step(self.bn_beta, beta_grads)

    # ---------- 预测 ----------

    def predict(self, X):
        self.training = False
        y_pred, _ = self.forward(X)
        self.training = True
        if self.activations[-1] == "softmax":
            return np.argmax(y_pred, axis=0).reshape(1, -1)
        return (y_pred > 0.5).astype(float)

    # ---------- 训练 ----------

    def train(self, X, y, epochs=1000, batch_size=64, verbose=True, X_val=None, y_val=None):
        """
        支持全批量或 mini-batch 训练, 可选验证集。
        X: (n_features, m)
        y: (n_classes, m)  (one-hot) 或 (1, m) (二分类)
        """
        m = X.shape[1]
        has_val = X_val is not None and y_val is not None

        for epoch in range(epochs):
            self.training = True
            perm = np.random.permutation(m)
            X_shuf = X[:, perm]
            y_shuf = y[:, perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                X_b = X_shuf[:, start:end]
                y_b = y_shuf[:, start:end]

                y_pred, cache = self.forward(X_b)
                epoch_loss += self.compute_loss(y_pred, y_b)
                n_batches += 1

                grads = self.backward(cache, y_b)
                self.update_params(grads)

            epoch_loss /= n_batches

            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch + 1:4d}/{epochs} | Loss: {epoch_loss:.4f}"
                if has_val:
                    val_acc = np.mean(self.predict(X_val).flatten() == y_val.flatten())
                    msg += f" | Val Acc: {val_acc:.4f}"
                print(msg)


# ============================================================
#  测试: MNIST
# ============================================================

def test_mnist():
    import gzip
    import os
    from urllib import request

    # --- 下载数据 ---
    mnist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_data")
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
            raw = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        one_hot = np.zeros((num_classes, len(raw)), dtype=np.float64)
        one_hot[raw, np.arange(len(raw))] = 1.0
        return one_hot, raw

    print("Loading MNIST data...")
    X_train = load_images(local_paths["train_images"]).T
    y_train_oh, y_train_raw = load_labels(local_paths["train_labels"])
    X_test = load_images(local_paths["test_images"]).T
    y_test_oh, y_test_raw = load_labels(local_paths["test_labels"])
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # --- 对比实验 ---
    configs = [
        {
            "name": "Baseline (SGD, 无正则化)",
            "optimizer": "sgd", "lr": 0.5,
            "l2": 0.0, "dropout": 0.0, "bn": False,
        },
        {
            "name": "SGD + Momentum",
            "optimizer": "sgd_momentum", "lr": 0.1,
            "l2": 0.0, "dropout": 0.0, "bn": False,
        },
        {
            "name": "SGD + Nesterov + L2 + Dropout",
            "optimizer": "nesterov", "lr": 0.1,
            "l2": 1e-4, "dropout": 0.2, "bn": False,
        },
        {
            "name": "Adam + L2 + Dropout + BatchNorm",
            "optimizer": "adam", "lr": 0.001,
            "l2": 1e-4, "dropout": 0.2, "bn": True,
        },
    ]

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"  {cfg['name']}")
        print(f"{'='*60}")

        np.random.seed(42)
        model = MLP(
            layer_sizes=[784, 256, 128, 10],
            activations=["relu", "relu", "softmax"],
            optimizer=cfg["optimizer"],
            learning_rate=cfg["lr"],
            l2_lambda=cfg["l2"],
            dropout_rate=cfg["dropout"],
            batchnorm=cfg["bn"],
        )
        model.train(
            X_train, y_train_oh,
            epochs=50, batch_size=64, verbose=True,
            X_val=X_test, y_val=y_test_raw,
        )

        acc = np.mean(model.predict(X_test).flatten() == y_test_raw)
        print(f">>> Final Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    test_mnist()
