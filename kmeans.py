"""
K-Means 聚类算法 — 纯 NumPy 手搓实现

包含:
  - KMeans          标准 K-Means (支持 random / k-means++ / manual 初始化)
  - MiniBatchKMeans 小批量 K-Means (大数据集适用)
  - 辅助函数        elbow_method, silhouette_score, make_blobs

核心优化:
  - 距离计算使用 (a-b)^2 = a^2 + b^2 - 2ab 展开, 避免 (N,1,D)-(1,K,D) 广播大数组
  - 向量化标签分配, 无 Python 循环
  - 就地操作减少内存分配
"""

from __future__ import annotations

import numpy as np
from typing import Literal


# ---------------------------------------------------------------------------
#  距离计算
# ---------------------------------------------------------------------------

def euclidean_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    计算两组样本之间的欧氏距离矩阵.

    Parameters
    ----------
    X : (n_samples_x, n_features)
    Y : (n_samples_y, n_features)

    Returns
    -------
    distances : (n_samples_x, n_samples_y)
        D[i, j] = ||X[i] - Y[j]||_2
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    X_sq = np.sum(X ** 2, axis=1)          # (n_x,)
    Y_sq = np.sum(Y ** 2, axis=1)          # (n_y,)
    D_sq = X_sq[:, np.newaxis] + Y_sq[np.newaxis, :] - 2.0 * (X @ Y.T)
    np.maximum(D_sq, 0.0, out=D_sq)        # 消除浮点误差导致的负值
    return np.sqrt(D_sq, out=D_sq)         # 就地开方


def euclidean_distances_sq(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """与 euclidean_distances 相同, 但返回平方距离 (省去 sqrt, 用于比较大小)."""
    X_sq = np.sum(X ** 2, axis=1)
    Y_sq = np.sum(Y ** 2, axis=1)
    D_sq = X_sq[:, np.newaxis] + Y_sq[np.newaxis, :] - 2.0 * (X @ Y.T)
    np.maximum(D_sq, 0.0, out=D_sq)
    return D_sq


# ---------------------------------------------------------------------------
#  初始化策略
# ---------------------------------------------------------------------------

def _init_random(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """从数据中随机选 k 个样本作为初始质心."""
    indices = rng.choice(X.shape[0], size=k, replace=False)
    return X[indices].copy()


def _init_kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    K-Means++ 初始化 (Arthur & Vassilvitskii, 2007).

    核心思想: 逐个选质心, 每个新质心的选择概率与到已选质心的最短距离成正比.
    这样可以避免初始质心扎堆, 显著减少收敛所需迭代次数.

    时间复杂度: O(n * k * d), 但换来更少的迭代, 整体通常更快.
    """
    n, d = X.shape
    centers = np.empty((k, d), dtype=X.dtype)

    # 第一个质心: 随机选
    idx = rng.integers(n)
    centers[0] = X[idx]

    # 记录每个点到最近已选质心的平方距离
    # 初始化为到第一个质心的距离
    min_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)  # (n,)

    for c in range(1, k):
        # 按距离比例采样: 距离越远, 被选中概率越大
        probs = min_dist_sq / min_dist_sq.sum()
        idx = rng.choice(n, p=probs)
        centers[c] = X[idx]

        # 更新最小距离 (只需和新加入的质心比较)
        new_dist_sq = np.sum((X - centers[c]) ** 2, axis=1)
        np.minimum(min_dist_sq, new_dist_sq, out=min_dist_sq)

    return centers


# ---------------------------------------------------------------------------
#  K-Means 主类
# ---------------------------------------------------------------------------

class KMeans:
    """
    标准 K-Means 聚类算法.

    算法流程 (Lloyd's algorithm):
        1. 初始化 k 个质心
        2. 分配: 将每个样本分到最近质心所属的簇
        3. 更新: 将每个质心移到其簇内样本的均值位置
        4. 重复 2-3 直到质心不再变化 (或变化小于 tol)

    Parameters
    ----------
    n_clusters : int
        簇的数量 k.
    init : {'random', 'k-means++'} 或 np.ndarray
        初始化方式. 'k-means++' 为默认, 效果通常最好.
        如果传入 ndarray, shape 应为 (n_clusters, n_features).
    n_init : int
        独立运行算法的次数, 取 inertia 最小的那次结果.
        (相当于多次随机重启, 选最优)
    max_iter : int
        单次运行的最大迭代次数.
    tol : float
        质心移动量小于此值时认为收敛, 提前停止.
    random_state : int | None
        随机种子, 用于可复现.
    verbose : bool
        是否打印迭代过程.

    Attributes
    ----------
    cluster_centers_ : (n_clusters, n_features)
        最终质心位置.
    labels_ : (n_samples,)
        每个样本的簇标签 (从 0 开始).
    inertia_ : float
        样本到其所属簇质心的距离平方和: Σ ||x_i - μ_{c_i}||^2.
       越小越好, 是衡量聚类紧致程度的指标.
    n_iter_ : int
        收敛时的迭代次数.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal["random", "k-means++"] | np.ndarray = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def _check_data(self, X: np.ndarray) -> np.ndarray:
        """数据校验与转换."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X 必须是二维数组, 当前 ndim={X.ndim}")
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"样本数 ({X.shape[0]}) 不能少于 n_clusters ({self.n_clusters})"
            )
        return X

    def _single_run(self, X: np.ndarray, rng: np.random.Generator):
        """
        执行一次完整的 K-Means 迭代.

        Returns
        -------
        centers : 质心
        labels  : 标签
        inertia : 惯性
        n_iter  : 迭代次数
        """
        n, d = X.shape
        k = self.n_clusters

        # ---- 初始化质心 ----
        if isinstance(self.init, np.ndarray):
            centers = self.init.copy()
        elif self.init == "k-means++":
            centers = _init_kmeans_plus_plus(X, k, rng)
        elif self.init == "random":
            centers = _init_random(X, k, rng)
        else:
            raise ValueError(f"不支持的初始化方式: {self.init}")

        labels = np.empty(n, dtype=np.intp)
        prev_centers = centers.copy()

        for iteration in range(1, self.max_iter + 1):
            # ---- Step 1: 分配 (Assignment) ----
            # 计算每个样本到每个质心的距离, 取最近的
            dist_sq = euclidean_distances_sq(X, centers)  # (n, k)
            np.argmin(dist_sq, axis=1, out=labels)

            # ---- Step 2: 更新 (Update) ----
            prev_centers[:] = centers
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    # 质心 = 簇内样本均值
                    centers[c] = X[mask].mean(axis=0)
                # else: 空簇, 质心保持不变

            # ---- 收敛判断 ----
            shift = np.sum((centers - prev_centers) ** 2)
            if self.verbose:
                print(f"  迭代 {iteration:3d} | 质心位移 {shift:.6f}")

            if shift <= self.tol:
                if self.verbose:
                    print(f"  ✓ 收敛于第 {iteration} 次迭代")
                break

        # 计算最终 inertia
        final_dist_sq = euclidean_distances_sq(X, centers)
        inertia = float(final_dist_sq[np.arange(n), labels].sum())

        return centers, labels, inertia, iteration

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        拟合模型.

        Parameters
        ----------
        X : (n_samples, n_features) 训练数据.

        Returns
        -------
        self
        """
        X = self._check_data(X)
        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_n_iter = 0

        for run in range(self.n_init):
            if self.verbose:
                print(f"=== 第 {run + 1}/{self.n_init} 次运行 ===")

            centers, labels, inertia, n_iter = self._single_run(X, rng)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        if self.verbose:
            print(f"\n最终 inertia = {self.inertia_:.4f}, "
                  f"迭代次数 = {self.n_iter_}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新样本所属簇.

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        labels : (n_samples,) 每个样本的簇标签.
        """
        X = np.asarray(X, dtype=np.float64)
        dist_sq = euclidean_distances_sq(X, self.cluster_centers_)
        return np.argmin(dist_sq, axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将样本转换到距离空间: 每个样本到各质心的距离.

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        distances : (n_samples, n_clusters)
        """
        X = np.asarray(X, dtype=np.float64)
        return euclidean_distances(X, self.cluster_centers_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """拟合并返回标签."""
        self.fit(X)
        return self.labels_


# ---------------------------------------------------------------------------
#  Mini-Batch K-Means
# ---------------------------------------------------------------------------

class MiniBatchKMeans:
    """
    小批量 K-Means (Sculley, 2010).

    每次迭代只用一小批数据更新质心, 而不是全量数据.
    优势:
      - 速度比标准 K-Means 快很多 (尤其是 n > 10k)
      - 内存友好, 可以处理超大数据集
    劣势:
      - 结果略逊于标准 K-Means (惯性稍大)

    更新公式:
        对每个簇 c, 设本批次中属于该簇的样本集合为 C:
            n_c = n_c + |C|           (累计计数)
            μ_c = μ_c + Σ(x - μ_c) / n_c   (流式均值)

    Parameters
    ----------
    n_clusters : int
    batch_size : int
        每次迭代使用的样本数.
    max_iter : int
        最大迭代次数 (每次迭代处理一个 batch).
    tol : float
    random_state : int | None
    verbose : bool
    """

    def __init__(
        self,
        n_clusters: int = 8,
        batch_size: int = 1024,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray) -> "MiniBatchKMeans":
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        k = self.n_clusters
        rng = np.random.default_rng(self.random_state)

        # k-means++ 初始化
        centers = _init_kmeans_plus_plus(X, k, rng)
        counts = np.zeros(k, dtype=np.float64)   # 每个簇的累计样本数
        prev_centers = centers.copy()

        for iteration in range(1, self.max_iter + 1):
            # 采样一个小批量
            batch_idx = rng.choice(n, size=min(self.batch_size, n), replace=False)
            X_batch = X[batch_idx]

            # 分配
            dist_sq = euclidean_distances_sq(X_batch, centers)
            batch_labels = np.argmin(dist_sq, axis=1)

            # 更新质心 (流式均值)
            for c in range(k):
                mask = batch_labels == c
                count_c = mask.sum()
                if count_c == 0:
                    continue
                # 簇内样本相对于当前质心的偏移
                delta = X_batch[mask] - centers[c]
                centers[c] += delta.sum(axis=0) / (counts[c] + count_c)
                counts[c] += count_c

            # 收敛判断
            shift = np.sum((centers - prev_centers) ** 2)
            prev_centers[:] = centers

            if self.verbose and iteration % 10 == 0:
                print(f"  迭代 {iteration:3d} | 质心位移 {shift:.6f}")
            if shift <= self.tol:
                if self.verbose:
                    print(f"  ✓ 收敛于第 {iteration} 次迭代")
                break

        # 最终全量分配
        dist_sq = euclidean_distances_sq(X, centers)
        labels = np.argmin(dist_sq, axis=1)
        inertia = float(dist_sq[np.arange(n), labels].sum())

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iteration
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dist_sq = euclidean_distances_sq(X, self.cluster_centers_)
        return np.argmin(dist_sq, axis=1)


# ---------------------------------------------------------------------------
#  辅助工具函数
# ---------------------------------------------------------------------------

def elbow_method(
    X: np.ndarray,
    k_range: range | list[int],
    init: str = "k-means++",
    n_init: int = 10,
    random_state: int | None = None,
) -> list[float]:
    """
    肘部法则: 计算不同 k 值对应的 inertia, 帮助确定最佳簇数.

    使用方法: 画出 k vs inertia 的折线图, 找到"拐点" (肘部),
    拐点处的 k 通常是较好的选择.

    Parameters
    ----------
    X : 数据
    k_range : 要测试的 k 值范围, 例如 range(2, 11)

    Returns
    -------
    inertias : 各 k 值对应的 inertia 列表
    """
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, init=init, n_init=n_init,
                    random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"k={k:2d}  inertia={km.inertia_:.2f}")
    return inertias


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    轮廓系数 (Silhouette Coefficient).

    对每个样本 i:
        a(i) = i 到同簇其他样本的平均距离 (凝聚度, 越小越好)
        b(i) = i 到最近其他簇样本的平均距离 (分离度, 越大越好)
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    最终取所有样本 s(i) 的均值. 范围 [-1, 1]:
        +1  聚类很好
         0  簇有重叠
        -1  样本被分到了错误的簇

    Parameters
    ----------
    X : (n_samples, n_features)
    labels : (n_samples,) 簇标签

    Returns
    -------
    score : float  轮廓系数均值
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n = X.shape[0]
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k < 2:
        raise ValueError("轮廓系数要求至少有 2 个簇")
    if k == n:
        raise ValueError("每个样本一个簇, 无法计算轮廓系数")

    # 预计算距离矩阵
    dist_matrix = euclidean_distances(X, X)  # (n, n)

    silhouette_values = np.empty(n, dtype=np.float64)

    for i in range(n):
        own_label = labels[i]

        # a(i): 到同簇其他样本的平均距离
        same_mask = (labels == own_label)
        same_mask[i] = False          # 排除自身
        n_same = same_mask.sum()
        if n_same == 0:
            silhouette_values[i] = 0.0
            continue
        a_i = dist_matrix[i, same_mask].mean()

        # b(i): 到最近其他簇的平均距离
        b_i = np.inf
        for lbl in unique_labels:
            if lbl == own_label:
                continue
            other_mask = (labels == lbl)
            b_i = min(b_i, dist_matrix[i, other_mask].mean())

        silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)

    return float(silhouette_values.mean())


def make_blobs(
    n_samples: int = 300,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    random_state: int | None = None,
):
    """
    生成高斯分布的聚类测试数据.

    Parameters
    ----------
    n_samples : 总样本数
    n_features : 特征维度
    centers : 簇数
    cluster_std : 每个簇的标准差
    random_state : 随机种子

    Returns
    -------
    X : (n_samples, n_features)  样本
    y : (n_samples,)             真实标签
    """
    rng = np.random.default_rng(random_state)
    samples_per_center = n_samples // centers
    remainder = n_samples % centers

    X_parts = []
    y_parts = []
    for c in range(centers):
        count = samples_per_center + (1 if c < remainder else 0)
        center = rng.uniform(-10, 10, size=n_features)
        X_parts.append(rng.normal(loc=center, scale=cluster_std,
                                  size=(count, n_features)))
        y_parts.append(np.full(count, c, dtype=np.intp))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # 打乱顺序
    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


# ---------------------------------------------------------------------------
#  演示
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. 生成测试数据
    X, y_true = make_blobs(n_samples=2000, n_features=2, centers=4,
                           cluster_std=1.2, random_state=42)
    print(f"数据形状: {X.shape}, 真实簇数: {len(np.unique(y_true))}")

    # 2. 肘部法则选 k
    print("\n--- 肘部法则 ---")
    k_list = list(range(2, 9))
    inertias = elbow_method(X, k_list, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(k_list, inertias, "bo-")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True)

    # 3. 标准K-Means
    km = KMeans(n_clusters=4, init="k-means++", n_init=10,
                random_state=42, verbose=True)
    km.fit(X)
    print(f"\n标准 K-Means: inertia={km.inertia_:.2f}, 迭代={km.n_iter_}")

    axes[1].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="viridis",
                    s=8, alpha=0.6)
    axes[1].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                    c="red", marker="x", s=200, linewidths=3,
                    label="Centroids")
    axes[1].set_title(f"K-Means (inertia={km.inertia_:.1f})")
    axes[1].legend()

    # 4. 轮廓系数
    score = silhouette_score(X, km.labels_)
    print(f"轮廓系数: {score:.4f}")

    # 5. Mini-Batch K-Means 对比
    mbkm = MiniBatchKMeans(n_clusters=4, batch_size=256,
                           max_iter=300, random_state=42, verbose=False)
    mbkm.fit(X)
    print(f"\nMiniBatch K-Means: inertia={mbkm.inertia_:.2f}, "
          f"迭代={mbkm.n_iter_}")

    axes[2].scatter(X[:, 0], X[:, 1], c=mbkm.labels_, cmap="viridis",
                    s=8, alpha=0.6)
    axes[2].scatter(mbkm.cluster_centers_[:, 0],
                    mbkm.cluster_centers_[:, 1],
                    c="red", marker="x", s=200, linewidths=3,
                    label="Centroids")
    axes[2].set_title(f"MiniBatch K-Means (inertia={mbkm.inertia_:.1f})")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("kmeans_demo.png", dpi=150)
    print("\n可视化已保存到 kmeans_demo.png")
    plt.show()
