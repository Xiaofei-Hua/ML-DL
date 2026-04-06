# Adam 优化器（Adaptive Moment Estimation）详细讲解

## 一、符号约定

| 符号 | 含义 | 代码对应 | 典型值 |
|------|------|----------|--------|
| $\theta$ | 模型参数（权重 W、偏置 b） | `params[k]` | — |
| $\alpha$ | 学习率（learning rate） | `self.lr` | 0.001 |
| $g_t$ | 第 $t$ 步的梯度 | `grads[k]` | — |
| $m_t$ | 梯度的一阶矩估计（均值） | `self.m[k]` | 初始为 0 |
| $v_t$ | 梯度的二阶矩估计（方差） | `self.v[k]` | 初始为 0 |
| $\beta_1$ | 一阶矩的衰减率 | `self.beta1` | 0.9 |
| $\beta_2$ | 二阶矩的衰减率 | `self.beta2` | 0.999 |
| $\hat{m}_t$ | 偏差修正后的一阶矩 | `m_hat` | — |
| $\hat{v}_t$ | 偏差修正后的二阶矩 | `v_hat` | — |
| $\varepsilon$ | 防止除零的极小常数 | `self.eps` | $10^{-8}$ |
| $t$ | 时间步（第几次更新） | `self.t` | 1, 2, 3, ... |

---

## 二、Adam 要解决什么问题？

回顾之前讲过的优化器：

| 优化器 | 问题 |
|--------|------|
| 普通 SGD | 所有参数用同一个学习率，不会自适应 |
| Momentum / Nesterov | 所有参数仍然用同一个学习率 |

**现实问题**：一个神经网络里有成千上万个参数，但不同参数的梯度大小差异很大。

- 有的参数梯度一直很小（比如浅层参数、不活跃的神经元） → 需要大学习率才能动
- 有的参数梯度一直很大（比如某些方向的梯度爆炸） → 需要小学习率才不会震荡

**Adam 的目标**：让每个参数**自动拥有独立的学习率**。梯度大的参数自动缩小学习率，梯度小的参数自动放大学习率。

---

## 三、前置知识：指数移动平均（EMA）

Adam 的核心操作是**指数移动平均**（Exponential Moving Average），在讲 Adam 公式之前先理解它。

### 什么是指数移动平均？

假设你每天记录温度：第 1 天 $x_1$，第 2 天 $x_2$，第 3 天 $x_3$，...

你想要一个"平滑"的温度值，不要剧烈波动。一种做法是指数移动平均：

$$S_1 = x_1$$
$$S_2 = \beta \cdot S_1 + (1-\beta) \cdot x_2$$
$$S_3 = \beta \cdot S_2 + (1-\beta) \cdot x_3$$
$$...$$

通用公式：

$$S_t = \beta \cdot S_{t-1} + (1-\beta) \cdot x_t$$

直觉：$S_t$ 是"历史累积"和"当前新值"的加权平均。

- $\beta$ 越大（如 0.99）→ 越重视历史，变化越平滑（记忆越长）
- $\beta$ 越小（如 0.9）→ 越重视当前，变化越快（记忆越短）

### 展开看看权重分布

把 $S_t$ 展开：

$$S_t = (1-\beta) \cdot x_t + (1-\beta) \cdot \beta \cdot x_{t-1} + (1-\beta) \cdot \beta^2 \cdot x_{t-2} + ... + (1-\beta) \cdot \beta^{t-1} \cdot x_1$$

每一项 $x_i$ 前面的系数是 $(1-\beta)\beta^{t-i}$，随着时间越久远，系数按 $\beta$ 的指数衰减。

例如 $\beta = 0.9$：
- 当前值 $x_t$ 权重：$1 - 0.9 = 0.1$
- 上一步 $x_{t-1}$ 权重：$0.1 \times 0.9 = 0.09$
- 上上步 $x_{t-2}$ 权重：$0.1 \times 0.81 = 0.081$
- ...以此类推

所有权重之和 $= 1$（所以叫"加权平均"）。

---

## 四、Adam 公式逐步推导

### 第 1 步：初始化

$$m_0 = 0, \quad v_0 = 0, \quad t = 0$$

$m$ 和 $v$ 初始化为 0。这在代码里是：

```python
self.m = {k: np.zeros_like(v) for k, v in params.items()}
self.v = {k: np.zeros_like(v) for k, v in params.items()}
self.t = 0
```

**注意这个"初始化为 0"的细节，后面会带来一个问题，在第 3 步解决。**

### 第 2 步：每一步更新 $m_t$ 和 $v_t$

在第 $t$ 步（$t$ 从 1 开始），拿到梯度 $g_t$ 后：

**更新一阶矩（梯度的 EMA，即梯度的"平均值方向"）**：

$$\boxed{m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t}$$

这就是对梯度做指数移动平均。$m_t$ 可以理解为"最近梯度的平均方向"。

直觉：$m_t$ 类似于 Momentum 中的速度 $v_t$，但衰减率不同（$\beta_1 = 0.9$ 对应记忆约 10 步）。

**更新二阶矩（梯度平方的 EMA，即梯度的"大小"）**：

$$\boxed{v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2}$$

注意这里是对 $g_t^2$（梯度的平方）做指数移动平均。$v_t$ 可以理解为"最近梯度大小的一种估计"。

直觉：$v_t$ 衡量梯度"有多活跃"。梯度一直很大的参数，$v_t$ 也会很大；梯度很小的参数，$v_t$ 也很小。

代码对应：

```python
self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g    # m_t
self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2 # v_t
```

### 第 3 步：偏差修正（Bias Correction）

这是 Adam 中最关键、也最容易忽略的步骤。

**问题出在哪？**

$m$ 和 $v$ 都初始化为 0，而不是真实的梯度值。在训练初期，EMA 的结果会被这个 0 严重拉低。

具体来看。展开 $m_t$：

$$m_1 = (1-\beta_1) g_1$$
$$m_2 = \beta_1(1-\beta_1) g_1 + (1-\beta_1) g_2$$
$$m_3 = \beta_1^2(1-\beta_1) g_1 + \beta_1(1-\beta_1) g_2 + (1-\beta_1) g_3$$

如果 $m$ 的初始值是真实的平均值（比如 $\approx 0$，因为梯度方向正负抵消），那没问题。但我们初始化成了精确的 0，而且 $\beta_1 = 0.9$，$\beta_2 = 0.999$。

**关键**：$m_t$ 的期望值是：

$$E[m_t] = (1 - \beta_1^t) \cdot E[g]$$

而不是 $E[g]$。也就是说 $m_t$ 偏小了 $\beta_1^t$ 倍。训练刚开始时 $t$ 很小，$\beta_1^t \approx 1$，所以 $m_t$ 会远小于真实梯度均值。

$v_t$ 同理，偏差更严重（$\beta_2 = 0.999$ 更接近 1）。

**修正方法**：除以 $(1 - \beta^t)$ 来补偿：

$$\boxed{\hat{m}_t = \frac{m_t}{1 - \beta_1^t}}$$

$$\boxed{\hat{v}_t = \frac{v_t}{1 - \beta_2^t}}$$

**验证修正是否正确**：

$$E[\hat{m}_t] = \frac{E[m_t]}{1 - \beta_1^t} = \frac{(1-\beta_1^t) E[g]}{1 - \beta_1^t} = E[g] \quad \checkmark$$

修正后的 $\hat{m}_t$ 的期望确实等于真实梯度均值。

**随着 $t$ 增大**：$\beta_1^t \to 0$，$1 - \beta_1^t \to 1$，修正项 $\to 1$，即训练后期修正不再需要。

举几个具体数字（$\beta_1 = 0.9$，$\beta_2 = 0.999$）：

| $t$ | $1 - \beta_1^t$ | $1 - \beta_2^t$ | 含义 |
|----|----------------|----------------|------|
| 1 | 0.1 | 0.001 | $m_1$ 被缩小 10 倍，$v_1$ 被缩小 1000 倍！ |
| 10 | 0.651 | 0.010 | 偏差仍然很大 |
| 100 | 0.99997 | 0.095 | $m$ 基本修正了，$v$ 仍有偏差 |
| 1000 | ≈ 1 | 0.632 | $v$ 才基本修正 |

$\beta_2 = 0.999$ 导致 $v$ 的修正非常慢，所以**偏差修正对 Adam 来说必不可少**。

代码对应：

```python
m_hat = self.m[k] / (1 - self.beta1 ** self.t)  # m̂_t
v_hat = self.v[k] / (1 - self.beta2 ** self.t)  # v̂_t
```

### 第 4 步：更新参数

$$\boxed{\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}}$$

逐项解释：

- **分子 $\hat{m}_t$**：梯度方向（梯度的平均方向），决定往哪个方向走
- **分母 $\sqrt{\hat{v}_t} + \varepsilon$**：自适应学习率的缩放因子
  - $\hat{v}_t$ 是梯度平方的 EMA，$\sqrt{\hat{v}_t}$ 类似于"梯度的标准差"
  - 梯度大的参数 → $\hat{v}_t$ 大 → 分母大 → 实际学习率变小（自动减速）
  - 梯度小的参数 → $\hat{v}_t$ 小 → 分母小 → 实际学习率变大（自动加速）
- **$\varepsilon$**：防止分母为 0（当梯度恰好为 0 时）

代码对应：

```python
params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

---

## 五、Adam 的组成部分来源

Adam 并非凭空发明，而是把两个已有优化器的思想结合在一起：

| 组成部分 | 来源 | Adam 中的对应 |
|----------|------|---------------|
| 对梯度做 EMA → 用方向更新 | **Momentum** | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ |
| 对梯度平方做 EMA → 自适应学习率 | **RMSProp** | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ |
| 初始化偏差修正 | **Adam 新增** | $\hat{m}_t = m_t / (1-\beta_1^t)$ |

所以 Adam = Momentum + RMSProp + Bias Correction

---

## 六、完整公式汇总（一次训练步）

给定第 $t$ 步的梯度 $g_t$：

$$\begin{cases} m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t & \text{（一阶矩：梯度方向）} \\ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 & \text{（二阶矩：梯度大小）} \\ \hat{m}_t = \dfrac{m_t}{1 - \beta_1^t} & \text{（偏差修正）} \\ \hat{v}_t = \dfrac{v_t}{1 - \beta_2^t} & \text{（偏差修正）} \\ \theta_{t+1} = \theta_t - \alpha \cdot \dfrac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} & \text{（参数更新）} \end{cases}$$

---

## 七、对应代码（MLP_plus.py）

```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr          # α
        self.beta1 = beta1    # β₁
        self.beta2 = beta2    # β₂
        self.eps = eps        # ε
        self.m = None         # m₀ = 0
        self.v = None         # v₀ = 0
        self.t = 0            # t = 0

    def step(self, params, grads):
        self.t += 1           # t → t+1

        if self.m is None:    # 首次调用时初始化
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g       # m_t
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2  # v_t

            m_hat = self.m[k] / (1 - self.beta1 ** self.t)                  # m̂_t
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)                  # v̂_t

            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)      # θ_{t+1}
```

---

## 八、与 Momentum / Nesterov 的对比

| 特性 | Momentum / Nesterov | Adam |
|------|---------------------|------|
| 学习率 | 所有参数共享同一个 $\alpha$ | 每个参数自动拥有独立的有效学习率 |
| 方向信息 | 用速度 $v_t$ 保留 | 用一阶矩 $m_t$ 保留（类似） |
| 自适应缩放 | 无 | 用二阶矩 $v_t$ 实现 |
| 需要调学习率 | 是，敏感 | $\alpha = 0.001$ 通常就能工作 |
| 偏差修正 | 无 | 需要（因为 $m, v$ 初始化为 0） |

---

## 九、为什么 Adam 默认学习率是 0.001？

和 Momentum/Nesterov 通常用 0.1 不同，Adam 默认学习率 $\alpha = 0.001$，差了 100 倍。

原因是分母 $\sqrt{\hat{v}_t}$ 的存在。对于一个梯度量级约为 1 的参数：

- 分子 $\hat{m}_t \approx 1$
- 分母 $\sqrt{\hat{v}_t} \approx \sqrt{1} = 1$
- 实际更新量 $\approx \alpha \cdot 1/1 = \alpha$

但 Momentum 的更新量是 $\alpha \cdot v_t$，而 $v_t$ 可以累积到 10 甚至更大（多步梯度叠加）。

所以 Adam 的有效更新幅度比 Momentum 大很多倍，需要更小的 $\alpha$ 来补偿。
