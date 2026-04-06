# Nesterov 动量（Nesterov Accelerated Gradient）详细讲解

## 一、符号约定

| 符号 | 含义 | 代码对应 |
|------|------|----------|
| $\theta$ | 模型参数（权重 W、偏置 b） | `params[k]` |
| $\alpha$ | 学习率（learning rate） | `self.lr` |
| $\mu$ | 动量系数（momentum），通常取 0.9 | `self.momentum` |
| $v$ | 速度（velocity），梯度的累积 | `self.v[k]` |
| $g$ / $\nabla L$ | 损失函数对参数的梯度 | `grads[k]` |
| $L$ | 损失函数（如交叉熵） | `compute_loss()` |
| $t$ | 当前训练步数 | 每次 `step()` 调用 |

> 注：$g$ 和 $\nabla L$ 是同一个东西，只是写法不同——$g$ 是代码里的变量名，$\nabla L$ 是数学符号。

---

## 二、背景：普通 SGD

最基础的梯度下降，每一步只看当前位置的梯度方向走：

$$\theta_{t+1} = \theta_t - \alpha \cdot g_t$$

问题：下山时每一步都独立地看脚下，步伐慢、容易在山谷中来回震荡。

---

## 三、Momentum SGD：引入惯性

引入"速度" $v$，记录历史梯度的累加。像球从山坡滚下来，越滚越快：

$$v_t = \mu \cdot v_{t-1} + g_t$$

$$\theta_{t+1} = \theta_t - \alpha \cdot v_t$$

直觉：
- $\mu \cdot v_{t-1}$：旧速度的惯性（保留过去的信息）
- $g_t$：当前梯度（加入新信息）
- 更新量 $v_t$ = 惯性 + 新梯度

### Momentum 的问题

Momentum 的更新方向是"旧惯性"加"当前位置的梯度"。但它不知道**如果按惯性再往前多走一步，梯度会变成什么样**。

就像闭着眼靠惯性滑行——你可能已经滑过头了还不知道。

---

## 四、Nesterov 的核心思想：先看一步，再决定

Nesterov 说：**在计算梯度之前，先按惯性往前跳一步，然后在新位置算梯度。**

这样用的是"未来位置"的梯度信息，而不是"当前位置"的。能更早发现前方是不是该减速了。

### 4.1 理论公式（原始写法）

分两步完成：

$$\theta_{lookahead} = \theta_t - \alpha \cdot \mu \cdot v_{t-1} \quad \text{（先按惯性往前看一步）}$$

$$v_t = \mu \cdot v_{t-1} + \nabla L(\theta_{lookahead}) \quad \text{（在未来位置算梯度）}$$

$$\theta_{t+1} = \theta_t - \alpha \cdot v_t \quad \text{（更新参数）}$$

**问题**：要在 $\theta_{lookahead}$ 处做一次完整的前向传播来计算梯度，**计算量翻倍**。

### 4.2 实用 Nesterov 公式是如何得到的

上面原始写法需要在 $\theta_{lookahead}$ 处多做一次前向传播，计算量翻倍。实际代码中广泛使用的是一个**不增加额外计算量的实用公式**：

$$v_t = \mu \cdot v_{t-1} + g_t$$

$$\boxed{\theta_{t+1} = \theta_t - \alpha \cdot (\mu \cdot v_t + g_t)}$$

> **注意**：这个实用公式**不是**通过将 $\mu \cdot v_{t-1} + g_t$ 替换为 $v_t$ 从原始公式推导出来的。如果做这个替换，你会得到 $\theta_{t+1} = \theta_t - \alpha \cdot v_t$，那恰好就是普通 Momentum，不是 Nesterov。

#### 推理过程（启发式，非严格数学推导）

既然无法在不增加计算量的情况下精确实现"在未来位置算梯度"，那能不能**换一种方式模拟同样的效果**？

核心观察：在普通 Momentum 中，更新量是 $v_t = \mu v_{t-1} + g_t$，其中新梯度的贡献权重是 $1$。如果我们想让新梯度"更有话语权"（模拟"往前看一步后更信任新梯度"的效果），可以这样做：

$$\text{更新量} = \underbrace{\mu \cdot v_t}_{\text{额外推一把惯性}} + \underbrace{g_t}_{\text{当前梯度}}$$

展开后：

$$= \mu(\mu v_{t-1} + g_t) + g_t = \mu^2 v_{t-1} + (1+\mu)g_t$$

对比普通 Momentum：

$$\text{更新量} = v_t = \mu v_{t-1} + g_t$$

**效果完全符合 Nesterov 的精神**：

- 旧惯性权重从 $\mu$ 降到 $\mu^2$（0.9 → 0.81）——削弱过去
- 新梯度权重从 $1$ 升到 $1+\mu$（1 → 1.9）——信任现在

**推理链条总结**：

```
原始 Nesterov 的精神："先看一步，更信任新梯度"
        ↓
但实现需要额外一次前向传播，代价太大
        ↓
能不能不增加计算量，但达到类似效果？
        ↓
给新梯度更大的权重，给旧惯性更小的权重
        ↓
θ -= α · (μ·v_t + g_t)   ← 这个公式做到了，而且没有增加计算量
```

因此，这个实用公式是一个**设计出来的近似公式**，不是从原始 Nesterov 严格推导出来的。它之所以被广泛使用（包括 PyTorch、TensorFlow 等框架），是因为：

1. 计算量和普通 Momentum 完全一样
2. 实验效果接近甚至等同于真正的 Nesterov

对比普通 Momentum：

| 优化器 | 速度公式 | 参数更新 |
|--------|----------|----------|
| Momentum | $v_t = \mu \cdot v_{t-1} + g_t$ | $\theta_{t+1} = \theta_t - \alpha \cdot v_t$ |
| Nesterov | $v_t = \mu \cdot v_{t-1} + g_t$（一样） | $\theta_{t+1} = \theta_t - \alpha \cdot (\mu \cdot v_t + g_t)$ |

速度公式完全一样，**唯一区别在参数更新那一行**。

---

## 五、直觉理解区别

把更新量展开，对比新旧梯度和旧惯性各自的贡献：

### 普通 Momentum

更新量 $= v_t = \mu \cdot v_{t-1} + g_t$

- 新梯度 $g_t$ 贡献权重 = **1**
- 旧惯性 $v_{t-1}$ 贡献权重 = **$\mu$** ≈ 0.9

### Nesterov

更新量 $= \mu \cdot v_t + g_t = \mu \cdot (\mu \cdot v_{t-1} + g_t) + g_t = \mu^2 \cdot v_{t-1} + \mu \cdot g_t + g_t$

- 新梯度 $g_t$ 贡献权重 = **$1 + \mu$** ≈ 1.9
- 旧惯性 $v_{t-1}$ 贡献权重 = **$\mu^2$** ≈ 0.81

### 结论

> Nesterov 让"新梯度"的权重更大（1 → 1.9），让"旧惯性"的权重更小（0.9 → 0.81）。
>
> 相当于说：**我更信任眼前的梯度，稍微削弱过去的惯性。** 如果前方梯度方向变了（比如接近谷底），Nesterov 能更快反应过来，减少震荡。

---

## 六、对应代码（MLP_plus.py）

```python
# MLP_plus.py 第 52-60 行
for k in params:
    g = grads[k]                                        # g 就是 ∇L
    self.v[k] = self.momentum * self.v[k] + g           # v_t = μ·v_{t-1} + g_t（两者一样）

    if self.nesterov:
        # Nesterov: θ -= α · (μ·v_t + g_t)
        params[k] -= self.lr * (self.momentum * self.v[k] + g)
    else:
        # 普通 Momentum: θ -= α · v_t
        params[k] -= self.lr * self.v[k]
```

关键区别就一行：

- **普通 Momentum**：`params[k] -= lr * v[k]`
- **Nesterov**：`params[k] -= lr * (momentum * v[k] + g)`

Nesterov 在更新时多加了一项 `momentum * v[k]`，相当于额外推进了一步惯性，使得新梯度 `g` 的声音更响。

---

## 七、总结

Nesterov 动量的优势：

1. **几乎不增加额外计算开销**——和普通 Momentum 的计算量完全一样（速度公式相同）
2. **能更快响应梯度方向的变化**——新梯度权重更大（$1+\mu$ vs 1）
3. **接近最优解时更有效地"刹车"**——旧惯性被削弱（$\mu^2$ vs $\mu$），减少在谷底附近的震荡
