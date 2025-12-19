# torch.nn.init.orthogonal_ 是什么初始化?
`torch.nn.init.orthogonal_` 是 **正交初始化（Orthogonal Initialization）**，它将权重矩阵初始化为一个**正交矩阵**。

---

## 1) 什么是正交矩阵？

正交矩阵满足：**Q^T @ Q = I**（转置乘以自身等于单位矩阵）

这意味着矩阵的**行向量/列向量彼此正交且模长为 1**。

---

## 2) 正交初始化的核心作用

### (A) 保持梯度范数稳定（缓解梯度消失/爆炸）

在深度网络中，前向传播时信号会经过多层线性变换：

- 如果权重矩阵的**奇异值**偏离 1 太多，信号会指数级放大/缩小
- 正交矩阵的**所有奇异值都是 1**，保证前向传播时**不改变信号的 L2 范数**

同理，反向传播时梯度也不会因为权重矩阵而被放大/缩小，有助于训练稳定。

### (B) 避免神经元"同质化"

随机初始化可能让多个神经元学到相似的特征（冗余），正交初始化强制让初始权重在不同方向上分散，增加表达能力。

---

## 3) 在 RL 中的常见用法（如这段代码）

```python
for m in self._model.modules():
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(m.weight)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
```

这段代码对**所有卷积层和全连接层**的权重做正交初始化。

### 为什么 PPO/A2C 等 on-policy 算法常用正交初始化？

1. **训练稳定性更重要**：on-policy 算法对初始化敏感，正交初始化能让训练早期更稳定
2. **配合 Value/Policy 双头网络**：RL 中常有 actor-critic 结构，正交初始化能让两个头部初始时不互相干扰
3. **OpenAI Baselines 的实践**：OpenAI 在 PPO/A2C 实现中广泛使用正交初始化，效果较好，成为 RL 社区的"约定俗成"

---

## 4) `orthogonal_` 的参数

```python
torch.nn.init.orthogonal_(tensor, gain=1.0)
```

- **`gain`**：缩放因子，默认为 1.0
  - 例如 `gain=np.sqrt(2)` 常用于 ReLU 激活函数后的层（配合 He 初始化的思想）
  - 在 RL 中，PPO 的 actor 输出层有时会用 `gain=0.01`，让初始策略接近均匀分布

---

## 5) 和其他初始化的对比

| 初始化方法 | 适用场景 | 特点 |
|-----------|---------|------|
| `xavier_uniform_` | 传统监督学习（tanh/sigmoid） | 保持方差稳定 |
| `kaiming_normal_` (He) | ReLU 激活的深度网络 | 考虑 ReLU 的非线性 |
| `orthogonal_` | RL、RNN、深度网络 | 保持范数，避免梯度问题 |
| `uniform_/normal_` | 简单网络 | 最基础的随机初始化 |

---

## 6) 在这段代码中的效果

在 `OffPPOCollectTrajPolicy._init_learn()` 中：

```python
for m in self._model.modules():
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(m.weight)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
```

- 对所有 Conv2d 和 Linear 层的权重做正交初始化
- **不包括 bias**（bias 通常初始化为 0 或小随机值）
- 这让 PPO 训练早期更稳定，尤其是在 Atari 这类高维视觉输入任务中

---

## 总结

**正交初始化**让权重矩阵在前向/反向传播时**不改变信号范数**，是 RL 领域（尤其是 PPO/A2C）的常用技巧，能显著提升训练稳定性和收敛速度。