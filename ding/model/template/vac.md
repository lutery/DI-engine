# `multi_head` 判断逻辑详解

## 代码分析

```python
multi_head = not isinstance(action_shape, int)
```

这行代码在判断 **是否需要使用多头输出结构**，关键在于 `action_shape` 的类型差异。

## 两种情况对比

### ✅ 情况 1: `action_shape` 是 `int` 类型

```python
action_shape = 6  # 单一动作空间，6个离散动作

multi_head = not isinstance(6, int)  # False
# multi_head = False → 使用单头输出
```

**含义**: 
- **单一离散动作空间**
- 智能体每次只输出 **一个动作**，从 6 个候选动作中选择
- 例如: Atari Pong 游戏 (上/下/左/右/开火/不动)

**网络结构**:
```python
self.actor_head = DiscreteHead(
    actor_head_hidden_size,  # 输入特征维度
    action_shape=6,          # 输出 6 维 logits
    ...
)

# 输出形状: (batch_size, 6)
# 例如: torch.Size([32, 6])
```

---

### ✅ 情况 2: `action_shape` 是 **列表/元组** 类型

```python
action_shape = [3, 3]  # 多个动作空间

multi_head = not isinstance([3, 3], int)  # True
# multi_head = True → 使用多头输出
```

**含义**:
- **多个独立的离散动作空间**
- 智能体需要 **同时输出多个动作**，每个动作空间独立选择
- 例如: 多关节机器人控制，每个关节有独立的离散控制选项

**网络结构**:
```python
self.actor_head = MultiHead(
    DiscreteHead,
    actor_head_hidden_size,
    action_shape=[3, 3],  # 两个独立的动作头
    ...
)

# 输出形状: [(batch_size, 3), (batch_size, 3)]
# 第一个头: 从 3 个动作中选择
# 第二个头: 从 3 个动作中选择
```

---

## 具体示例对比

### 示例 1: 单头 - Atari Pong

```python
# 配置
action_shape = 6  # 6个离散动作: [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]

# 网络结构
model = VAC(
    obs_shape=[4, 84, 84],
    action_shape=6,
    action_space='discrete'
)

# multi_head = False
# 输出 logits: torch.Size([batch, 6])
# 采样: 从 6 个动作中选 1 个

logits = model.compute_actor(obs)['logit']
# logits.shape = (32, 6)
action = torch.argmax(logits, dim=-1)  # 选择概率最大的动作
# action.shape = (32,)  例如: [2, 0, 5, 3, ...]
```

---

### 示例 2: 多头 - 星际争霸 II (SMAC)

```python
# 配置 - 控制多个单位，每个单位有独立的动作空间
action_shape = [5, 5, 5]  # 3个单位，每个单位有5个动作选项

# 网络结构
model = VAC(
    obs_shape=observation_dim,
    action_shape=[5, 5, 5],
    action_space='discrete'
)

# multi_head = True
# 输出 logits: [torch.Size([batch, 5]), torch.Size([batch, 5]), torch.Size([batch, 5])]

logits = model.compute_actor(obs)['logit']
# logits 是一个列表: [logit_unit1, logit_unit2, logit_unit3]
# logit_unit1.shape = (32, 5)
# logit_unit2.shape = (32, 5)
# logit_unit3.shape = (32, 5)

actions = [torch.argmax(logit, dim=-1) for logit in logits]
# actions = [unit1_action, unit2_action, unit3_action]
# 每个 action.shape = (32,)
```

---

### 示例 3: 多头 - 机器人多关节控制

```python
# 配置 - 机器人有 3 个关节，每个关节有不同数量的离散档位
action_shape = [3, 5, 7]  
# 关节1: 3个档位 (左/中/右)
# 关节2: 5个档位 (5种角度)
# 关节3: 7个档位 (7种力度)

model = VAC(
    obs_shape=robot_state_dim,
    action_shape=[3, 5, 7],
    action_space='discrete'
)

# multi_head = True
# 每个关节独立输出一个动作
logits = model.compute_actor(obs)['logit']
# logits = [
#     torch.Size([32, 3]),  # 关节1的 logits
#     torch.Size([32, 5]),  # 关节2的 logits
#     torch.Size([32, 7])   # 关节3的 logits
# ]
```

---

## 源码实现细节

### 单头实现 (`DiscreteHead`)

```python
# ding/torch_utils/network/discrete.py
class DiscreteHead(nn.Module):
    def __init__(self, hidden_size, output_size, layer_num, ...):
        super().__init__()
        # 简单的全连接层
        self.main = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)  # 输出维度 = output_size
        )
    
    def forward(self, x):
        return {'logit': self.main(x)}  # (batch, output_size)
```

---

### 多头实现 (`MultiHead`)

```python
# ding/torch_utils/network/multi_head.py
class MultiHead(nn.Module):
    def __init__(self, head_cls, hidden_size, output_size_list, ...):
        super().__init__()
        # 为每个动作空间创建独立的头
        self.heads = nn.ModuleList([
            head_cls(hidden_size, output_size, ...)
            for output_size in output_size_list
        ])
    
    def forward(self, x):
        # 返回多个 logits 的列表
        return {'logit': [head(x)['logit'] for head in self.heads]}
        # 例如: {'logit': [logit1, logit2, logit3]}
```

---

## 实际应用场景

### 单头 (Single Head) 适用场景

| 环境 | `action_shape` | 说明 |
|------|----------------|------|
| **CartPole** | `2` | 左/右 |
| **Atari Pong** | `6` | 6个离散动作 |
| **LunarLander** | `4` | 4个推进器控制 |
| **简单导航** | `4` | 上/下/左/右 |

---

### 多头 (Multi Head) 适用场景

| 环境 | `action_shape` | 说明 |
|------|----------------|------|
| **SMAC (星际II)** | `[n_actions] * n_agents` | 每个单位独立动作 |
| **多关节机器人** | `[3, 5, 7]` | 每个关节独立控制 |
| **多任务环境** | `[4, 6, 3]` | 同时执行多个子任务 |
| **GoBigger** | `[action_dim] * n_balls` | 控制多个球 |

---

## 完整的判断流程

```python
def __init__(self, action_shape, ...):
    # 压缩 action_shape (去除多余的维度)
    action_shape = squeeze(action_shape)
    # squeeze([6]) → 6
    # squeeze([3, 3]) → [3, 3]
    
    if self.action_space == 'discrete':
        # 判断是否需要多头
        multi_head = not isinstance(action_shape, int)
        
        if multi_head:
            # 场景: action_shape = [3, 5, 7]
            self.actor_head = MultiHead(
                DiscreteHead,
                actor_head_hidden_size,
                action_shape,  # 传入列表
                ...
            )
            # 输出: [logit1, logit2, logit3]
        else:
            # 场景: action_shape = 6
            self.actor_head = DiscreteHead(
                actor_head_hidden_size,
                action_shape,  # 传入整数
                ...
            )
            # 输出: logit (单个张量)
```

---

## 总结

| 特性 | `action_shape = int` | `action_shape = list` |
|------|---------------------|----------------------|
| **含义** | 单一动作空间 | 多个独立动作空间 |
| **multi_head** | `False` | `True` |
| **网络结构** | 单个 `DiscreteHead` | 多个 `DiscreteHead` (通过 `MultiHead` 管理) |
| **输出形状** | `(batch, action_dim)` | `[(batch, dim1), (batch, dim2), ...]` |
| **采样方式** | 选择 1 个动作 | 每个头独立选择 1 个动作 |
| **典型场景** | Atari, CartPole | SMAC, 多关节控制 |

**核心区别**: 
- `int` 类型 → **从一组动作中选一个**
- `list/tuple` 类型 → **同时从多组动作中各选一个**

这种设计让 VAC 模型能够灵活适配不同复杂度的离散动作空间！


