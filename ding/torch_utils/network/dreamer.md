# 什么是：输出的空间尺寸（H/W）按 TF/JAX 的 SAME 规则

TF/JAX 里的 **SAME padding** 规则，指的是：**卷积会自动在输入四周补零（padding），使得输出的空间尺寸（H/W）满足一个固定的“目标尺寸”公式**，而不是像 PyTorch 传统那样你手动给一个固定的 `padding` 数值。

---

## 1) SAME 规则到底在“规定什么输出尺寸”？

对每个空间维度（H 或 W）分别成立：

- **输出大小 = ⌈输入大小 / stride⌉**

也就是：

- `H_out = ceil(H_in / sH)`
- `W_out = ceil(W_in / sW)`

这就是我说的“输出的空间尺寸（H/W）按 TF/JAX 的 SAME 规则变化”。

### 常见直觉
- **stride=1**：`ceil(H/1)=H` → 输出高宽与输入相同（不变形）
- **stride=2**：输出大约减半，但如果输入是奇数会“向上取整”
  - 例如 `H_in=63` → `H_out=ceil(63/2)=32`

---

## 2) 为什么需要 padding 才能满足这个规则？

如果不 padding（TF 叫 `VALID`），卷积会“越卷越小”。  
SAME 的目标是：**即使 stride>1，也要让输出严格符合 `ceil(in/stride)`**，所以必须补足边界，让最后一格也能被卷积覆盖到。

因此 SAME 的本质是：**先算出为了达成目标 `ceil(in/stride)`，需要补多少零（而且可能是非对称的），再去做卷积。**

---

## 3) SAME 的 padding 是怎么计算出来的（和 `Conv2dSame` 一致）

对某一维（以 H 为例）：

1. 先定目标输出：
   - `H_out = ceil(H_in / sH)`
2. 反推需要的总 padding（上下合计）：
   - `pad_h = max((H_out - 1) * sH + (kH - 1) * dH + 1 - H_in, 0)`

然后把 `pad_h` 拆成：
- top = `pad_h // 2`
- bottom = `pad_h - pad_h // 2`

> 注意：当 `pad_h` 是奇数时，上下就会 **不对称**（例如 top=0, bottom=1）。这也是很多情况下无法用 `nn.Conv2d(padding=...)` 的对称 padding 精准复刻 SAME 的原因之一。

---

## 4) 一个具体例子（看出 “ceil” 的意义）

设 `k=3, s=2, d=1`

- 输入 `H_in=63`
- 目标 `H_out = ceil(63/2)=32`

反推总 padding：
- `pad_h = (32-1)*2 + (3-1)*1 + 1 - 63 = 62 + 2 + 1 - 63 = 2`

拆分：top=1 bottom=1（对称）

如果输入换成 `H_in=64`：
- `H_out=ceil(64/2)=32`
- `pad_h = (32-1)*2 + 3 - 64 = 62 + 3 - 64 = 1`
- 拆分：top=0 bottom=1（**非对称**）

这就体现了 SAME 的两个关键点：
- 输出尺寸由 `ceil(in/stride)` 决定（固定规则）
- padding 会随输入尺寸变化，且可能非对称

---

如果你愿意，我也可以用你项目里 DreamerV3 encoder 的实际卷积配置（每层的 kernel/stride/channel）把从 `84×84` 或 `64×64` 开始的每层输出 `H×W` 列成表。