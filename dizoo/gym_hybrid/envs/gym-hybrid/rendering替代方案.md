# gym-hybrid 渲染替代方案

本方案旨在解决 `gym` 新版本中移除了 `gym.envs.classic_control.rendering` 模块后，`gym_hybrid` 环境无法调用 `render()` 的问题。下面总结了修改思路与关键实现。

## 需求分析

- 兼容 `gym` 新版本，无需依赖已被移除的 `rendering` 模块。
- 保留原有渲染能力：在窗口中实时展示环境；在需要时返回 `rgb_array`。
- 尽可能复用现有资源文件（背景、目标图标等）。

## 方案概述

1. **改用 OpenCV 渲染管线**：
   - 使用 `numpy` 构造画布，并借助 `cv2.circle`、`cv2.arrowedLine` 等 API 绘制目标、智能体与方向箭头。
   - 通过 `cv2.imshow` 展示画面，实现与原方案类似的可视化效果。
2. **保留 `rgb_array` 输出**：
   - 若调用 `env.render(mode="rgb_array")`，直接返回当前帧的 `numpy.ndarray`。
3. **窗口生命周期管理**：
   - 记录窗口名称，保证多次渲染时复用同一窗口。
   - 在 `env.close()` 中销毁窗口，避免资源泄露。
4. **资源兜底处理**：
   - 若背景或目标图片缺失，自动生成纯色背景，防止读取失败导致崩溃。

## 代码关键点

- 在 `BaseEnv.__init__` 中新增 `_window_name` 字段并加载背景图片：

  ```python
  self._window_name = None
  self.bg = cv2.imread(os.path.join(dirname, 'bg.jpg'))
  if self.bg is not None:
      self.bg = cv2.cvtColor(self.bg, cv2.COLOR_BGR2RGB)
      self.bg = cv2.resize(self.bg, (800, 800))
  else:
      self.bg = np.ones((800, 800, 3), dtype=np.uint8) * 255
  ```

- 在 `BaseEnv.render` 中：
  - 构造帧数据、绘制元素，并根据 `mode` 返回或展示画面。
  - 未支持的 `mode` 会抛出 `NotImplementedError`，方便调试。
- 在 `BaseEnv.close` 中使用 `cv2.destroyWindow` 清理窗口。

## 验证步骤

1. 激活 `pytorch-gym` Conda 环境。
2. 执行以下脚本，确认能够正常渲染并返回图像数组：

   ```powershell
   conda activate pytorch-gym
   python -c "import gym, gym_hybrid; env = gym.make('Sliding-v0'); env.reset(); frame = env.render(mode='rgb_array'); print('frame shape', frame.shape); env.close()"
   ```

3. 运行 `dizoo/gym_hybrid/tests/render.py` 观察窗口渲染效果。

## 后续可选优化

- ✅ **已完成：gymnasium 兼容性改进**
  - `reset()` 方法现在支持 `seed`、`options` 和 `return_info` 参数，并能够根据调用方式返回兼容格式
  - `step()` 方法现在返回 5 个值：`(observation, reward, terminated, truncated, info)`，符合 gymnasium 0.26+ 标准
  - 保持向后兼容：旧版 gym 调用 `env.reset()` 仍返回单个 observation，新版调用 `env.reset(return_info=True)` 返回 `(obs, info)` 元组
- 根据需要扩展 `HardMoveEnv` 的渲染逻辑，使其也具备图形化展示能力。
