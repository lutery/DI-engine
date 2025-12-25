# #codebase  如果从 dizoo\box2d\lunarlander\config\lunarlander_r2d3_ppoexpert_config.py 作为入口，那么cfg.policy.collect.collector是啥？

从 lunarlander_r2d3_ppoexpert_config.py 作为入口时，`cfg.policy.collect.collector` **不是你在 main_config 里显式写出来的字段**，而是由 `compile_config(auto=True, create_cfg=...)` **自动补全并编译出来的**。

## 1) 它最终是什么？
最终 `cfg.policy.collect.collector` 会是一个 **type 为 `'sample'` 的串行采集器配置**，对应 Registry 里的 `SampleSerialCollector`。

其默认内容来自 `SampleSerialCollector.config`：

- `type`: `'sample'`
- `deepcopy_obs`: `False`
- `transform_obs`: `False`
- `collect_print_freq`: `100`
- 以及 `cfg_type`（由基类 `default_config` 自动加上，通常是 `SampleSerialCollectorDict`）

也就是说它大致等价于（示意）：
````python
# 编译后的大致结构（示意）
EasyDict({
  'type': 'sample',
  'deepcopy_obs': False,
  'transform_obs': False,
  'collect_print_freq': 100,
  'cfg_type': 'SampleSerialCollectorDict',
})
````

## 2) 为什么会变成 sample？
因为你的 `create_config` 里**没有**提供 `collector` 字段（只提供了 env/env_manager/policy），于是 `compile_config` 在 `auto=True` 时会走兼容逻辑：

- 若 `create_cfg` 没有 `collector`：默认补上 `create_cfg.collector = {'type': 'sample'}`  
  位置：`ding.config.compile_config`
- 之后会调用 `compile_collector_config`：
  - 用 `type='sample'` 找到 collector class（Registry）：`get_serial_collector_cls`
  - 合并 `SampleSerialCollector.default_config()` + create_cfg/user_cfg 覆盖项

而 `create_serial_collector(cfg.policy.collect.collector, ...)` 会按该配置创建实例：`create_serial_collector`。

如果你希望它不是 `sample`，需要在 `create_config` 里显式加上 `collector=dict(type='episode')` 或其他已注册的 collector 类型。