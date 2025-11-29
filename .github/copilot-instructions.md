# DI-engine 代码库 AI 指南

## 项目概览

DI-engine 是一个通用的 PyTorch/JAX 决策智能引擎,提供模块化的深度强化学习(DRL)算法实现。核心设计理念:**Python优先 + 异步原生 + 模块化抽象**。

### 核心架构组件

三大核心抽象: **Env**(环境), **Policy**(策略), **Model**(模型)

```
ding/
├── entry/          # 训练入口点: serial_entry, parallel_entry, serial_entry_onpolicy 等
├── policy/         # 策略实现: DQN, PPO, SAC, R2D2, QMIX 等 70+ 算法
├── model/          # 神经网络模型: template/, common/, wrapper/
├── envs/           # 环境管理器和包装器
├── worker/         # Collector(采集), Learner(学习), Evaluator(评估)
├── config/         # 配置系统: compile_config, read_config
├── utils/          # Registry 注册系统, 工具函数
└── data/           # 数据缓冲区和存储

dizoo/              # 环境示例和配置(Atari, Mujoco, SMAC 等 30+ 环境)
```

## Registry 注册系统 (关键模式)

**所有组件通过 Registry 模式注册和创建** - 这是代码库的核心模式:

```python
# 注册: 使用装饰器
@POLICY_REGISTRY.register('dqn')
class DQNPolicy(Policy):
    pass

# 创建: 通过配置字典
policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
```

**主要 Registry**:
- `POLICY_REGISTRY` - 策略算法
- `MODEL_REGISTRY` - 神经网络模型  
- `ENV_REGISTRY` - 环境类型
- `SERIAL_COLLECTOR_REGISTRY`, `LEARNER_REGISTRY`, `BUFFER_REGISTRY` 等

位置: `ding/utils/registry_factory.py`

## 配置驱动开发

### 配置结构
每个实验需要两个配置:
1. **main_config**: 超参数(学习率, batch size, 环境数量等)
2. **create_config**: 组件类型(policy type, env type, env_manager type)

```python
# 示例: dizoo/atari/config/serial/pong/pong_dqn_config.py
main_config = dict(
    env=dict(collector_env_num=8, env_id='PongNoFrameskip-v4'),
    policy=dict(type='dqn', learning_rate=0.0001, batch_size=32),
)
create_config = dict(
    env=dict(type='atari', import_names=['dizoo.atari.envs.atari_env']),
    policy=dict(type='dqn'),
)
```

### 配置编译
`compile_config()` 自动填充默认值、验证配置、设置随机种子:
```python
cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg)
```

## Entry Points (训练入口)

**选择正确的 entry**:
- `serial_entry.py` - 标准 off-policy 算法 (DQN, SAC, TD3)
- `serial_entry_onpolicy.py` - on-policy 算法 (PPO, A2C, MAPPO)
- `parallel_entry.py` - 分布式训练
- `serial_entry_offline.py` - 离线 RL (CQL, BCQ)
- `serial_entry_mbrl.py` - 基于模型的 RL

**标准流程**:
```python
# 1. 读取配置
cfg, create_cfg = read_config(config_path)

# 2. 编译配置  
cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

# 3. 创建环境
env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])

# 4. 创建策略
policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

# 5. 创建 worker 组件
learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode)
collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode)
evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode)
```

## Policy 开发规范

**Policy 三种模式**: `learn_mode`, `collect_mode`, `eval_mode`

```python
class CustomPolicy(Policy):
    config = dict(...)  # 默认配置
    
    def default_model(self) -> Tuple[str, List[str]]:
        return 'model_name', ['import.path']  # 返回模型注册名和导入路径
    
    def _init_learn(self):   # 学习模式初始化
        self._learn_model = self._model
        self._optimizer = torch.optim.Adam(self._model.parameters())
    
    def _forward_learn(self, data: dict) -> Dict[str, Any]:  # 学习前向传播
        # 核心学习逻辑
        return {'total_loss': loss, 'cur_lr': lr}
    
    def _init_collect(self):  # 采集模式初始化
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
    
    def _forward_collect(self, data: dict) -> dict:  # 采集前向传播
        # 环境交互逻辑
        return output
    
    def _init_eval(self):    # 评估模式初始化
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
    
    def _forward_eval(self, data: dict) -> dict:  # 评估前向传播
        return output
```

**注册**: `@POLICY_REGISTRY.register('custom_policy')`

## 模型系统

**模型在 Registry 中注册并由 Policy 自动加载**:

```python
@MODEL_REGISTRY.register('dqn')
class DQN(nn.Module):
    def forward(self, x):
        # 必须返回字典: {'logit': output} 或 {'q_value': output}
        return {'logit': self.head(x)}
```

**模型包装器** (`ding/model/wrapper/`): 为模型添加功能
- `model_wrap(model, 'eps_greedy_sample')` - epsilon-greedy 采样
- `model_wrap(model, 'argmax_sample')` - argmax 动作选择
- `model_wrap(model, 'hidden_state')` - RNN 隐藏状态管理
- `model_wrap(model, 'transformer_memory')` - Transformer 记忆管理

## 测试和构建

```bash
# 单元测试
make unittest                    # 运行所有单元测试
make unittest RANGE_DIR=ding/policy  # 测试特定目录
pytest ding/policy/tests/ -sv    # 直接 pytest

# 算法测试 (完整训练验证)
make algotest

# 代码格式化
yapf -ir ding/                   # 格式化代码 (yapf==0.29.0)
bash format.sh                   # 使用格式化脚本

# 覆盖率
make unittest  # 生成 coverage.xml
```

**pytest 标记**: `@pytest.mark.unittest`, `@pytest.mark.algotest`, `@pytest.mark.envtest`

## 环境集成

**添加新环境**:
1. 继承 `BaseEnv` (位于 `ding/envs/env/base_env.py`)
2. 实现: `reset()`, `step(action)`, `close()`, `seed(seed)`, `info()` 
3. 注册: `@ENV_REGISTRY.register('env_name')`
4. 在 `dizoo/` 下创建环境目录和配置

```python
@ENV_REGISTRY.register('custom_env')
class CustomEnv(BaseEnv):
    def __init__(self, cfg):
        self._cfg = cfg
        self._init_flag = False
    
    def reset(self):
        obs = ...  # 重置逻辑
        return obs
    
    def step(self, action):
        obs, reward, done, info = ...
        return BaseEnvTimestep(obs, reward, done, info)
```

**在 create_config 中指定**:
```python
create_config = dict(
    env=dict(type='custom_env', import_names=['path.to.custom_env']),
)
```

## 常见陷阱

1. **环境数量不匹配**: `cfg.env.collector_env_num` 必须等于 `cfg.policy.collect.env_num` (特别是 R2D2, NGU 等 RNN 策略)

2. **模型输出格式**: 模型必须返回字典, 键名要匹配 policy 期望 (如 `'logit'`, `'q_value'`, `'action'`)

3. **_command 后缀**: Entry 会自动添加 `'_command'` 到 policy type: `create_cfg.policy.type = create_cfg.policy.type + '_command'`

4. **EasyDict 使用**: 配置使用 `EasyDict` 允许点号访问: `cfg.policy.learn.learning_rate`

5. **Registry 导入**: 在注册前必须导入模块, 否则 `create_policy()` 会失败

6. **环境跟踪**: 设置 `export DIENGINEREGTRACE=ON` 可追踪注册位置以便调试

## 代码风格

- 遵循 PEP 8, 使用 `yapf==0.29.0` 格式化
- 类型提示: 函数参数和返回值使用类型注解
- 文档字符串: 使用 Overview, Arguments, Returns 结构
- 中文注释: 代码库广泛使用中文注释 (如 `registry.py`)

## 已支持的强化学习算法

### 基础 DRL 算法 (Value-Based)

**DQN 系列** - 离散动作空间:
- `DQN` - Deep Q-Network (`policy/dqn.py`)
- `C51` - Categorical DQN (`policy/c51.py`)
- `QRDQN` - Quantile Regression DQN (`policy/qrdqn.py`)
- `IQN` - Implicit Quantile Networks (`policy/iqn.py`)
- `FQF` - Fully parameterized Quantile Function (`policy/fqf.py`)
- `Rainbow` - 集成多种 DQN 改进 (`policy/rainbow.py`)
- `R2D2` - Recurrent Experience Replay in DQN (`policy/r2d2.py`)
- `R2D2-GTrXL` - R2D2 with Gated Transformer-XL (`policy/r2d2_gtrxl.py`)
- `MDQN` - Munchausen DQN (`policy/mdqn.py`)
- `BDQ` - Branching DQN for hybrid action space (`policy/bdq.py`)

### 基础 DRL 算法 (Policy-Based)

**Policy Gradient 系列**:
- `PG` - Policy Gradient (`policy/pg.py`)
- `A2C` - Advantage Actor-Critic (`policy/a2c.py`)
- `PPO` - Proximal Policy Optimization (`policy/ppo.py`)
  - 支持离散和连续动作空间
  - 可用于单智能体和多智能体 (MAPPO)
- `PPG` - Phasic Policy Gradient (`policy/ppg.py`)
- `ACER` - Actor-Critic with Experience Replay (`policy/acer.py`)
- `IMPALA` - Importance Weighted Actor-Learner Architecture (`policy/impala.py`)

**Actor-Critic (Continuous)** - 连续动作空间:
- `DDPG` - Deep Deterministic Policy Gradient (`policy/ddpg.py`)
- `TD3` - Twin Delayed DDPG (`policy/td3.py`)
- `D4PG` - Distributed Distributional DDPG (`policy/d4pg.py`)
- `SAC` - Soft Actor-Critic (`policy/sac.py`)
  - 支持离散和连续动作空间
- `SQL` - Soft Q-Learning (`policy/sql.py`)

**Hybrid Action Space** - 混合动作空间:
- `PDQN` - Parameterized DQN (`policy/pdqn.py`)
- `HPPO` - Hybrid PPO (使用 `policy/ppo.py` 配置)

### 多智能体强化学习 (MARL)

**Value Decomposition**:
- `QMIX` - Q-Mixing Networks (`policy/qmix.py`)
- `WQMIX` - Weighted QMIX (`policy/wqmix.py`)
- `QTran` - Q-Transformation (`policy/qtran.py`)
- `CollaQ` - Collaborative Q-learning (`policy/collaq.py`)

**Multi-Agent Policy Gradient**:
- `COMA` - Counterfactual Multi-Agent Policy Gradients (`policy/coma.py`)
- `MAPPO` - Multi-Agent PPO (使用 `policy/ppo.py`)
- `HAPPO` - Heterogeneous-Agent PPO (`policy/happo.py`)
- `MASAC` - Multi-Agent SAC (使用 `policy/sac.py`)

**Communication & Coordination**:
- `ATOC` - Actor-Attentional Communication (`policy/atoc.py`)

### 模仿学习 (Imitation Learning)

- `BC` - Behavior Cloning (`policy/bc.py`)
- `IBC` - Implicit Behavior Cloning (`policy/ibc.py`)
- `IL` - General Imitation Learning (`policy/il.py`)
- `GAIL` - Generative Adversarial Imitation Learning (通过 `serial_entry_gail.py`)
- `SQIL` - Soft Q Imitation Learning (通过 SAC 变体)
- `DQFD` - Deep Q-learning from Demonstrations (`policy/dqfd.py`)

### 离线强化学习 (Offline RL)

- `BCQ` - Batch-Constrained Q-learning (`policy/bcq.py`)
- `CQL` - Conservative Q-Learning (`policy/cql.py`)
- `TD3BC` - TD3 + Behavior Cloning (`policy/td3_bc.py`)
- `EDAC` - Ensemble-Diversified Actor Critic (`policy/edac.py`)
- `IQL` - Implicit Q-Learning (`policy/iql.py`)
- `DT` - Decision Transformer (`policy/dt.py`)

### 基于模型的强化学习 (Model-Based RL)

- `MBPO` - Model-Based Policy Optimization (`policy/mbpolicy/`)
- `DreamerV3` - World Models with Transformers (`policy/mbpolicy/`)
- `SVG` - Stochastic Value Gradients
- `STEVE` - Structured World Models

### 探索算法 (Exploration)

- `NGU` - Never Give Up (`policy/ngu.py`)
- `RND` - Random Network Distillation (作为辅助模块)
- `ICM` - Intrinsic Curiosity Module (作为辅助模块)
- `HER` - Hindsight Experience Replay (作为数据处理模块)

### LLM + RL 算法

- `PromptPG` - Prompt-based Policy Gradient (`policy/prompt_pg.py`)
- `PromptAWR` - Prompt-based Advantage Weighted Regression (`policy/prompt_awr.py`)
- `PPO-max` - PPO for language model alignment (PPO 变体)
- `DPO` - Direct Preference Optimization

### 生成模型 + RL

- `QGPO` - Q-weighted Generative Policy Optimization (`policy/qgpo.py`)
- `PlanDiffuser` - Diffusion-based Planning (`policy/plan_diffuser.py`)
- `Diffuser` - 扩散模型用于决策
- `Decision Diffuser` - 决策扩散模型

### 其他专用算法

- `R2D3` - R2D2 + Demonstrations (`policy/r2d3.py`)
- `PC` - Procedure Cloning BFS (`policy/pc.py`)
- `PPOFPolicy` - PPO with specific features (`policy/ppof.py`)
- `TD3VAE` - TD3 with VAE (`policy/td3_vae.py`)
- `SQN` - Stochastic Q-Network (`policy/sqn.py`)

### 算法选择指南

**离散动作空间**: DQN, C51, Rainbow, R2D2, PPO, A2C, IMPALA  
**连续动作空间**: DDPG, TD3, SAC, PPO  
**混合动作空间**: PDQN, HPPO, BDQ  
**多智能体**: QMIX, MAPPO, HAPPO, COMA  
**离线学习**: BCQ, CQL, TD3BC, IQL, DT  
**模仿学习**: BC, IBC, GAIL, DQFD  
**分布式训练**: R2D2, IMPALA, APEX (通过 parallel_entry)

## 调试技巧

**启用详细日志**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Registry 追踪**:
```bash
export DIENGINEREGTRACE=ON  # 启用注册追踪
```

**检查配置**:
```python
from ding.config import save_config_formatted
save_config_formatted(cfg, 'debug_config.py')  # 保存可读配置
```

**单步调试 Entry**:
```python
# 在 entry 文件末尾设置断点
if __name__ == '__main__':
    serial_pipeline((main_config, create_config), seed=0)
```
