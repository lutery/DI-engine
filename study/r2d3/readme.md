通过分析代码，我发现R2D3算法的训练入口和调用方式如下：

R2D3算法的训练入口
1. 直接运行配置文件
R2D3算法的主要训练方式是直接运行配置文件。例如，在LunarLander环境中：

# dizoo/box2d/lunarlander/config/lunarlander_r2d3_ppoexpert_config.py
if __name__ == "__main__":
    from ding.entry import serial_pipeline_r2d3
    serial_pipeline_r2d3([main_config, create_config], [expert_main_config, expert_create_config], seed=0)
2. 核心入口函数
R2D3算法的核心入口函数是：

函数位置: ding/entry/serial_entry_r2d3.py
函数名: serial_pipeline_r2d3
导入路径: from ding.entry import serial_pipeline_r2d3
调用流程
1. 命令行调用方式
虽然CLI工具(ding/entry/cli.py)中没有直接的R2D3模式选项，但可以通过以下方式调用：

# 直接运行配置文件
python dizoo/box2d/lunarlander/config/lunarlander_r2d3_ppoexpert_config.py
2. 函数调用方式
from ding.entry import serial_pipeline_r2d3

# 调用R2D3训练管道
serial_pipeline_r2d3(
    input_cfg=[main_config, create_config],  # 主算法配置
    expert_cfg=[expert_main_config, expert_create_config],  # 专家策略配置
    seed=0  # 随机种子
)
3. 参数说明
input_cfg: 主算法配置，包含环境和策略配置
expert_cfg: 专家策略配置，用于生成演示数据
seed: 随机种子，用于实验可复现性
训练流程
初始化环境: 根据配置创建环境和策略
加载专家模型: 从expert_cfg加载预训练的专家模型
收集专家数据: 使用专家策略收集演示数据
混合数据训练: 按比例(pho参数)混合专家数据和智能体自身数据
更新策略: 使用R2D3算法更新策略网络
注意事项
专家模型路径: 需要在配置文件中指定专家模型的路径(model_path)
数据比例: 通过pho参数控制专家数据和智能体数据的混合比例
序列长度: R2D3需要设置合适的序列长度(learn_unroll_len和burnin_step)
R2D3算法没有集成到CLI工具的直接模式中，需要通过直接运行配置文件或调用serial_pipeline_r2d3函数来使用。