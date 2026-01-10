from typing import Union, Optional, List, Any, Tuple
import os
import torch
import numpy as np
from ditk import logging
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from .utils import random_collect, mark_not_expert


def serial_pipeline_r2d3(
        input_cfg: Union[str, Tuple[dict, dict]], # todo
        expert_cfg: Union[str, Tuple[dict, dict]], # todo
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        expert_model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline r2d3 entry: we create this serial pipeline in order to\
            implement r2d3 in DI-engine. For now, we support the following envs\
            Lunarlander, Pong, Qbert. The demonstration\
            data come from the expert model. We use a well-trained model to \
            generate demonstration data online
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - expert_model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.\
            The default model is DQN(**cfg.policy.model)
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        # 如果输出的是字符串类型，则认为是配置文件路径
        cfg, create_cfg = read_config(input_cfg)
        expert_cfg, expert_create_cfg = read_config(expert_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
        expert_cfg, expert_create_cfg = expert_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command' # 这里将policy type改成带command的干啥？意思就是创建的策略模型要继承自带command的基类，所以后续的在create_policy中创建的策略实际的名字叫：'offppo_collect_traj_command'
    expert_create_cfg.policy.type = expert_create_cfg.policy.type + '_command' # todo 同上
    env_fn = None if env_setting is None else env_setting[0]  # 在r2d3中，传入的是None
    # 以下大概是将传入的进行整合和整理 todo
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    expert_cfg = compile_config(
        expert_cfg, seed=seed, env=env_fn, auto=True, create_cfg=expert_create_cfg, save_cfg=True
    )
    # Create main components: env, policy
    # 获取环境的配置，如果是并行向量环境则得使用特定的方法
    # env_fn： 环境的创建函数 todo
    # collector_env_cfg： 采集环境的配置列表 todo
    # evaluator_env_cfg： 评估环境的配置列表 todo
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    # 调用ENV_MANAGER_REGISTRY注册的工厂函数，传入环境函数和配置，构建对应类型的环境管理器，比如游戏环境管理器
    # cfg.env.manager： 环境管理器的配置，比如子进程以及创建环境时传入的参数等
    # env_fn： 环境的创建函数
    # collector_env_cfg： 环境创建对应的配置列表，每一个环境对应一个配置，这样方便进行不同环境的特殊处理，当然也可以不处理，所有环境都用同一个配置
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    # todo 这里是创建专家采集环境
    expert_collector_env = create_env_manager(
        expert_cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg]
    )
    # todo 这里是创建评估环境
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    # 为保证不同模型的环境一致性，手动指定不同环境的种子
    expert_collector_env.seed(cfg.seed)
    collector_env.seed(cfg.seed)
    # todo 这里的dynamic_seed为什么是False呢？
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    # 创建专家策略和待训练策略
    expert_policy = create_policy(expert_cfg.policy, model=expert_model, enable_field=['collect', 'command'])
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    # 专家策略加载预训练模型参数，运行在cpu上
    # 这里的collect_mode是一个对外的接口集合，
    expert_policy.collect_mode.load_state_dict(torch.load(expert_cfg.policy.collect.model_path, map_location='cpu'))
    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    # 创建训练辅助器，但是和实际的训练无直接关系
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    # 后续调试看看这里的collector具体是啥
    # 根据compile_config中的信息，如果没有配置，则注入默认的采集器，所以其值为：sample
    # 创建待训练模型和专家模型对应的采集器实例
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    expert_collector = create_serial_collector(
        expert_cfg.policy.collect.collector,
        env=expert_collector_env,
        policy=expert_policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=expert_cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    # 这里才是缓冲区，那他和collector中的缓冲区有啥区别呢？如何配合使用的 todo
    # 在compile_config中会将未None的Type设置为：advanced
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    # 
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )
    expert_commander = BaseSerialCommander(
        expert_cfg.policy.other.commander, learner, expert_collector, evaluator, replay_buffer,
        expert_policy.command_mode
    )  # we create this to avoid the issue of eps, this is an issue due to the sample collector part.
    expert_collect_kwargs = expert_commander.step()
    if 'eps' in expert_collect_kwargs:
        expert_collect_kwargs['eps'] = -1
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')
    if expert_cfg.policy.learn.expert_replay_buffer_size != 0:  # for ablation study
        expert_buffer = create_buffer(expert_cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
        expert_data = expert_collector.collect(
            n_sample=expert_cfg.policy.learn.expert_replay_buffer_size,
            train_iter=learner.train_iter,
            policy_kwargs=expert_collect_kwargs
        )

        for i in range(len(expert_data)):
            # set is_expert flag(expert 1, agent 0)
            # expert_data[i]['is_expert'] = 1  # for transition-based alg.
            expert_data[i]['is_expert'] = [1] * expert_cfg.policy.collect.unroll_len  # for rnn/sequence-based alg.
        expert_buffer.push(expert_data, cur_collector_envstep=0)
        for _ in range(cfg.policy.learn.per_train_iter_k):  # pretrain
            if evaluator.should_eval(learner.train_iter):
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break
            # Learn policy from collected data
            # Expert_learner will train ``update_per_collect == 1`` times in one iteration.
            train_data = expert_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            learner.train(train_data, collector.envstep)
            if learner.policy.get_attribute('priority'):
                expert_buffer.update(learner.priority_info)
        learner.priority_info = {}
    # Accumulate plenty of data at the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(
            cfg.policy, policy, collector, collector_env, commander, replay_buffer, postprocess_data_fn=mark_not_expert
        )
    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

        for i in range(len(new_data)):
            # set is_expert flag(expert 1, agent 0)
            new_data[i]['is_expert'] = [0] * expert_cfg.policy.collect.unroll_len

        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            if expert_cfg.policy.learn.expert_replay_buffer_size != 0:
                # Learner will train ``update_per_collect`` times in one iteration.

                # The hyperparameter pho, the demo ratio, control the propotion of data coming\
                # from expert demonstrations versus from the agent's own experience.
                expert_batch_size = int(
                    np.float32(np.random.rand(learner.policy.get_attribute('batch_size')) < cfg.policy.collect.pho
                               ).sum()
                )
                agent_batch_size = (learner.policy.get_attribute('batch_size')) - expert_batch_size
                train_data_agent = replay_buffer.sample(agent_batch_size, learner.train_iter)
                train_data_expert = expert_buffer.sample(expert_batch_size, learner.train_iter)
                if train_data_agent is None:
                    # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                    logging.warning(
                        "Replay buffer's data can only train for {} steps. ".format(i) +
                        "You can modify data collect config, e.g. increasing n_sample, n_episode."
                    )
                    break
                train_data = train_data_agent + train_data_expert
                learner.train(train_data, collector.envstep)
                if learner.policy.get_attribute('priority'):
                    # When collector, set replay_buffer_idx and replay_unique_id for each data item, priority = 1.\
                    # When learner, assign priority for each data item according their loss
                    learner.priority_info_agent = deepcopy(learner.priority_info)
                    learner.priority_info_expert = deepcopy(learner.priority_info)
                    learner.priority_info_agent['priority'] = learner.priority_info['priority'][0:agent_batch_size]
                    learner.priority_info_agent['replay_buffer_idx'] = learner.priority_info['replay_buffer_idx'][
                        0:agent_batch_size]
                    learner.priority_info_agent['replay_unique_id'] = learner.priority_info['replay_unique_id'][
                        0:agent_batch_size]

                    learner.priority_info_expert['priority'] = learner.priority_info['priority'][agent_batch_size:]
                    learner.priority_info_expert['replay_buffer_idx'] = learner.priority_info['replay_buffer_idx'][
                        agent_batch_size:]
                    learner.priority_info_expert['replay_unique_id'] = learner.priority_info['replay_unique_id'][
                        agent_batch_size:]

                    # Expert data and demo data update their priority separately.
                    replay_buffer.update(learner.priority_info_agent)
                    expert_buffer.update(learner.priority_info_expert)
            else:
                # Learner will train ``update_per_collect`` times in one iteration.
                train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                if train_data is None:
                    # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                    logging.warning(
                        "Replay buffer's data can only train for {} steps. ".format(i) +
                        "You can modify data collect config, e.g. increasing n_sample, n_episode."
                    )
                    break
                learner.train(train_data, collector.envstep)
                if learner.policy.get_attribute('priority'):
                    replay_buffer.update(learner.priority_info)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
