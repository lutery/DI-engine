from typing import Union, Dict, Optional
from easydict import EasyDict
import torch
import torch.nn as nn
from copy import deepcopy
from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import ReparameterizationHead, RegressionHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder, IMPALAConvEncoder
from ding.torch_utils.network.dreamer import ActionHead, DenseHead


@MODEL_REGISTRY.register('vac')
class VAC(nn.Module):
    """
    Overview:
        VAC网络就是一个典型的actor-critic架构的实现。而PPO也属于这类型的网络
        The neural network and computation graph of algorithms related to (state) Value Actor-Critic (VAC), such as \
        A2C/PPO/IMPALA. This model now supports discrete, continuous and hybrid action space. The VAC is composed of \
        four parts: ``actor_encoder``, ``critic_encoder``, ``actor_head`` and ``critic_head``. Encoders are used to \
        extract the feature from various observation. Heads are used to predict corresponding value or action logit. \
        In high-dimensional observation space like 2D image, we often use a shared encoder for both ``actor_encoder`` \
        and ``critic_encoder``. In low-dimensional observation space like 1D vector, we often use different encoders.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``, ``compute_actor_critic``.
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType, EasyDict],
        action_space: str = 'discrete',
        share_encoder: bool = True,
        encoder_hidden_size_list: SequenceType = [128, 128, 64], # 最后一个维度是特征提取编码器的输出维度,其余的是特征提取器内部的维度变化,从这里来看，特征提取默认不使用残差
        actor_head_hidden_size: int = 64,
        actor_head_layer_num: int = 1,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        sigma_type: Optional[str] = 'independent',
        fixed_sigma_value: Optional[int] = 0.3,
        bound_type: Optional[str] = None,
        encoder: Optional[torch.nn.Module] = None,
        impala_cnn_encoder: bool = False,
    ) -> None:
        """
        Overview:
            Initialize the VAC model according to corresponding input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
            - action_space (:obj:`str`): The type of different action spaces, including ['discrete', 'continuous', \
                'hybrid'], then will instantiate corresponding head, including ``DiscreteHead``, \
                ``ReparameterizationHead``, and hybrid heads.
            - share_encoder (:obj:`bool`): Whether to share observation encoders between actor and decoder.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element is used as the input size of ``actor_head`` and ``critic_head``.
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of ``actor_head`` network, defaults \
                to 64, it is the hidden size of the last layer of the ``actor_head`` network.
            - actor_head_layer_num (:obj:`int`): The num of layers used in the ``actor_head`` network to compute action.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of ``critic_head`` network, defaults \
                to 64, it is the hidden size of the last layer of the ``critic_head`` network.
            - critic_head_layer_num (:obj:`int`): The num of layers used in the ``critic_head`` network.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.ReLU()``.
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details. you can choose one of ['BN', 'IN', 'SyncBN', 'LN']
            - sigma_type (:obj:`Optional[str]`): The type of sigma in continuous action space, see \
                ``ding.torch_utils.network.dreamer.ReparameterizationHead`` for more details, in A2C/PPO, it defaults \
                to ``independent``, which means state-independent sigma parameters.
            - fixed_sigma_value (:obj:`Optional[int]`): If ``sigma_type`` is ``fixed``, then use this value as sigma.
            - bound_type (:obj:`Optional[str]`): The type of action bound methods in continuous action space, defaults \
                to ``None``, which means no bound.
            - encoder (:obj:`Optional[torch.nn.Module]`): The encoder module, defaults to ``None``, you can define \
                your own encoder module and pass it into VAC to deal with different observation space.
            - impala_cnn_encoder (:obj:`bool`): Whether to use IMPALA CNN encoder, defaults to ``False``. 一种具备残差结构的卷积网络，比普通卷积网络更深更强大，但在r2d3 ppo中没有使用
        """
        super(VAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.obs_shape, self.action_shape = obs_shape, action_shape
        self.impala_cnn_encoder = impala_cnn_encoder # 在r2d3 ppo中没有传入，为False
        self.share_encoder = share_encoder # 看起来是共享编码器，在r2d3 ppo中没有传入参数，为True

        # Encoder Type
        def new_encoder(outsize, activation):
            '''
            Docstring for new_encoder
            
            :param outsize: 输出的特征维度
            :param activation: 激活函数
            '''

            if impala_cnn_encoder: # r2d3 ppo中没有使用 todo 后续自己看看，说不定可以应用在自己的项目中
                return IMPALAConvEncoder(obs_shape=obs_shape, channels=encoder_hidden_size_list, outsize=outsize)
            else:
                if isinstance(obs_shape, int) or len(obs_shape) == 1: # 针对输入的特征是一个向量的特征，例如 mujoco 模拟环境中的状态特征
                    return FCEncoder(
                        obs_shape=obs_shape,
                        hidden_size_list=encoder_hidden_size_list,
                        activation=activation,
                        norm_type=norm_type
                    )
                elif len(obs_shape) == 3: # 针对输入的特征是一个图像，例如 Atari 游戏环境中的图像特征
                    return ConvEncoder(
                        obs_shape=obs_shape,
                        hidden_size_list=encoder_hidden_size_list,
                        activation=activation,
                        norm_type=norm_type
                    )
                else: # 除此之外的其他输入特征均不支持
                    raise RuntimeError(
                        "not support obs_shape for pre-defined encoder: {}, please customize your own encoder".
                        format(obs_shape)
                    )

        if self.share_encoder:
            if encoder:
                # 如果外部有传入则使用外部的，如果没有传入则自己创建
                if isinstance(encoder, torch.nn.Module):
                    self.encoder = encoder
                else:
                    raise ValueError("illegal encoder instance.")
            else:
                # 创建默认obs编码器
                self.encoder = new_encoder(encoder_hidden_size_list[-1], activation)
        else:
            # 如果不共享编码器，那么动作编码器和价值编码器是分开计算权重的
            if encoder:
                if isinstance(encoder, torch.nn.Module):
                    self.actor_encoder = encoder
                    self.critic_encoder = deepcopy(encoder)
                else:
                    raise ValueError("illegal encoder instance.")
            else:
                self.actor_encoder = new_encoder(encoder_hidden_size_list[-1], activation)
                self.critic_encoder = new_encoder(encoder_hidden_size_list[-1], activation)

        # Head Type 价值预测头，这里就是直接输出预测的价值
        self.critic_head = RegressionHead(
            encoder_hidden_size_list[-1], # 输入的特征维度
            1, # 输出的特征维度
            critic_head_layer_num,
            activation=activation,
            norm_type=norm_type,
            hidden_size=critic_head_hidden_size
        )
        self.action_space = action_space # 动作空间的类型，字符串类型，根据不同的动作空间的类型，创建不同的动作头
        assert self.action_space in ['discrete', 'continuous', 'hybrid'], self.action_space # 仅支持离散、连续、混合动作空间
        # todo self.multi_head 是啥？
        if self.action_space == 'continuous':
            self.multi_head = False # 连续动作空间不支持多头 todo
            # 构建连续动作预测
            self.actor_head = ReparameterizationHead(
                encoder_hidden_size_list[-1],
                action_shape,
                actor_head_layer_num,
                sigma_type=sigma_type,
                activation=activation,
                norm_type=norm_type,
                bound_type=bound_type,
                hidden_size=actor_head_hidden_size,
            )
        elif self.action_space == 'discrete':
            actor_head_cls = DiscreteHead # 离散动作头的类型
            multi_head = not isinstance(action_shape, int)
            self.multi_head = multi_head # 离散动作空间是否是多个独立的离散动作空间
            if multi_head:
                self.actor_head = MultiHead(
                    actor_head_cls,
                    actor_head_hidden_size,
                    action_shape,
                    layer_num=actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
            else:
                self.actor_head = actor_head_cls(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
        elif self.action_space == 'hybrid':  # HPPO
            # 混合动作空间就没有多头之分了 todo 后续再说，本次不看
            # hybrid action space: action_type(discrete) + action_args(continuous),
            # such as {'action_type_shape': torch.LongTensor([0]), 'action_args_shape': torch.FloatTensor([0.1, -0.27])}
            action_shape.action_args_shape = squeeze(action_shape.action_args_shape)
            action_shape.action_type_shape = squeeze(action_shape.action_type_shape)
            actor_action_args = ReparameterizationHead(
                encoder_hidden_size_list[-1],
                action_shape.action_args_shape,
                actor_head_layer_num,
                sigma_type=sigma_type,
                fixed_sigma_value=fixed_sigma_value,
                activation=activation,
                norm_type=norm_type,
                bound_type=bound_type,
                hidden_size=actor_head_hidden_size,
            )
            actor_action_type = DiscreteHead(
                actor_head_hidden_size,
                action_shape.action_type_shape,
                actor_head_layer_num,
                activation=activation,
                norm_type=norm_type,
            )
            self.actor_head = nn.ModuleList([actor_action_type, actor_action_args])

        # 根据是否共享特征提取层，构建不同组合
        if self.share_encoder:
            self.actor = [self.encoder, self.actor_head]
            self.critic = [self.encoder, self.critic_head]
        else:
            self.actor = [self.actor_encoder, self.actor_head]
            self.critic = [self.critic_encoder, self.critic_head]
        # Convenient for calling some apis (e.g. self.critic.parameters()),
        # but may cause misunderstanding when `print(self)`
        # 构建实际的动作和评价预测网络
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)

    def forward(self, x: torch.Tensor, mode: str) -> Dict:
        """
        Overview:
            VAC forward computation graph, input observation tensor to predict state value or action logit. Different \
            ``mode`` will forward with different network modules to get different outputs and save computation.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
            - mode (:obj:`str`): The forward mode, all the modes are defined in the beginning of this class.
        Returns:
            - outputs (:obj:`Dict`): The output dict of VAC's forward computation graph, whose key-values vary from \
                different ``mode``.

        Examples (Actor):
            >>> model = VAC(64, 128)
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 128])

        Examples (Critic):
            >>> model = VAC(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 64])

        Examples (Actor-Critic):
            >>> model = VAC(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> assert critic_outputs['value'].shape == torch.Size([4])
            >>> assert outputs['logit'].shape == torch.Size([4, 64])

        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(x)

    def compute_actor(self, x: Union[torch.Tensor, Dict]) -> Dict:
        """
        Overview:
            VAC forward computation graph for actor part, input observation tensor to predict action logit.
        Arguments:
            - x (:obj:`Union[torch.Tensor, Dict]`): The input observation tensor data. If a dictionary is provided, \
                it should contain keys 'observation' and optionally 'action_mask'.
        Returns:
            - outputs (:obj:`Dict`): The output dict of VAC's forward computation graph for actor, including ``logit`` \
                and optionally ``action_mask`` if the input is a dictionary.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): The predicted action logit tensor, for discrete action space, it will be \
                the same dimension real-value ranged tensor of possible action choices, and for continuous action \
                space, it will be the mu and sigma of the Gaussian distribution, and the number of mu and sigma is the \
                same as the number of continuous actions. Hybrid action space is a kind of combination of discrete \
                and continuous action space, so the logit will be a dict with ``action_type`` and ``action_args``.
            - action_mask (:obj:`Optional[torch.Tensor]`): The action mask tensor, included if the input is a \
                dictionary containing 'action_mask'.
        Shapes:
            - logit (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``

        Examples:
            >>> model = VAC(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 64])
        """
        if isinstance(x, dict):
            action_mask = x['action_mask']
            x = self.encoder(x['observation']) if self.share_encoder else self.actor_encoder(x['observation'])
        else:
            action_mask = None
            x = self.encoder(x) if self.share_encoder else self.actor_encoder(x)

        if self.action_space == 'discrete':
            result = {'logit': self.actor_head(x)['logit']}
            if action_mask is not None:
                result['action_mask'] = action_mask
            return result
        elif self.action_space == 'continuous':
            x = self.actor_head(x)  # mu, sigma
            return {'logit': x}
        elif self.action_space == 'hybrid':
            action_type = self.actor_head[0](x)
            action_args = self.actor_head[1](x)
            return {'logit': {'action_type': action_type['logit'], 'action_args': action_args}}

    def compute_critic(self, x: Union[torch.Tensor, Dict]) -> Dict:
        """
        Overview:
            VAC forward computation graph for critic part, input observation tensor to predict state value.
        Arguments:
            - x (:obj:`Union[torch.Tensor, Dict]`): The input observation tensor data. If a dictionary is provided, \
                it should contain the key 'observation'.
        Returns:
            - outputs (:obj:`Dict`): The output dict of VAC's forward computation graph for critic, including ``value``.
        ReturnsKeys:
            - value (:obj:`torch.Tensor`): The predicted state value tensor.
        Shapes:
            - value (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch size, (B, 1) is squeezed to (B, ).

        Examples:
            >>> model = VAC(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> assert critic_outputs['value'].shape == torch.Size([4])
        """
        if isinstance(x, dict):
            x = self.encoder(x['observation']) if self.share_encoder else self.critic_encoder(x['observation'])
        else:
            x = self.encoder(x) if self.share_encoder else self.critic_encoder(x)
        x = self.critic_head(x)
        return {'value': x['pred']}

    def compute_actor_critic(self, x: Union[torch.Tensor, Dict]) -> Dict:
        """
        Overview:
            VAC forward computation graph for both actor and critic part, input observation tensor to predict action \
            logit and state value.
        Arguments:
            - x (:obj:`Union[torch.Tensor, Dict]`): The input observation tensor data. If a dictionary is provided, \
                it should contain keys 'observation' and optionally 'action_mask'.
        Returns:
            - outputs (:obj:`Dict`): The output dict of VAC's forward computation graph for both actor and critic, \
                including ``logit``, ``value``, and optionally ``action_mask`` if the input is a dictionary.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): The predicted action logit tensor, for discrete action space, it will be \
                the same dimension real-value ranged tensor of possible action choices, and for continuous action \
                space, it will be the mu and sigma of the Gaussian distribution, and the number of mu and sigma is the \
                same as the number of continuous actions. Hybrid action space is a kind of combination of discrete \
                and continuous action space, so the logit will be a dict with ``action_type`` and ``action_args``.
            - value (:obj:`torch.Tensor`): The predicted state value tensor.
            - action_mask (:obj:`torch.Tensor`, optional): The action mask tensor, included if the input is a \
                dictionary containing 'action_mask'.
        Shapes:
            - logit (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - value (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch size, (B, 1) is squeezed to (B, ).

        Examples:
            >>> model = VAC(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> assert critic_outputs['value'].shape == torch.Size([4])
            >>> assert outputs['logit'].shape == torch.Size([4, 64])


        .. note::
            ``compute_actor_critic`` interface aims to save computation when shares encoder and return the combination \
            dict output.
        """
        if isinstance(x, dict):
            action_mask = x['action_mask']
            if self.share_encoder:
                actor_embedding = critic_embedding = self.encoder(x['observation'])
            else:
                actor_embedding = self.actor_encoder(x['observation'])
                critic_embedding = self.critic_encoder(x['observation'])
        else:
            action_mask = None
            if self.share_encoder:
                actor_embedding = critic_embedding = self.encoder(x)
            else:
                actor_embedding = self.actor_encoder(x)
                critic_embedding = self.critic_encoder(x)

        value = self.critic_head(critic_embedding)['pred']

        if self.action_space == 'discrete':
            logit = self.actor_head(actor_embedding)['logit']
            result = {'logit': logit, 'value': value}
            if action_mask is not None:
                result['action_mask'] = action_mask
            return result
        elif self.action_space == 'continuous':
            x = self.actor_head(actor_embedding)
            return {'logit': x, 'value': value}
        elif self.action_space == 'hybrid':
            action_type = self.actor_head[0](actor_embedding)
            action_args = self.actor_head[1](actor_embedding)
            return {'logit': {'action_type': action_type['logit'], 'action_args': action_args}, 'value': value}


@MODEL_REGISTRY.register('dreamervac')
class DREAMERVAC(nn.Module):
    """
    Overview:
        The neural network and computation graph of DreamerV3 (state) Value Actor-Critic (VAC).
        This model now supports discrete, continuous action space.
    Interfaces:
        ``__init__``, ``forward``.
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
            self,
            action_shape: Union[int, SequenceType, EasyDict],
            dyn_stoch=32,
            dyn_deter=512,
            dyn_discrete=32,
            actor_layers=2,
            value_layers=2,
            units=512,
            act='SiLU',
            norm='LayerNorm',
            actor_dist='normal',
            actor_init_std=1.0,
            actor_min_std=0.1,
            actor_max_std=1.0,
            actor_temp=0.1,
            action_unimix_ratio=0.01,
    ) -> None:
        """
        Overview:
            Initialize the ``DREAMERVAC`` model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
        """
        super(DREAMERVAC, self).__init__()
        action_shape = squeeze(action_shape)
        self.action_shape = action_shape

        if dyn_discrete:
            feat_size = dyn_stoch * dyn_discrete + dyn_deter
        else:
            feat_size = dyn_stoch + dyn_deter
        self.actor = ActionHead(
            feat_size,  # pytorch version
            action_shape,
            actor_layers,
            units,
            act,
            norm,
            actor_dist,
            actor_init_std,
            actor_min_std,
            actor_max_std,
            actor_temp,
            outscale=1.0,
            unimix_ratio=action_unimix_ratio,
        )
        self.critic = DenseHead(
            feat_size,  # pytorch version
            (255, ),
            value_layers,
            units,
            'SiLU',  # act
            'LN',  # norm
            'twohot_symlog',
            outscale=0.0,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
