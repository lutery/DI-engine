from collections import namedtuple
from typing import Optional
from typing import Tuple

import gym
import numpy as np
import cv2
import os
from gym import spaces
from gym.utils import seeding

# gym.logger.set_level(40)  # noqa

from .agents import BaseAgent, MovingAgent, SlidingAgent, HardMoveAgent

# Action Id
ACCELERATE = 0
TURN = 1
BREAK = 2

Target = namedtuple('Target', ['x', 'y', 'radius'])


class Action:
    """"
    Action class to store and standardize the action for the environment.
    """

    def __init__(self, id_: int, parameters: list):
        """"
        Initialization of an action.

        Args:
            id_: The id of the selected action.
            parameters: The parameters of an action.
        """
        self.id = id_
        self.parameters = parameters

    @property
    def parameter(self) -> float:
        """"
        Property method to return the parameter related to the action selected.

        Returns:
            The parameter related to this action_id
        """
        if len(self.parameters) == 2:
            return self.parameters[self.id]
        else:
            return self.parameters[0]


class BaseEnv(gym.Env):
    """"
    Gym environment parent class.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        max_turn: float = np.pi / 2,
        max_acceleration: float = 0.5,
        delta_t: float = 0.005,
        max_step: int = 200,
        penalty: float = 0.001,
        break_value: float = 0.1,
    ):
        """Initialization of the gym environment.

        Args:
            seed (int): Seed used to get reproducible results.
            max_turn (float): Maximum turn during one step (in radian).
            max_acceleration (float): Maximum acceleration during one step.
            delta_t (float): Time duration of one step.
            max_step (int): Maximum number of steps in one episode.
            penalty (float): Score penalty given at the agent every step.
            break_value (float): Break value when performing break action.
        """
        # Agent Parameters
        self.max_turn = max_turn
        self.max_acceleration = max_acceleration
        self.break_value = break_value

        # Environment Parameters
        self.delta_t = delta_t
        self.max_step = max_step
        self.field_size = 1.0
        self.target_radius = 0.1
        self.penalty = penalty

        # Initialization
        self.seed(seed)
        self.target = None
        self.current_step = None
        self.agent = BaseAgent(break_value=break_value, delta_t=delta_t)
        self._window_name = None

        parameters_min = np.array([0, -1])
        parameters_max = np.array([1, +1])

        self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Box(parameters_min, parameters_max)))
        self.observation_space = spaces.Box(np.ones(10), -np.ones(10))
        dirname = os.path.dirname(__file__)
        self.bg = cv2.imread(os.path.join(dirname, 'bg.jpg'))
        if self.bg is not None:
            self.bg = cv2.cvtColor(self.bg, cv2.COLOR_BGR2RGB)
            self.bg = cv2.resize(self.bg, (800, 800))
        else:
            self.bg = np.ones((800, 800, 3), dtype=np.uint8) * 255
        self.target_img = cv2.imread(os.path.join(dirname, 'target.png'), cv2.IMREAD_UNCHANGED)
        if self.target_img is not None:
            self.target_img = cv2.resize(self.target_img, (60, 60))

    def seed(self, seed: Optional[int] = None) -> list:
        self.np_random, seed = seeding.np_random(seed)  # noqa
        return [seed]

    def reset(self) -> list:
        self.current_step = 0

        limit = self.field_size - self.target_radius
        low = [-limit, -limit, self.target_radius]
        high = [limit, limit, self.target_radius]
        self.target = Target(*self.np_random.uniform(low, high))

        low = [-self.field_size, -self.field_size, 0]
        high = [self.field_size, self.field_size, 2 * np.pi]
        self.agent.reset(*self.np_random.uniform(low, high))

        return self.get_state()

    def step(self, raw_action: Tuple[int, list]) -> Tuple[list, float, bool, dict]:
        action = Action(*raw_action)
        last_distance = self.distance
        self.current_step += 1

        if action.id == TURN:
            rotation = self.max_turn * max(min(action.parameter, 1), -1)
            self.agent.turn(rotation)
        elif action.id == ACCELERATE:
            acceleration = self.max_acceleration * max(min(action.parameter, 1), 0)
            self.agent.accelerate(acceleration)
        elif action.id == BREAK:
            self.agent.break_()

        if self.distance < self.target_radius and self.agent.speed == 0:
            reward = self.get_reward(last_distance, True)
            done = True
        elif abs(self.agent.x) > self.field_size or abs(self.agent.y
                                                        ) > self.field_size or self.current_step > self.max_step:
            reward = -1
            done = True
        else:
            reward = self.get_reward(last_distance)
            done = False

        return self.get_state(), reward, done, {}

    def get_state(self) -> list:
        state = [
            self.agent.x, self.agent.y, self.agent.speed,
            np.cos(self.agent.theta),
            np.sin(self.agent.theta), self.target.x, self.target.y, self.distance,
            0 if self.distance > self.target_radius else 1, self.current_step / self.max_step
        ]
        return state

    def get_reward(self, last_distance: float, goal: bool = False) -> float:
        return last_distance - self.distance - self.penalty + (1 if goal else 0)

    @property
    def distance(self) -> float:
        return self.get_distance(self.agent.x, self.agent.y, self.target.x, self.target.y)

    @staticmethod
    def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2)).item()

    def render(self, mode: str = 'human'):
        frame = self.bg.copy() if self.bg is not None else np.ones((800, 800, 3), dtype=np.uint8) * 255
        height, width = frame.shape[:2]
        unit_x = width / 2
        unit_y = height / 2
        agent_radius_ratio = 0.05

        # 计算目标区域并绘制
        target_px = int(unit_x * (1 + self.target.x / self.field_size))
        target_py = int(unit_y * (1 - self.target.y / self.field_size))
        target_radius_px = max(1, int(self.target_radius / self.field_size * min(unit_x, unit_y)))
        cv2.circle(frame, (target_px, target_py), target_radius_px, (0, 153, 0), 2)

        # 绘制智能体
        agent_px = int(unit_x * (1 + self.agent.x / self.field_size))
        agent_py = int(unit_y * (1 - self.agent.y / self.field_size))
        agent_radius_px = max(1, int(agent_radius_ratio * min(unit_x, unit_y)))
        cv2.circle(frame, (agent_px, agent_py), agent_radius_px, (26, 77, 230), -1)

        # 绘制朝向箭头
        arrow_length = int(0.12 * min(unit_x, unit_y))
        arrow_end_x = int(agent_px + arrow_length * np.cos(self.agent.theta))
        arrow_end_y = int(agent_py - arrow_length * np.sin(self.agent.theta))
        cv2.arrowedLine(frame, (agent_px, agent_py), (arrow_end_x, arrow_end_y), (0, 0, 0), 2, tipLength=0.2)

        # 添加边框
        frame[:6, :] = (60, 60, 30)
        frame[-6:, :] = (60, 60, 30)
        frame[:, :6] = (60, 60, 30)
        frame[:, -6:] = (60, 60, 30)

        if mode == 'rgb_array':
            return frame

        if mode == 'human':
            if self._window_name is None:
                self._window_name = f'{self.__class__.__name__}'
                cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(self._window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            return frame

        raise NotImplementedError(f'不支持的render模式: {mode}')

    def close(self):
        if self._window_name is not None:
            cv2.destroyWindow(self._window_name)
            self._window_name = None


class MovingEnv(BaseEnv):

    def __init__(
        self,
        seed: int = None,
        max_turn: float = np.pi / 2,
        max_acceleration: float = 0.5,
        delta_t: float = 0.005,
        max_step: int = 200,
        penalty: float = 0.001,
        break_value: float = 0.1,
    ):
        super(MovingEnv, self).__init__(
            seed=seed,
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value,
        )

        self.agent = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )


class SlidingEnv(BaseEnv):

    def __init__(
        self,
        seed: int = None,
        max_turn: float = np.pi / 2,
        max_acceleration: float = 0.5,
        delta_t: float = 0.005,
        max_step: int = 200,
        penalty: float = 0.001,
        break_value: float = 0.1
    ):
        super(SlidingEnv, self).__init__(
            seed=seed,
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value
        )

        self.agent = SlidingAgent(break_value=break_value, delta_t=delta_t)


class HardMoveEnv(gym.Env):
    """"
    HardMove environment. Please refer to https://arxiv.org/abs/2109.05490 for details.
    """

    def __init__(
        self,
        num_actuators: int = 4,
        seed: Optional[int] = None,
        max_turn: float = np.pi / 2,
        max_acceleration: float = 0.5,
        delta_t: float = 0.005,
        max_step: int = 25,
        penalty: float = 0.001,
        break_value: float = 0.1,
    ):
        """Initialization of the gym environment.

        Args:
            seed (int): Seed used to get reproducible results.
            max_turn (float): Maximum turn during one step (in radian).
            max_acceleration (float): Maximum acceleration during one step.
            delta_t (float): Time duration of one step.
            max_step (int): Maximum number of steps in one episode.
            penalty (float): Score penalty given at the agent every step.
            break_value (float): Break value when performing break action.
        """
        # Agent Parameters
        self.num_actuators = num_actuators
        self.max_turn = max_turn
        self.max_acceleration = max_acceleration
        self.break_value = break_value

        # Environment Parameters
        self.delta_t = delta_t
        self.max_step = max_step
        self.field_size = 1.0
        self.target_radius = 0.1
        self.penalty = penalty

        # Initialization
        self.seed(seed)
        self.target = None
        self.viewer = None
        self.current_step = None
        self.agent = HardMoveAgent(break_value=break_value, delta_t=delta_t, num_actuators=self.num_actuators)

        parameters_min = np.array([-1 for i in range(self.num_actuators)])
        parameters_max = np.array([+1 for i in range(self.num_actuators)])

        self.action_space = spaces.Tuple(
            (spaces.Discrete(int(2 ** self.num_actuators)), spaces.Box(parameters_min, parameters_max))
        )
        self.observation_space = spaces.Box(np.ones(10), -np.ones(10))

    def seed(self, seed: Optional[int] = None) -> list:
        self.np_random, seed = seeding.np_random(seed)  # noqa
        return [seed]

    def reset(self) -> list:
        self.current_step = 0

        limit = self.field_size - self.target_radius
        low = [-limit, -limit, self.target_radius]
        high = [limit, limit, self.target_radius]
        self.target = Target(*self.np_random.uniform(low, high))

        low = [-self.field_size, -self.field_size, 0]
        high = [self.field_size, self.field_size, 2 * np.pi]
        self.agent.reset(*self.np_random.uniform(low, high))

        return self.get_state()

    def step(self, raw_action: Tuple[int, list]) -> Tuple[list, float, bool, dict]:
        move_direction_meta = raw_action[0]  # shape (1,) in {2**n}
        move_distances = raw_action[1]  # shape (2**n,)
        last_distance = self.distance
        self.current_step += 1

        self.agent.move(move_direction_meta, move_distances)
        if self.distance < self.target_radius:
            reward = self.get_reward(last_distance, True)
            done = True
        elif abs(self.agent.x) > self.field_size or abs(self.agent.y
                                                        ) > self.field_size or self.current_step > self.max_step:
            reward = -1
            done = True
        else:
            reward = self.get_reward(last_distance)
            done = False

        return self.get_state(), reward, done, {}

    def get_state(self) -> list:
        state = [
            self.agent.x, self.agent.y, self.agent.speed,
            np.cos(self.agent.theta),
            np.sin(self.agent.theta), self.target.x, self.target.y, self.distance,
            0 if self.distance > self.target_radius else 1, self.current_step / self.max_step
        ]
        return state

    def get_reward(self, last_distance: float, goal: bool = False) -> float:
        return last_distance - self.distance - self.penalty + (1 if goal else 0)

    @property
    def distance(self) -> float:
        return self.get_distance(self.agent.x, self.agent.y, self.target.x, self.target.y)

    @staticmethod
    def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2)).item()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
