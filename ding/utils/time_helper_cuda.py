from typing import Callable
import torch
from .time_helper_base import TimeWrapper


def get_cuda_time_wrapper() -> Callable[[], 'TimeWrapper']:
    """
    Overview:
        Return the ``TimeWrapperCuda`` class, this wrapper aims to ensure compatibility in no cuda device

    Returns:
        - TimeWrapperCuda(:obj:`class`): See ``TimeWrapperCuda`` class

    .. note::
        Must use ``torch.cuda.synchronize()``, reference: <https://blog.csdn.net/u013548568/article/details/81368019>

    """

    # TODO find a way to autodoc the class within method
    class TimeWrapperCuda(TimeWrapper):
        """
        Overview:
            A class method that inherit from ``TimeWrapper`` class
            专门用于模型在gpu中耗时的计时器类
            GPU 操作是异步的：model(data) 会立即返回，但 GPU 还在后台计算
            CPU 计时只测量了提交任务的时间，而不是实际执行时间
            结果会比实际耗时短很多
            具体查看md文档

            Notes:
                Must use torch.cuda.synchronize(), reference: \
                <https://blog.csdn.net/u013548568/article/details/81368019>

        Interfaces:
            ``start_time``, ``end_time``
        """
        # cls variable is initialized on loading this class
        start_record = torch.cuda.Event(enable_timing=True)
        end_record = torch.cuda.Event(enable_timing=True)

        # overwrite
        @classmethod
        def start_time(cls):
            """
            Overview:
                Implement and overide the ``start_time`` method in ``TimeWrapper`` class
            """
            torch.cuda.synchronize()
            cls.start = cls.start_record.record()

        # overwrite
        @classmethod
        def end_time(cls):
            """
            Overview:
                Implement and overide the end_time method in ``TimeWrapper`` class
            Returns:
                - time(:obj:`float`): The time between ``start_time`` and ``end_time``
            """
            cls.end = cls.end_record.record()
            torch.cuda.synchronize()
            return cls.start_record.elapsed_time(cls.end_record) / 1000

    return TimeWrapperCuda
