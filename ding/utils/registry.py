import inspect
import os
from collections import OrderedDict
from typing import Optional, Iterable, Callable

_innest_error = True

# 从配置文件或者环境变量中获取
_DI_ENGINE_REG_TRACE_IS_ON = os.environ.get('DIENGINEREGTRACE', 'OFF').upper() == 'ON'


class Registry(dict):
    """
    Overview:
        A helper class for managing registering modules, it extends a dictionary
        and provides a register functions.
    Interfaces:
        ``__init__``, ``register``, ``get``, ``build``, ``query``, ``query_details``
    Examples (creating):
        >>> some_registry = Registry({"default": default_module})

    Examples (registering: normal way):
        >>> def foo():
        >>>     ...
        >>> some_registry.register("foo_module", foo)

    Examples (registering: decorator way):
        >>> @some_registry.register("foo_module")
        >>> @some_registry.register("foo_modeul_nickname")
        >>> def foo():
        >>>     ...

    Examples (accessing):
        >>> f = some_registry["foo_module"]
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Overview:
            Initialize the Registry object.
        Arguments:
            - args (:obj:`Tuple`): The arguments passed to the ``__init__`` function of the parent class, \
                dict.
            - kwargs (:obj:`Dict`): The keyword arguments passed to the ``__init__`` function of the parent class, \
                dict.
        """

        super(Registry, self).__init__(*args, **kwargs)
        self.__trace__ = dict() # 用于存储注册模块时所在的文件名和行号

    def register(
            self,
            module_name: Optional[str] = None,
            module: Optional[Callable] = None,
            force_overwrite: bool = False
    ) -> Callable:
        """
        Overview:
            Register the module.
        Arguments:
            - module_name (:obj:`Optional[str]`): The name of the module. 注册的模块名称
            - module (:obj:`Optional[Callable]`): The module to be registered. 注册的模块,可以是函数或者类
            - force_overwrite (:obj:`bool`): Whether to overwrite the module with the same name.
        """

        if _DI_ENGINE_REG_TRACE_IS_ON:
            frame = inspect.stack()[1][0] # todo 这里是存储啥的
            info = inspect.getframeinfo(frame) # 获取调用注册函数的文件名和行号
            filename = info.filename # 获取文件名
            lineno = info.lineno # 获取行号
        # used as function call 
        # 在这里仅适用于普通的函数调用注册方式，类似some_registry.register("my_func", my_function)
        # 如果是装饰器模式这里不会调用
        if module is not None:
            assert module_name is not None
            # 将模块注册到字典中
            Registry._register_generic(self, module_name, module, force_overwrite)
            if _DI_ENGINE_REG_TRACE_IS_ON:
                # 如果开启了跟踪，那么记录注册模块的文件名和行号
                self.__trace__[module_name] = (filename, lineno)
            return

        # used as decorator 用于装饰器 @some_registry.register("my_func")的写法
        def register_fn(fn: Callable) -> Callable:
            '''
            fn： 在装饰器的使用时，这里才是正式传入函数或者类对象
            '''
            if module_name is None:
                name = fn.__name__ # 如果在装饰器使用时没有传入名字，即直接调用@some_registry.register，则使用的是对象本身的名字
            else:
                name = module_name
            # 将模块注册到字典中
            Registry._register_generic(self, name, fn, force_overwrite)
            if _DI_ENGINE_REG_TRACE_IS_ON:
                self.__trace__[name] = (filename, lineno)
            return fn   # 返回原函数（装饰器规范）

        return register_fn

    @staticmethod
    def _register_generic(module_dict: dict, module_name: str, module: Callable, force_overwrite: bool = False) -> None:
        """
        Overview:
            Register the module.
        Arguments:
            - module_dict (:obj:`dict`): The dict to store the module. 用于存储在注册模块的字典
            - module_name (:obj:`str`): The name of the module. 注册的模块名称
            - module (:obj:`Callable`): The module to be registered. 注册的模块,可以是函数或者类
            - force_overwrite (:obj:`bool`): Whether to overwrite the module with the same name.
        """

        if not force_overwrite:
            # 如果不是强制覆盖模式，则必须要保证是新注册的模块名称
            assert module_name not in module_dict, module_name
        # 将模块存储到字典中
        module_dict[module_name] = module

    def get(self, module_name: str) -> Callable:
        """
        Overview:
            Get the module.
        Arguments:
            - module_name (:obj:`str`): The name of the module.
        """

        return self[module_name]

    def build(self, obj_type: str, *obj_args, **obj_kwargs) -> object:
        """
        Overview:
            Build the object.
        Arguments:
            - obj_type (:obj:`str`): The type of the object. 注册类型，比如子进程
            - obj_args (:obj:`Tuple`): The arguments passed to the object. 创建方法，比如创建环境的函数
            - obj_kwargs (:obj:`Dict`): The keyword arguments passed to the object. 相关参数，比如创建环境的参数
        """

        try:
            build_fn = self[obj_type] # 这里可以这么使用因为继承了字典类型
            return build_fn(*obj_args, **obj_kwargs)
        except Exception as e:
            # get build_fn fail
            if isinstance(e, KeyError):
                raise KeyError("not support buildable-object type: {}".format(obj_type))
            # build_fn execution fail
            global _innest_error
            if _innest_error:
                argspec = inspect.getfullargspec(build_fn)
                message = 'Hint: for {}(alias={})'.format(build_fn, obj_type)
                message += '\n\nExpected args are:\n {}\nGiven arguments keys are:\n{}\n'.format(
                    argspec, obj_kwargs.keys()
                )
                print(message)
                _innest_error = False
            raise e

    def query(self) -> Iterable:
        """
        Overview:
            all registered module names.
        """

        return self.keys()

    def query_details(self, aliases: Optional[Iterable] = None) -> OrderedDict:
        """
        Overview:
            Get the details of the registered modules.
        Arguments:
            - aliases (:obj:`Optional[Iterable]`): The aliases of the modules.
        """

        assert _DI_ENGINE_REG_TRACE_IS_ON, "please exec 'export DIENGINEREGTRACE=ON' first"
        if aliases is None:
            aliases = self.keys()
        return OrderedDict((alias, self.__trace__[alias]) for alias in aliases)
