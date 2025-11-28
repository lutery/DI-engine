# self[obj_type]是什么样子的使用方式？它是怎么调用的？能讲解一下吗
from ding.utils.registry import Registry

## 1. 创建注册表
env_registry = Registry()

## 2. 注册函数（方式一：直接注册）
def create_lunar_lander(**kwargs):
    print(f"创建 LunarLander 环境，参数: {kwargs}")
    return "LunarLander环境实例"

env_registry.register("lunarlander", create_lunar_lander)

## 3. 注册函数（方式二：装饰器）
@env_registry.register("cartpole")
def create_cartpole(**kwargs):
    print(f"创建 CartPole 环境，参数: {kwargs}")
    return "CartPole环境实例"

## 4. 使用 self[obj_type] 访问
## 在 Registry 内部：
## build_fn = self[obj_type]  ## 获取注册的函数

## 5. 外部调用 build 方法
env1 = env_registry.build("lunarlander", env_id="LunarLander-v2", seed=0)
## 输出: 创建 LunarLander 环境，参数: {'env_id': 'LunarLander-v2', 'seed': 0}

env2 = env_registry.build("cartpole", max_steps=500)
## 输出: 创建 CartPole 环境，参数: {'max_steps': 500}

## 6. 也可以直接访问（不推荐，应使用 build）
build_fn = env_registry["lunarlander"]  ## 等同于 self[obj_type]
env3 = build_fn(env_id="LunarLander-v2")

# `register_fn` 的作用详解

`register_fn` 是一个**装饰器函数**，它的存在是为了支持 `@register` 装饰器语法。让我详细解释一下：

## 两种注册方式

Registry 的 `register` 方法支持**两种不同的使用方式**：

### 方式1：直接函数调用（module 参数不为 None）

```python
# 直接调用注册
def my_function():
    print("Hello")

some_registry.register("my_func", my_function)
# 执行这段代码时：module 参数 = my_function（不为None）
# 直接在 register 方法内完成注册，不需要返回 register_fn
```

### 方式2：装饰器语法（module 参数为 None）

```python
# 使用装饰器注册
@some_registry.register("my_func")  # ← 这里调用 register("my_func")
def my_function():                   # ← 返回的 register_fn 会接收这个函数
    print("Hello")

# 等价于：
# temp_decorator = some_registry.register("my_func")  # 返回 register_fn
# my_function = temp_decorator(my_function)            # register_fn 接收函数并注册
```

## 执行流程对比

````python
from ding.utils.registry import Registry

# 创建注册表
env_registry = Registry()

# ============================================================
# 方式1：直接调用（module 不为 None）
# ============================================================
def create_env_v1():
    return "环境实例v1"

# 调用时传入了 module 参数
env_registry.register("env_v1", create_env_v1)

# 执行流程：
# 1. 进入 register 方法
# 2. module is not None → 为 True
# 3. 直接执行 Registry._register_generic(...)
# 4. 注册完成，return（返回 None）

# ============================================================
# 方式2：装饰器（module 为 None，只传 module_name）
# ============================================================
@env_registry.register("env_v2")  # ← 步骤1: 调用 register("env_v2")
def create_env_v2():               # ← 步骤4: register_fn 接收这个函数
    return "环境实例v2"

# 执行流程：
# 步骤1: 调用 register("env_v2")
#   - module_name = "env_v2"
#   - module = None（默认值）
#   
# 步骤2: 检查 module is not None → 为 False，跳过直接注册
#
# 步骤3: 执行到 "used as decorator" 部分
#   - 定义内部函数 register_fn
#   - return register_fn（返回这个装饰器函数）
#   
# 步骤4: Python 装饰器机制自动调用
#   - register_fn(create_env_v2) 被调用
#   - 在 register_fn 内部执行注册：
#     Registry._register_generic(self, "env_v2", create_env_v2)
#   - return create_env_v2（返回原函数）

# ============================================================
# 方式3：无参数装饰器（使用函数名作为注册名）
# ============================================================
@env_registry.register()  # ← 不传名称，使用函数名
def create_env_v3():
    return "环境实例v3"

# 执行流程：
# 1. register() 被调用，module_name = None
# 2. 返回 register_fn
# 3. register_fn(create_env_v3) 被调用
# 4. 在 register_fn 内：name = fn.__name__ = "create_env_v3"
# 5. 使用函数名 "create_env_v3" 进行注册

# ============================================================
# 测试所有注册方式
# ============================================================
print("注册的模块:", list(env_registry.query()))
# 输出: ['env_v1', 'env_v2', 'create_env_v3']

# 使用注册的模块
env1 = env_registry.build("env_v1")
env2 = env_registry.build("env_v2")
env3 = env_registry.build("create_env_v3")

print(env1)  # 输出: 环境实例v1
print(env2)  # 输出: 环境实例v2
print(env3)  # 输出: 环境实例v3
````

## 为什么需要 register_fn？

```python
def register(self, module_name=None, module=None, force_overwrite=False):
    # 情况1: 直接调用 - register("name", func)
    if module is not None:
        # 直接注册，立即完成
        Registry._register_generic(self, module_name, module, force_overwrite)
        return  # 直接返回，不需要装饰器
    
    # 情况2: 装饰器调用 - @register("name")
    # 此时 register("name") 已经执行完了，但函数定义还没到
    # 所以需要返回一个函数（register_fn），等待接收被装饰的函数
    def register_fn(fn):
        # 这个函数会在 Python 解析到函数定义时被自动调用
        # fn 就是被装饰的函数
        name = module_name if module_name else fn.__name__
        Registry._register_generic(self, name, fn, force_overwrite)
        return fn  # 返回原函数（装饰器规范）
    
    return register_fn  # 返回装饰器函数
```

## Python 装饰器原理

```python
# 装饰器语法
@decorator
def func():
    pass

# 等价于
def func():
    pass
func = decorator(func)  # decorator 接收函数，返回新函数（或原函数）

# 带参数的装饰器
@decorator("arg")
def func():
    pass

# 等价于
def func():
    pass
temp = decorator("arg")  # 先调用 decorator("arg")，返回真正的装饰器
func = temp(func)         # 再用返回的装饰器包装函数
```

## 总结

1. **`register_fn` 不是重复注册**，它是为了支持装饰器语法而存在的
2. **直接调用时**（`module != None`）：立即注册，不返回 `register_fn`
3. **装饰器调用时**（`module == None`）：返回 `register_fn`，等待接收被装饰的函数
4. **`register_fn` 的作用**：作为真正的装饰器函数，接收被装饰的函数并完成注册

这种设计模式叫做**"装饰器工厂"**，允许同一个函数既能直接调用，又能作为装饰器使用。