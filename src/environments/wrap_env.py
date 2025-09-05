import importlib
from typing import List, Dict, Any, Callable
import gymnasium as gym

def _import_class_from_string(path: str) -> Callable:
    """
    根据字符串路径动态导入一个类。

    :param path: 格式为 "module.path:ClassName" 的字符串。
    :return: 导入的类对象。
    """
    try:
        module_path, class_name = path.split(':')
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        # 抛出更明确的错误，方便调试
        raise ImportError(f"无法从路径 '{path}' 导入类。请确保路径和类名正确，并且模块在PYTHONPATH中。错误: {e}")

def create_wrapper_from_config(
    wrapper_configs: List[Dict[str, Any]]
) -> Callable[[gym.Env], gym.Env]:
    """
    解析来自配置文件(YAML/JSON)的 wrapper 列表，并返回一个单一的可调用 wrapper 函数。

    :param wrapper_configs: 从配置文件中读取的 wrappers 配置列表。
                          例如: [{'name': 'path:Class', 'kwargs': {...}}, ...]
    :return: 一个接受环境作为输入的函数，该函数返回被所有 wrapper 包装过的环境。
             如果 wrapper_configs 为空或 None，则返回一个什么都不做的函数。
    """
    if not wrapper_configs:
        # 如果没有配置 wrapper，返回一个直接返回原环境的函数
        return lambda env: env

    # 1. 将配置中的字符串路径转换为实际的类对象
    parsed_wrappers = []
    for config in wrapper_configs:
        wrapper_path = config.get("name")
        if not wrapper_path:
            raise ValueError("Wrapper 配置中必须包含 'name' 字段。")
        
        wrapper_kwargs = config.get("kwargs", {}) # 如果kwargs不存在，默认为空字典
        wrapper_class = _import_class_from_string(wrapper_path)
        parsed_wrappers.append((wrapper_class, wrapper_kwargs))

    # 2. 创建一个函数，该函数将按顺序应用所有解析出的 wrapper
    def _wrapper(env: gym.Env) -> gym.Env:
        """
        将解析后的 wrapper 列表依次应用于环境。
        """
        for w_class, w_kwargs in parsed_wrappers:
            env = w_class(env, **w_kwargs)
        return env

    return _wrapper