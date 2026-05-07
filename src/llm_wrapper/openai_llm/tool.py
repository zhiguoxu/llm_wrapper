from __future__ import annotations

from inspect import signature
from typing import Callable, Type, Any, TypeVar, Generic, cast, Literal

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from llm_wrapper.utils import filter_kwargs_by_pydantic, is_pydantic_class

ToolOutput = TypeVar("ToolOutput")


class Tool(BaseModel, Generic[ToolOutput]):
    function: Callable[..., ToolOutput] | dict
    args_schema: Type[BaseModel] | None

    def __init__(self,
                 function: Callable[..., ToolOutput] | dict,
                 type: Literal['function'] = 'function',  # 兼容 openai_llm 的格式
                 args_schema: Type[BaseModel] | None = None,
                 name: str | None = None,
                 description: str | None = None):
        if isinstance(function, Callable):
            args_schema = args_schema or create_schema_from_function(function)
            args_schema.__name__ = name or args_schema.__name__
            args_schema.__doc__ = description or function.__doc__ or args_schema.__doc__
        kwargs = filter_kwargs_by_pydantic(self, locals())
        super().__init__(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        assert isinstance(self.function, Callable)
        return self.function(*args, **kwargs)

    @property
    def name(self):
        return self.args_schema.__name__

    @property
    def description(self):
        return self.args_schema.__doc__


def tool(*args: str | Callable[..., ToolOutput],
         args_schema: Type[BaseModel] | None = None
         ) -> Tool[ToolOutput] | Callable[[Callable[..., ToolOutput]], Tool[ToolOutput]]:
    tool_name: str | None = None

    def make_tool(func: Callable[..., ToolOutput]) -> Tool[ToolOutput]:
        name = tool_name or func.__name__
        if func.__doc__ is None:
            raise ValueError(f"Function【{name}】must have a docstring as it's description.")

        return Tool(function=func,
                    args_schema=args_schema,
                    name=name)

    if len(args) == 1 and isinstance(args[0], str):
        tool_name = args[0]
        return make_tool

    if len(args) == 1 and callable(args[0]):
        return make_tool(args[0])

    if len(args) == 0:
        return make_tool

    raise TypeError(f"Arguments type error: {args}")


def create_schema_from_function(func: Callable[..., Any]) -> Type[BaseModel]:
    """Create schema from function."""

    fields = {}
    params = signature(func).parameters
    for param_name in params:
        param_type = params[param_name].annotation
        param_default = params[param_name].default

        if param_type is params[param_name].empty:
            param_type = Any

        if param_default is params[param_name].empty:
            # Required field
            fields[param_name] = (param_type, FieldInfo())
        elif isinstance(param_default, FieldInfo):
            # Field with pydantic.Field as default value
            fields[param_name] = (param_type, param_default)
        else:
            fields[param_name] = (param_type, FieldInfo(default=param_default))

    return create_model(func.__name__, **fields)  # type: ignore[call-overload]


ToolLike = Tool | Callable | Type[BaseModel]


def to_tool(tool_like: ToolLike) -> Tool:
    if isinstance(tool_like, Tool):
        return tool_like

    if callable(tool_like):
        if is_pydantic_class(tool_like) and tool_like.__doc__ is None:
            tool_like.__doc__ = f"{tool_like.__name__}'s init function."

        return cast(Tool, tool(tool_like))

    assert False
