import inspect
from typing import Callable, Dict, Any, Type, TypeVar

from pydantic import BaseModel

Model = TypeVar('Model', bound='BaseModel')


def get_method_parameters(obj: Callable[..., Any],
                          exclude: set[str] | None = None,
                          exclude_self_cls: bool = True) -> list[str]:
    params = [param for param in inspect.signature(obj).parameters.keys()]
    exclude = exclude or set()
    if exclude_self_cls:
        exclude.update({"self", "cls"})
    return list(filter(lambda item: item not in exclude, params) if exclude else params)


def filter_kwargs_by_method(obj: Callable[..., Any],
                            kwargs: Dict[str, Any],
                            exclude: set[str] | None = None,
                            exclude_none: bool = False) -> Dict[str, Any]:
    params = get_method_parameters(obj)
    exclude = exclude or set()
    return {
        k: v for k, v in kwargs.items()
        if k in params and (k not in exclude) and (not exclude_none or v is not None)
    }


def filter_kwargs_by_pydantic(model_type: Type[Model] | Model,
                              kwargs: Dict[str, Any],
                              exclude: set[str] | None = None,
                              exclude_none: bool = False) -> Dict[str, Any]:
    params = model_type.__fields__.keys()  # type: ignore
    exclude = exclude or set()
    return {
        k: v
        for k, v in kwargs.items()
        if k in params and (k not in exclude) and (not exclude_none or v is not None)
    }


def is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and (
            issubclass(obj, BaseModel) or BaseModel in obj.__bases__
    )


AddableT = TypeVar("AddableT")


def add(a: AddableT, b: AddableT) -> AddableT:
    if a is None or b is None:
        return a or b
    return a + b  # type: ignore[operator]
