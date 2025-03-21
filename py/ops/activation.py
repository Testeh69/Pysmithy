from typing import Union, Callable
import smithy as sm

def validate_and_apply(function: Callable[[float], float]) -> Callable[[Union[int, float]], float]:
    def wrapper(arg: Union[int, float]) -> float:
        if not isinstance(arg, (int, float)):
            raise TypeError(f"Expected int or float, got {type(arg).__name__}")
        return function(arg)  # Ne pas reconvertir ici, la fonction interne le gère déjà
    return wrapper

@validate_and_apply
def relu(arg: float) -> float:
    return sm.relu(arg)

@validate_and_apply
def sigmoid(arg: float) -> float:
    return sm.sigmoid(arg)

@validate_and_apply
def tanh(arg: float) -> float:
    return sm.tanh(arg)

@validate_and_apply
def cosh(arg: float) -> float:
    return sm.cosh(arg)



