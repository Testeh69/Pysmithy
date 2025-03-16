from typing import Union
import smithy  



def relu(arg: Union[int, float]) -> float:
    if not isinstance(arg, (int, float)):
        raise TypeError(f"Expected int or float, got {type(arg).__name__}")
    return smithy.activation.relu(float(arg))



def sigmoid(arg:Union[int,float])->float:
    if not isinstance(arg,(int, float)):
        raise TypeError (f"Expected int or float, got {type(arg).__name__}")
    return smithy.activation.sigmoid(float(arg))


def tanh(arg:Union[int,float])->float:
    if not isinstance(arg,(int, float)):
        raise TypeError (f"Expected int or float, got {type(arg).__name__}")
    return smithy.activation.tanh(float(arg))

def cosh(arg:Union[int,float])->float:
    if not isinstance(arg,(int, float)):
        raise TypeError (f"Expected int or float, got {type(arg).__name__}")
    return smithy.activation.cosh(float(arg))



