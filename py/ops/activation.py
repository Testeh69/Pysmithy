from typing import Union
import smithy  as sm



def relu(arg: Union[int, float]) -> float:
    if not isinstance(arg, (int, float)):
        raise TypeError(f"Expected int or float, got {type(arg).__name__}")
    return sm.relu(float(arg))



def sigmoid(arg:Union[int,float])->float:
    if not isinstance(arg,(int, float)):
        raise TypeError (f"Expected int or float, got {type(arg).__name__}")
    return sm.sigmoid(float(arg))


def tanh(arg:Union[int,float])->float:
    if not isinstance(arg,(int, float)):
        raise TypeError (f"Expected int or float, got {type(arg).__name__}")
    return sm.tanh(float(arg))

def cosh(arg:Union[int,float])->float:
    if not isinstance(arg,(int, float)):
        raise TypeError (f"Expected int or float, got {type(arg).__name__}")
    return sm.cosh(float(arg))




