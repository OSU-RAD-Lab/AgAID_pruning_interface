def lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation between start and end as t goes from 0 to 1."""
    return start * (1 - t) + end * t


def inverse_lerp(start: float, end: float, value: float) -> float:
    """linear interpolates 0 to 1 as value goes from start to end"""
    return (value-start)/(end-start)


def exerp(start: float, end: float, t: float) -> float:
    """exponential interpolation between start and end based on t."""
    return start*pow(end/start,t)