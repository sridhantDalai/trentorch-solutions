import numpy as np
import statistics as stat

def skewness(x: np.ndarray) -> float:
    """
    Pearson's moment coefficient of skewness: the average CUBED
    z-score of every value.

        skew(x) = mean(((x - mean(x)) / std(x))^3)

    Cubing (rather than squaring, like variance does) preserves sign:
    a distribution with a long right tail gives a positive number, a
    long left tail gives a negative one, a symmetric distribution
    gives (near) zero.
    """
    
    mean = stat.mean(x)
    std = stat.pstdev(x)
    z = (x - mean) / std
    return float(stat.mean(z ** 3))


def summarize_distribution(x: np.ndarray) -> dict:
    """
    Returns {"mean": ..., "median": ..., "std": ..., "skew": ...},
    the standard quick summary of a feature's distribution. `skewness`
    is already provided above.
    """
    res = {
        "mean" : stat.mean(x),
        "median" : stat.median(x),
        "std" : stat.pstdev(x), 
        "skew" : skewness(x)
    }
    return res
