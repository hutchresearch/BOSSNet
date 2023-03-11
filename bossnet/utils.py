from dataclasses import dataclass

# Dataclass to hold metadata stats.
@dataclass(frozen=True)
class DataStats:
    """
    A dataclass that holds metadata statistics related to a single value.

    Attributes:
    - MEAN: float, the mean value of the value.
    - STD: float, the standard deviation of the value.
    - PNMAX: float, the post-normalization maximum value of the value.
    - PNMIN: float, the post-normalization minimum value of the value.
    """
    MEAN: float
    STD: float
    PNMAX: float
    PNMIN: float