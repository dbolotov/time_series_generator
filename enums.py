from enum import Enum

class SeriesType(str, Enum):
    NOISE = "Noise"
    OU_PROCESS = "OU Process"
    CUSTOM = "Custom"

class FillMethod(str, Enum):
    NONE = "None"
    FORWARD = "Forward Fill"
    ZERO = "Fill with Zero"

class TrendType(str, Enum):
    LINEAR = "Linear"
    NONE = "None"
    QUADRATIC = "Quadratic"
    EXPONENTIAL = "Exp"

class SeasonalityType(str, Enum):
    SINE = "Sine"
    NONE = "None"
    SAWTOOTH = "Sawtooth"