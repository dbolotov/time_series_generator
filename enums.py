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
    NONE = "None"
    LINEAR = "Linear"
    QUADRATIC = "Quadratic"
    EXPONENTIAL = "Exponential"

class SeasonalityType(str, Enum):
    NONE = "None"
    SINE = "Sine"
    SAWTOOTH = "Sawtooth"