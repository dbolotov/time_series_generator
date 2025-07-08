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
    QUADRATIC = "Quadratic"
    EXPONENTIAL = "Exp"
    NONE = "None"

class SeasonalityType(str, Enum):
    SINE = "Sine"
    SAWTOOTH = "Sawtooth"
    NONE = "None"

class AnomalyType(str, Enum):
    NONE = "None"
    VALUE_SPIKE = "Value Spike"