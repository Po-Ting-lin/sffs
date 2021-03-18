from enum import Enum


class GammaMode(Enum):
    OFF = 0
    ADAPTIVE_MODE = 1
    CONSTANT_MODE = 2


class SvmClassificationMode(Enum):
    PRE_TRAINED = 0
    TRAIN_WITH_ADAPTIVE = 1
    TRAIN_WITH_DEFAULT = 2
