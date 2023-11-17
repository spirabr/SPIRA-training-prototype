from enum import Enum


class ClassName(Enum):
    CONTROL_CLASS = 0
    PATIENT_CLASS = 1


class OperationMode(Enum):
    TRAIN = 0
    TEST = 1


class OptimizerCategory(Enum):
    ADAM = 'adam'
    ADAMW = 'adamw'
    RADAM = 'radam'
