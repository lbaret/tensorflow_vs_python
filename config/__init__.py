import os

# Isolated datasets names
CIFAR_DATA_NAME = 'cifar'
MNIST_DATA_NAME = 'mnist'
FASHION_DATA_NAME = 'fashion'

# Aggregation of datasets names
DATA_NAMES_LIST = [
    CIFAR_DATA_NAME,
    MNIST_DATA_NAME,
    FASHION_DATA_NAME
]


# Paths
WORK_DIRECTORY = os.path.abspath(os.path.dirname(__name__))
DATA_DIRECTORY = os.path.join(WORK_DIRECTORY, 'data')

EXP_DIRECTORY = os.path.join(WORK_DIRECTORY, 'experimentations')
TF_EXP_DIRECTORY = os.path.join(EXP_DIRECTORY, 'tensorflow')
PT_EXP_DIRECTORY = os.path.join(EXP_DIRECTORY, 'pytorch')