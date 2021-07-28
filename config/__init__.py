import os

WORK_DIRECTORY = os.path.abspath(os.path.dirname(__name__))
DATA_DIRECTORY = os.path.join(WORK_DIRECTORY, 'data')

EXP_DIRECTORY = os.path.join(WORK_DIRECTORY, 'experimentations')
TF_EXP_DIRECTORY = os.path.join(EXP_DIRECTORY, 'tensorflow')
PT_EXP_DIRECTORY = os.path.join(EXP_DIRECTORY, 'pytorch')