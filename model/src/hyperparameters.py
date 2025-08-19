# ------------------------
# Hyperparameters
# ------------------------
from src.data import Input

LASERS_PER_PLAYER = 5
NB_COMPONENT_TYPES = 5
NUM_ACTIONS = len(Input)  # number of binary action outputs
FEATURE_DIM = 128
LSTM_UNITS = 64
SEQ_LEN = 10 * 10  # rollout horizon (timesteps collected before update). is simulation's hz times 5 seconds
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64
CLIP_EPS = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.20
LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
