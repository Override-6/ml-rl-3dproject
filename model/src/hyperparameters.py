# ------------------------
# Hyperparameters
# ------------------------
from src.data import Input

# Simulation
LASERS_PER_PLAYER = 1
NB_COMPONENT_TYPES = 5
# number of binary action outputs
# + the directional laser's rotation
NUM_ACTIONS = len(Input) + 1
SIMULATION_HZ = 10 # number of ticks per seconds
SIMULATION_TIME = 25 # number of simulated seconds

# Model-specific
FEATURE_DIM = 128
LSTM_UNITS = 64
SEQ_LEN = SIMULATION_HZ * SIMULATION_TIME  # rollout horizon (timesteps collected before update). is simulation's hz times 5 seconds
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64
CLIP_EPS = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.05
LR = 1e-3
GAMMA = 0.99
LAMBDA = 0.95

# Collapse detection / recovery:

# Number of required learning steps that did not trigger a collapse warning
# before going back to default model hyperparameters
REQUIRED_STABLE_UPDATES_AFTER_COLLAPSE = 100
LR_MULTIPLIER_PER_COLLAPSE_DETECTION = 0.25
BACKUP_RATE = 200 # store model weights every BACKUP_RATE steps