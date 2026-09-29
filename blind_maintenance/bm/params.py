"""Constants of the environment and controllers (spec sections 2-7).

All numbers are copied from protocol/Blind_Maintenance_Benchmark_RU_v1.md.
"""

T = 1000                       # steps per episode

# --- initial state (section 2) ---
R0 = 18.0
D0 = 0.08
S0 = 0.0
K0 = 999
E0 = 1.0

# --- death (section 2) ---
S_DEATH = 0.56                 # death if S >= 0.56
# death if R <= 0

# --- availability (3.1) ---
P_FLIP_B = 0.04
MU_B = (0.48, 1.10)
SD_A = 0.16
A_MAX = 1.5

# --- wear rate (3.2) ---
M_LEVELS = (0.48, 0.92, 1.65)
STATIC_INDEX = 1
MIN_DWELL = 45
P_SWITCH = 0.020

# --- auxiliary processes (3.3) ---
AR_COEF = 0.90
AR_SD = 0.12
WALK_SD = 0.033
P_H_ZERO = 0.90
H_SD = 0.020
EPS_PROD_SD = 0.025
EPS_SENSOR_SD = 0.050
N_CH = 8

# --- sensors (4) ---
D_CLIP_MAX = 1.35
RAW_CLIP = 1.7

# --- step physics (5) ---
BASE_COST = 0.30
SERVICE_COST = 2.25
SERVICE_R_MIN = 2.75           # = 2.25 + 0.30 + 0.20 (exact in float64)
SERVICE_K_MIN = 5
WORK_COST = 0.12
R_MAX = 52.0
PROD_GAIN = 1.42
SERVICE_D_FACTOR = 0.16
EMA_KEEP = 0.70
EMA_NEW = 0.30
U_MAX = 1.15
SYMPTOM_C = 0.995              # y_t = 1[c < 0.995]

# --- controller C (6) ---
C_E_TRIGGER = 0.965
C_RISK_THRESHOLD = 0.24
C_MIN_EXAMPLES = 18
C_LABEL_DELAY = 4
C_POS_WEIGHT = 2.2
C_L2 = 0.0008
C_LR = 0.13
C_NORM_EPS = 0.8
C_W_CLIP = 4.0
C_LOGIT_CLIP = 12.0
C_DERIV_GAIN = 4.0

# --- comparison strategies (7) ---
FIXED_PERIODS = (5, 6, 7, 8, 9, 10, 12, 15)
OUTPUT_ONLY_THETA = 0.965
ORACLE_D_THRESHOLD = 0.53
THETA_GRID = (0.85, 0.90, 0.94, 0.965, 0.98, 0.995)
L_GRID = (1, 2, 4, 8)

# --- samples (9) ---
DEV_N = 64
DEV_BASE = {"dynamic": 31000000, "static": 32000000}
DEV_STRIDE = 7919
CONF_N = {"dynamic": 512, "static": 256}

# --- analysis (10) ---
N_BOOT = 10000
ANALYSIS_SEED = 20260929       # fixed separate seed for the bootstrap

WORK = 0
SERVICE = 1
