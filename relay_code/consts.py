SAMPLES_IN_5_SECONDS_AT_500HZ = 2500

SAMPLES_IN_10_SECONDS_AT_500HZ = 5000

LEAD_2_INDEX = {
    'I': 0,
    'II': 1,
    'III': 2,
    'aVR': 3,
    'aVL': 4,
    'aVF': 5,
    'V1': 6,
    'V2': 7,
    'V3': 8,
    'V4': 9,
    'V5': 10,
    'V6': 11
}



INDEX_2_LEAD = {v: k for k, v in LEAD_2_INDEX.items()}


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

MINIMAL_IMPROVEMENT_PRETRAINING = 1e-3

MINIMAL_IMPROVEMENT_FINETUNING = 1e-3

DROPOUT_DYNAMIC_REG_FACTOR = 0.05

WEIGHT_DECAY_DYNAMIC_REG_FACTOR = 5

S = 20 # num of tokens/nodes per lead
DIM_IT1 = 39
DIM_IT2 = 768 # todo

