# data loading
PATH_TO_TRAIN_DATA = "data/01_raw/train_chess_data_02.parquet.gzip"
PATH_TO_VAL_DATA = "data/01_raw/val_chess_data.parquet.gzip"

TRAIN_SAMPLES = 20_000_000
VAL_SAMPLES = 450_000

LOAD_BATCH_SIZE = 100_000

#training
BATCH_SIZE = 256
LEARNING_RATE = 0.01
SAMPLES = 20_000_000

NUM_WORKERS = 1

# model settings
N_LAYERS = 3
N_UNITS_PER_LAYER = [1024, 1024, 512]
DROPOUT_PER_LAYER = [0.1, 0.1, 0.1]


SAMPLES = max(TRAIN_SAMPLES, SAMPLES)