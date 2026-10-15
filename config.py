import os


# =========================
# Dataset
# =========================

DATA_DIR = (
    r"F:\\VSCode\\Code\\Python\\DuNets-RMA-LPD"
    r"\\datasets-600\\circle4_h00.07_Windows"
)


# processed dataset
PROCESSED_DIR = "./processed"


# =========================
# GN
# =========================

GN_SAVE_DIR = "./processed/gn"


# =========================
# Training
# =========================

CHECKPOINT_DIR = "./checkpoints"


NUM_ITER = 10


BATCH_SIZE = 4

EPOCHS = 50

LR = 1e-3


DEVICE = "cuda"