"""
Contains all constant values
"""

# === analyst.py ===
OBS_DONOR = "donor"
PLOT_TITLE_DONOR_DIST = "Sample Distribution per Donor"
OBS_CELL_TYPE = "cell_type"
PLOT_TITLE_CELL_TYPE_DIST = "Sample Distribution per Cell Type"
LABEL1_0_EXP = "0 Expression Genes"
LABEL2_0_EXP = "Other Genes"
PLOT_TITLE_0_EXP = "Expression Overview"
FILE_DESTINATION = "tmp"

# === Analysis ===
NUMBER_COMPONENTS = 4

# === Model ===
# Layer size
# 33538 (input size) * 1/3
SIZE_INPUT_LAYER = 33538
SIZE_LAYER_ONE = 10062
SIZE_LAYER_TWO = 3018
SIZE_LAYER_THREE = 906
SIZE_LATENT_SPACE = 200

# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.01


# === Strings ===
PROGRAM_NAME = "Classification Heart Model - CHM"
PROGRAM_DESCRIPTION = "Provides methods to analyze or normalize data, used for training a classification model."
