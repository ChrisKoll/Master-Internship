"""
Contains the constant values for the HANDLER module.
"""

### HANDLER

# Arg ANNDATA
ARG_ANNDATA = "anndata"

# Arg COMMAND
ARG_COMMAND = "command"
ARG_COMMAND_EXPECTED1 = "subset"
ARG_COMMAND_EXPECTED2 = "normalize"
ARG_COMMAND_EXPECTED3 = "train"

## SUBSET command

# Arg ROWS
ARG_ROWS = "rows"

# Arg COLUMNS
ARG_COLUMNS = "cols"

# Arg NO-SHUFFLE
ARG_NO_SHUFFLE = "no-shuffle"

## NORMALIZE

# Arg METHOD
ARG_METHOD = "method"

## TRAIN command

# Arg MODEL
ARG_MODEL = "model"

M_CONFIG_SECTION1 = "ModelType"
M_CONFIG_S1_KEY1 = "Type"
M_CONFIG_S1_K1_EXPECTED1 = "standard"
M_CONFIG_S1_K1_EXPECTED2 = "variational"
M_CONFIG_SECTION2 = "ModelStructure"
M_CONFIG_S2_KEY1 = "sizeinputlayer"
M_CONFIG_S2_KEY2 = "sizelayer1"
M_CONFIG_S2_KEY3 = "sizelayer2"
M_CONFIG_S2_KEY4 = "sizelayer3"
M_CONFIG_S2_KEY5 = "sizelatentspace"

# Arg HYPERPARAMS
ARG_HYPERPARAMS = "hyperparams"
H_CONFIG_SECTION = "Hyperparameters"

# Arg Subset
ARG_SUBSET = "subset"

# Arg NORMALIZATION
ARG_NORMALIZATION = "normalization"

## ERRORS
VALUE_ERR_IFP = "INVALID FILE PATH"
VALUE_ERR_IK = "INVALID KEY"
VALUE_ERR_MA = "MISSING ARGUMENT"
VALUE_ERR_UC = "UNKNOWN COMMENT"
