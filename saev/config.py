from . import modeling

#################
# COMPATIBILITY #
#################


# For compatibility with older (pickled) checkpoints.
# The classes are the same, just named differently.


ViTSAERunnerConfig = modeling.Config
