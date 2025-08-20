from .base import BaseDatasetGenerator
from .gpqa import GPQAGenerator
from .gsm8k import GSM8KGenerator
from .mmlu import MMLUMathGenerator
from .mediq import iCRAFTGenerator, iMEDQAGenerator
from .aime import AIMEGenerator
from .mip import MIPGenerator

def get_dataset_generator(name: str, **kwargs):
    """
    Returns the dataset generator class based on the provided name.
    """
    if name == "gpqa":
        return GPQAGenerator(**kwargs)
    elif name == "gsm8k":
        return GSM8KGenerator(**kwargs)
    elif name == "mmlu":
        return MMLUMathGenerator(**kwargs)
    elif name == "icraft":
        return iCRAFTGenerator(**kwargs)
    elif name == "imedqa":
        return iMEDQAGenerator(**kwargs)
    elif name == "aime":
        return AIMEGenerator(**kwargs)
    elif name == "mip-formula":
        return MIPGenerator(dataset="formula", **kwargs)
    elif name == "mip-math500":
        return MIPGenerator(dataset="math500", **kwargs)
    elif name == "mip-gsm8k":
        return MIPGenerator(dataset="gsm8k", **kwargs)
    elif name == "mip-svamp":
        return MIPGenerator(dataset="svamp", **kwargs)
    else:
        raise ValueError(f"Unknown dataset generator: {name}")