import torch


def get_device(cfg_device: str = "auto") -> torch.device:
    """Resolve device string from config to torch.device."""
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)
