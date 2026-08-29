"""Download (if needed) and load the three pretrained PnP denoisers.

Weight files, in `models/`:
- DnCNN   (CNN, 0.7M parameters)              -> dncnn_sigma2_gray.pth (2.6 MB)
- SwinIR  (Swin Transformer, 11.5M parameters)-> swinir_gray_dn15_fp16.pth (22 MB)
- DRUNet  (CNN U-Net, 32M parameters)         -> drunet_deepinv_gray_finetune_26k.pth (124 MB)

The released SwinIR checkpoint weighs 117 MB, but 73 MB of it are attention-mask/index
buffers that the network recomputes for the actual image size: only the 11.5M learned
weights matter. They are kept in float16, which gives the 22 MB file (the reconstruction
changes by less than 0.2%).
"""
import os

import torch
import deepinv as dinv

DNCNN_URL = "https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_gray.pth?download=true"
DRUNET_URL = "https://huggingface.co/deepinv/drunet/resolve/main/drunet_deepinv_gray_finetune_26k.pth?download=true"
SWINIR_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth"
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))   # the weights live next to this file


def fetch(path, url):
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        torch.hub.download_url_to_file(url, path, progress=True)
    return path


def load_swinir(device):
    slim = os.path.join(MODELS_DIR, "swinir_gray_dn15_fp16.pth")
    if not os.path.exists(slim):
        full = fetch(os.path.join(MODELS_DIR, "swinir_gray_dn15.pth"), SWINIR_URL)
        ck = torch.load(full, map_location="cpu", weights_only=True)["params"]
        ck = {k: v.half() for k, v in ck.items() if "attn_mask" not in k and "relative_position_index" not in k}
        torch.save(ck, slim)
    model = dinv.models.SwinIR(in_chans=1, pretrained=None)
    state = {k: v.float() for k, v in torch.load(slim, map_location="cpu", weights_only=True).items()}
    model.load_state_dict(state, strict=False)
    return model.to(device), slim


def load_denoisers(device, names=("DnCNN", "SwinIR", "DRUNet")):
    """Load the requested denoisers (all three by default); e.g. names=("DnCNN", "DRUNet") skips the slow SwinIR."""
    denoisers, paths = {}, {}
    if "DnCNN" in names:
        paths["DnCNN"] = fetch(os.path.join(MODELS_DIR, "dncnn_sigma2_gray.pth"), DNCNN_URL)
        denoisers["DnCNN"] = dinv.models.DnCNN(in_channels=1, out_channels=1, pretrained=paths["DnCNN"], device=device)
    if "SwinIR" in names:
        denoisers["SwinIR"], paths["SwinIR"] = load_swinir(device)
    if "DRUNet" in names:
        paths["DRUNet"] = fetch(os.path.join(MODELS_DIR, "drunet_deepinv_gray_finetune_26k.pth"), DRUNET_URL)
        denoisers["DRUNet"] = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained=paths["DRUNet"], device=device)
    for name, model in denoisers.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:<8} {n_params / 1e6:5.1f}M parameters | weights file {os.path.getsize(paths[name]) / 2**20:5.1f} MB")
    return denoisers
