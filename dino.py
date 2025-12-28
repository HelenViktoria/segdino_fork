import torch


def load_dino_backbone(repo_dir, dino_size, dino_ckpt):
    if dino_size == "b":
        backbone = torch.hub.load(repo_dir, 'dinov3_vitb16', source='local', weights=dino_ckpt)
    else:
        backbone = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=dino_ckpt)
    return backbone