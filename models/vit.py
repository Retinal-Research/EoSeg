from typing import Optional
import torch
import torch.nn as nn

import timm
from transformers import AutoModel

from models.vit_rvsa_mtp import RVSA_MTP


class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.ckpt_path = ckpt_path

        name_upper = backbone_name.upper()

        if name_upper.startswith("MTP"):
            self.backbone = self.build_mtp_backbone(
                backbone_name=backbone_name,
                img_size=img_size,
                patch_size=patch_size,
                ckpt_path=ckpt_path,
            )
        elif "/" in backbone_name:
            self.backbone = self.transformers_to_timm(
                AutoModel.from_pretrained(backbone_name),
                img_size,
            )
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=ckpt_path is None,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=0,
            )

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def build_mtp_backbone(
        self,
        backbone_name: str,
        img_size: tuple[int, int],
        patch_size: int = 16,
        ckpt_path: Optional[str] = None,
    ):
        """
        Build MTP backbone.
        Accepted names example:
            - MTP_B
            - MTP_L
            - MTP
        """
        name_upper = backbone_name.upper()

        if "L" in name_upper:
            embed_dim = 1024
            depth = 24
            num_heads = 16
            interval = 6
            out_indices = [7, 11, 15, 23]
        else:
            embed_dim = 768
            depth = 12
            num_heads = 12
            interval = 3
            out_indices = [3, 5, 7, 11]

        backbone = RVSA_MTP(
            img_size=img_size,
            in_chans=3,
            patch_size=patch_size,
            drop_path_rate=0.1,
            out_indices=out_indices,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            use_checkpoint=False,
            use_abs_pos_emb=True,
            interval=interval,
            use_rel_pos_bias=True,
            pretrained="/scratch/xinli38/eomt/models/last_vit_l_rvsa_ss_is_rd_pretrn_model_encoder.pth",
        )

        backbone.init_weights()
        return backbone

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        backbone.blocks = backbone.layer

        del (
            backbone.patch_embed.mask_token,
            backbone.embeddings,
            backbone.layer,
        )

        return backbone
