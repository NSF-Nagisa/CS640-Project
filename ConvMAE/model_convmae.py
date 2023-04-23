# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block, Attention
from timm.models.layers import DropPath
from pos_embedding import get_2d_sincos_pos_embed

class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMaeBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.conv = nn.Conv2d(dim, dim, 1)
        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim) # 5 * 5 depthwise convolution
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask):
        x = x + self.drop_path(self.conv(self.attn(mask * self.norm1(x))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 

class ConvMae(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # ConvMAE encoder specifics
        # stage 1
        # 224 * 224 *3 -> 56 * 56 * 256
        self.patch_embed1 = PatchEmbed(img_size=224, patch_size=4, in_chans=3, embed_dim=256)
        self.blocks1 = nn.ModuleList([
            ConvMaeBlock(
                dim=256, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer
            ) for i in range(2)
        ])
        # shrink to 1/4 of the original
        self.stage1_output = nn.Conv2d(256, 1024, 4, stride=4)
        # stage 2
        # 56 * 56 * 256 -> 28 * 28 * 512
        self.patch_embed2 = PatchEmbed(img_size=56, pathch_size=2, in_chans=256, embed_dim=512)
        self.blocks2 = nn.ModuleList([
            ConvMaeBlock(
                dim=384, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer
            ) for i in range(2)
        ])
        # shrink to 1/2 of the original
        self.stage2_output = nn.Conv2d(512, 1024, 2, stride=2)
        # stage 3
        # we reduce the number of layers to reduce the computation burden of training
        self.patch_embed3 = PatchEmbed(img_size=28, patch_size=2, in_chans=512, embed_dim=1024)
        self.blocks3 = nn.ModuleList([
            Block(dim=1024, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer
            ) for i in range(4)
        ])
        num_patches = self.patch_embed3.num_patches
        # remove the class token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # ConvMAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (4 * 2 * 2)**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 16
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        L = self.patch_embed3.num_patches # length should be the same with the length after shrinkage
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] 
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        # x = self.patch_embed(x)

        # add pos embed w/o cls token

        # masking: length -> length * mask_ratio
        # block-wise masking here
        # each stage shares the mask but it should be reshape in each stage
        ids_keep, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask = mask.reshape(-1, 14, 14).unsqueeze(-1) # (batch_size, 14, 14, 1) base mask
        # mask for stage1, 56 * 56
        mask1 = mask.repeat(1, 1, 1, 16).reshape(-1, 14, 14, 4, 4), # (batchsize, 14, 14, 4, 4) match the shape of patch_embed1
        mask1 = mask1.permute(0, 1, 3, 2, 4).reshape(x.shape[0], 56, 56).unsqueeze(1) # (batch_size, 1, 56, 56)
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, 1 - mask1)
        stage1_embed = self.stage1_output(x).flatten(2)
        stage1_embed = torch.gather(stage1_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[2]))
        # mask for stage2, 28 * 28
        mask2 = mask.repeat(1, 1, 1, 4).reshape(-1, 14, 14, 2, 2), # (batchsize, 14, 14, 2, 2) match the shape of patch_embed2
        mask2 = mask2.permute(0, 1, 3, 2, 4).reshape(x.shape[0], 28, 28).unsqueeze(1) # (batch_size, 1, 28, 28)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, 1 - mask2)
        stage2_embed = self.stage2_output(x).flatten(2)
        stage2_embed = torch.gather(stage2_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[2]))
        # mask for stage3, 14 * 14
        x = self.patch_embed3(x)
        x = x.flatten(2)
        x = x + self.pos_embed
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks3:
            x = blk(x)
        x = x + stage1_embed + stage2_embed
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # # remove cls token
        # x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = ConvMae(
        embed_dim=768, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
convmae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
