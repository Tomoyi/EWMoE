from functools import partial

import torch
import torch.nn as nn

# from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import PatchEmbed # todo manxin

from timm.models.vision_transformer import Mlp # todo manxin added
from timm.models.layers import DropPath, trunc_normal_  # todo manxin added
from util.pos_embed import get_2d_sincos_pos_embed
from util.pos_embed import get_2d_sincos_pos_embed_v1

from utils.img_utils import PeriodicPad2d
from afnonet_8 import Block,Decoder_Block

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=(720,1440), patch_size=(8,8), in_chans=20,
                 embed_dim=768, depth=12, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size =img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_embed = PatchEmbed_v1(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(depth)])
        self.blocks = nn.ModuleList([
            Block(dim = embed_dim,
                  mlp_ratio = 4.,
                  drop=0.,
                  drop_path=0.,
                  act_layer = nn.GELU,
                  norm_layer=norm_layer,
                  double_skip = True,
                  num_blocks = 8,
                  sparsity_threshold = 0.01,
                  hard_thresholding_fraction = 1.0)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # manxin todo no cls token
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        # print(self.decoder_pos_embed.shape)
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
        #                                       requires_grad=False)  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList([
        #     decoder_Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        self.decoder_blocks = nn.ModuleList([
            Decoder_Block(dim=decoder_embed_dim,
                  mlp_ratio=4.,
                  drop=0.,
                  drop_path=0.,
                  act_layer=nn.GELU,
                  norm_layer=norm_layer,
                  double_skip=True,
                  num_blocks=8,
                  sparsity_threshold=0.01,
                  hard_thresholding_fraction=1.0)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.in_chans * self.patch_size[0] * self.patch_size[1], bias=True)  # decoder to patch
        # manxin todo
        # self.head = nn.Linear(decoder_embed_dim, self.in_chans * self.patch_size[0] * self.patch_size[1], bias=False)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
# todo manxin rewrite pos_emb initial method
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed_v1(self.pos_embed.shape[-1], ([self.img_size, self.patch_size]),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # manxin todo no cls token
        decoder_pos_embed = get_2d_sincos_pos_embed_v1(self.decoder_pos_embed.shape[-1],
                                                    ([self.img_size, self.patch_size]), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed_v1(self.decoder_pos_embed.shape[-1],
        #                                                ([self.img_size, self.patch_size]), cls_token=True)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.absolute_pos_embed, std=.02)
        torch.nn.init.normal_(self.decoder_absolute_pos_embed,std=.02)

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
        imgs: (N, 20, H, W)
        x: (N, L, patch_size**2 *20)
        """
        p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0],20, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 20))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 20))
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
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] + self.absolute_pos_embed

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        if mask_ratio == 0.75:
            x = x.reshape(x.shape[0], int(self.img_size[0] // self.patch_size[0] // 2), self.img_size[1] // self.patch_size[1] // 2, self.embed_dim)
        elif mask_ratio == 0.:
            x = x.reshape(x.shape[0], int(self.img_size[0] // self.patch_size[0]), self.img_size[1] // self.patch_size[1], self.embed_dim)
        else:
            exit -2
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # x = self.norm(x)s
        x = x.reshape(shape=(
            x.shape[0], (int(x.shape[1]) * int(x.shape[2])),
            x.shape[3]))
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)+self.decoder_absolute_pos_embed

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # manxin todo no cls token
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x_ + self.decoder_pos_embed

        # manxin todo no cls token
        x = x.reshape(x.shape[0], int(self.img_size[0] // self.patch_size[0]), self.img_size[1] // self.patch_size[1], self.decoder_embed_dim)
        # [1, 16200, 512] ==> [1, 90, 180, 512]
        # manxin todo no cls token no block && simple nn layer
        # apply Transformer blocks [1, 90, 180, 512] (90*180 = 16200)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # manxin todo no cls token
        # # remove cls token
        # x = x[:, 1:, :]
        # # x = self.head(x)
        # manxin todo no cls token
        x = x.reshape(shape=(
        x.shape[0], (int(self.img_size[0] // self.patch_size[0]))* int(self.img_size[1] // self.patch_size[1]),
        x.shape[3]))
        return x

    def forward_loss(self, imgs, pred, mask, mask_ratio):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        if mask_ratio == 0.75:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        elif mask_ratio == 0.:
            mask = torch.ones_like(loss)
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            exit -3

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs[0], mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs[1], pred, mask, mask_ratio)
        return loss, pred, mask

class PatchEmbed_v1(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=20, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # print(x)
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        # print("projx=====================")
        # print(x)
        x = x.flatten(2)
        # print("flatenx=====================")
        # print(x)
        x = x.transpose(1, 2)
        # print("transx=====================")
        # print(x)
        return x

# todo manxin added

class Attention_v1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = (2,6,12)
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def _construct_index(self):
        coords_zi = torch.range(0, self.window_size[0])
        coords_zj = -torch.range(0, self.window_size[0]) * self.window_size[0]

        coords_hi = torch.range(0, self.window_size[1])
        coords_hj = -torch.range(0, self.window_size[1]) * self.window_size[1]

        coords_w = torch.range(0, self.window_size[2])

        coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
        coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))

        coords_flatten_1 = torch.flatten(coords_1, start_dim=1)
        coords_flatten_2 = torch.flatten(coords_2, start_dim=1)
        coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
        coords = coords.permute((1, 2, 0))

        coords[:, :, 2] += self.window_size[2] - 1
        coords[:, :, 1] *= 2 * self.window_size[2] - 1
        coords[:, :, 0] *= (2 * self.window_size[2] - 1) * self.window_size[1] * self.window_size[1]

        position_index = torch.sum(coords, dim=-1)
        position_index = torch.flatten(position_index)

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        print(q.shape)
        print(k.shape)
        print(v.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        print(attn.shape)

        attn = attn.softmax(dim=-1)
        print(attn.shape)

        attn = self.attn_drop(attn)
        print(attn.shape)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        print(x.shape)

        x = self.proj(x)
        print(x.shape)

        x = self.proj_drop(x)
        print(x.shape)

        return x

# def decoder_Block
class decoder_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.filter = Attention_v1(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        self.attn = Attention_v1(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
# todo manxin
        # if self.double_skip:
        x = x + residual
        residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
         embed_dim=768, depth=6, num_heads=12,
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    # model = MaskedAutoencoderViT(
    #     img_size=kwargs.img_size, patch_size=kwargs.patch_size, embed_dim=768, depth=12, num_heads=12,
    #     decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    #     mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
