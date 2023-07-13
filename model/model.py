import torch
from math import ceil
import torch.nn as nn
from mmaction.models.backbones import SwinTransformer3D
import torch.nn.functional as F
import sys
sys.path.append(r"/home/um202070049/clip2movie")
from base import BaseModel
from einops import rearrange, repeat
class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Clip2MovModel(BaseModel):
    def __init__(self,
                 swin_arch,     #若为字典，则需包含 embed_dims，depths，num_heads三个key
                 num_keyframe,
                 frame_size=[224,224],
                 shot_embed=1024,
                 patch_width=12, 
                 patch_height=12,
                 attn_head=8,   #attn_head需要整除shot_embed
                 cross_attn_depth=2,
                 cls_attn_depth=2,
                 pretrained_swin=None,
                 pretrained_2Dswin=True,
                 in_chans=3,
                 shot_window=[4,7,7],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 frozen_stages=-1,
                 use_checkpoint=False
                 ):
        super().__init__()
        patch_size = [num_keyframe, patch_height, patch_width]
        self.shot_emb_dim = shot_embed
        if isinstance(swin_arch, str):
            swin_depths = 4
            if swin_arch in ['t', 'tiny', 's', 'small']: patch_emb = 96
            else: patch_emb = 128
        else: 
            swin_depths = len(swin_arch['depths'])
            patch_emb = swin_arch['embed_dims']
        self.shot_swintransformer = SwinTransformer3D(arch=swin_arch,
                                                      pretrained=pretrained_swin, 
                                                      pretrained2d=pretrained_2Dswin, 
                                                      patch_size=patch_size,
                                                      in_channels=in_chans,
                                                      window_size=shot_window,
                                                      mlp_ratio=mlp_ratio,
                                                      qkv_bias=qkv_bias,
                                                      drop_rate=drop_rate,
                                                      attn_drop_rate=attn_drop_rate,
                                                      drop_path_rate=drop_path_rate,
                                                      frozen_stages=frozen_stages,
                                                      with_cp=use_checkpoint,
                                                      out_indices=(swin_depths-1,)
                                                      )
        frame_h, frame_w = frame_size
        h, w = int(ceil(frame_h/patch_height)), int(ceil(frame_w/patch_width))
        for i in range(swin_depths-1):
            h, w = int(ceil(h / 2)), int(ceil(w / 2))
        #print(h, w)
        self.proj = nn.Conv3d(patch_emb*2**(swin_depths-1), self.shot_emb_dim, kernel_size=(1,h,w))
        crossattn_layer = nn.TransformerDecoderLayer(d_model=self.shot_emb_dim, nhead=attn_head, dim_feedforward=1024, batch_first=True) #input_shape:(batch_size,序列长度，embedding维度)
        self.crossattn = nn.TransformerDecoder(crossattn_layer, num_layers=cross_attn_depth)
        attn_layer = nn.TransformerEncoderLayer(d_model=self.shot_emb_dim, nhead=attn_head, dim_feedforward=1024, batch_first=True)
        self.attn = nn.TransformerEncoder(attn_layer, num_layers=cls_attn_depth)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.shot_emb_dim))
        self.mlp = Mlp(in_features=self.shot_emb_dim, out_features=1, drop=drop_rate)
    def forward(self, clip, movie):
        B, num_clip = clip.shape[0], clip.shape[1]
        clip = self.feature_extract(clip)
        movie = self.feature_extract(movie)
        clip = rearrange(clip, 'b n c d h w -> (b n) c d h w')
        movie = rearrange(movie, 'b n c d h w -> (b n) c d h w')
        #print(clip.shape, movie.shape)
        clip = self.proj(clip)
        movie = self.proj(movie)
        #print(clip.shape, movie.shape)
        clip = rearrange(clip, '(b n) c d h w -> b n d (c h w)', b=B, n=num_clip)
        movie = rearrange(movie, '(b n) c d h w -> b n d (c h w)', b=B, n=1)
        #print(clip.shape, movie.shape)
        out_logit = self.retrival(clip, movie)
        return out_logit    #训练: b * n * 1
    def feature_extract(self, x):   #shape x: (batch_size, num_clip, in_chan, frame_num, height, wodth),x为电影时num_clip=1
        B, num_clip, in_chan, D, H, W = x.shape
        x = x.reshape(-1, in_chan, D, H, W)
        x = self.shot_swintransformer(x)
        out_chan, out_D, out_H, out_W = x.shape[1:]
        x = rearrange(x, '(b n) c d h w -> b n c d h w', b=B, n=num_clip)
        return x
    def retrival(self, clip_feat, movie_feat):  #feat shape: B * num_clip * clip_shot * shot_embed
        clip_feat = self.crossAttention(clip_feat, movie_feat)
        cls_head = self.clsTokenAggr(clip_feat)
        out_logit = self.mlp(cls_head)
        return out_logit
    def crossAttention(self, clip_feat, movie_feat):
        B, num_clip, clip_shot, shot_emb_dim = clip_feat.shape
        B, _, movie_shot, shot_emb_dim = movie_feat.shape
        #if self.training:
        #assert Bc == Bm, 'batch size of clip and movie should be same when training!'
        clip_feat = clip_feat.reshape(-1, clip_shot, shot_emb_dim)
        movie_feats = movie_feat.expand(-1, num_clip, -1, -1).contiguous().view(-1, movie_shot, shot_emb_dim)       #expand为in-place操作，可能影响多卡训练
        #print(clip_feat.shape, movie_feats.shape)
        clip_feat = self.crossattn(clip_feat, movie_feats)
        clip_feat = clip_feat.reshape(B, num_clip, clip_shot, -1)
        return clip_feat    # shape: (Bc, num_clip, clip_shot, shot_emb_dim)
        '''
        else:
            clip_feats = torch.repeat_interleave(clip_feat, Bm, dim=1).reshape(-1, clip_shot, shot_emb_dim)
            movie_feats = torch.repeat_interleave(movie_feat, Bc*num_clip, dim=0).reshape(-1, movie_shot, shot_emb_dim)
            clip_feats = self.crossattn(clip_feats, movie_feats)
            clip_feats = clip_feats.reshape(Bc, num_clip*Bm, clip_shot, -1)
            return clip_feats   #shape: (Bc, num_clip*Bm, clip_shot, shot_emb_dim)
        '''
    def clsTokenAggr(self, clip_feat):
        B, num_clip, clip_shot, shot_emb_dim = clip_feat.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B*num_clip)
        clip_feat = clip_feat.reshape(-1, clip_shot, shot_emb_dim)
        clip_feat = torch.cat((cls_tokens, clip_feat), dim=1)
        clip_feat = self.attn(clip_feat)
        cls_tokens = clip_feat[:, 0].reshape(B, num_clip, -1)
        return cls_tokens   # B, num_clip, d
'''
if __name__ == '__main__':
    import torch
    clip_test, movie_test, clip_test_n = torch.randn(2, 10, 3, 100, 128, 128), torch.randn(2, 1, 3, 300, 128, 128), torch.randn(2,20,3,100,128,128)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import time
    t1 = time.time()
    clip_test, movie_test, clip_test_n = clip_test.to(device), movie_test.to(device), clip_test_n.to(device)
    model_test = Clip2MovModel(swin_arch={"embed_dims":96,  "depths":[1], "num_heads":[2]}, num_keyframe=5, frame_size=(128,128), pretrained_2Dswin=False)
    model_test = model_test.to(device)
    t2 = time.time()
    pres = model_test(clip_test, movie_test)
    nres = model_test(clip_test_n, movie_test)
    print(pres, nres, "used time: ", t2 - t1)
'''
