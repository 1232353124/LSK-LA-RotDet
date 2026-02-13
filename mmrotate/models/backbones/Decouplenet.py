import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from torch import Tensor
import copy
import os
import antialiased_cnns
import torch.nn.functional as F
from mmengine.registry import MODELS
from mmengine.logging import MMLogger
from mmengine.runner import load_checkpoint

# FID is feature integration downsampling
class FID(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.Gconv = nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.pii = PII(dim*2, 8)
        self.conv_D = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=2, padding=1, groups=dim*2)
        self.act = nn.GELU()
        self.batch_norm_c = nn.BatchNorm2d(dim*2)
        self.max_m1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.max_m2 = antialiased_cnns.BlurPool(dim*2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(dim*2)
        self.fusion = nn.Conv2d(dim*4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x):  # x = [B, C, H, W]

        # Gconv + PII
        x = self.Gconv(x)  # h = [B, 2C, H, W]
        x = self.pii(x)

        # MaxD + anti-aliased
        max = self.max_m1(x)  # m = [B, 2C, H/2, W/2]
        max = self.max_m2(max)  # m = [B, 2C, H/2, W/2]
        max = self.batch_norm_m(max)

        # ConvD
        conv = self.conv_D(x)  # h = [B, 2C, H/2, W/2]
        conv = self.act(conv)
        conv = self.batch_norm_c(conv)  # h = [B, 2C, H/2, W/2]

        # Concat
        x = torch.cat([conv, max], dim=1)  # x = [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]  -->  [B, 2C, H/2, W/2]

        return x


try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


class PII(nn.Module):

    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = int((dim / 2) - self.dim_conv)
        self.conv = nn.Conv2d(self.dim_conv*2, self.dim_conv*2, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:

        x1, x2, x3, x4 = torch.split(x, [self.dim_conv, self.dim_untouched, self.dim_conv, self.dim_untouched], dim=1)
        x = torch.cat((x1, x3), 1)
        x1 = self.conv(x)
        x = torch.cat((x1, x2, x4), 1)

        return x


# MRLA is medium-range lightweight Attention
class MRLA(nn.Module):
    def __init__(self, channel, att_kernel):
        super(MRLA, self).__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        channels12 = int(channel / 2)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(channel, channels12, 1, 1, bias=False),
            nn.BatchNorm2d(channels12),
            nn.GELU(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(channels12, channels12, 3, 1, 1, groups=channels12, bias=False),
            nn.BatchNorm2d(channels12),
            nn.GELU(),
        )
        self.init = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
        )
        self.H_att = nn.Conv2d(channel, channel, (att_kernel, 1), 1, (att_padding, 0), groups=channel, bias=False)
        self.V_att = nn.Conv2d(channel, channel, (1, att_kernel), 1, (0, att_padding), groups=channel, bias=False)
        self.batchnorm = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_tem = self.init(F.avg_pool2d(x, kernel_size=2, stride=2))
        x_h = self.H_att(x_tem)
        x_w = self.V_att(x_tem)
        mrla = self.batchnorm(x_h + x_w)

        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = out[:, :self.channel, :, :] * F.interpolate(self.gate_fn(mrla),
                                                          size=(out.shape[-2], out.shape[-1]),
                                                          mode='nearest')
        return out


# GA is long range Attention
class GA(nn.Module):
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


# MBFD is multi-branch feature decoupling module
class MBFD(nn.Module):

    def __init__(self, dim, stage, att_kernel):
        super().__init__()
        self.dim = dim
        self.stage = stage
        self.dim_learn = dim // 4
        self.dim_untouched = dim - self.dim_learn - self.dim_learn
        self.Conv = nn.Conv2d(self.dim_learn, self.dim_learn, 3, 1, 1, bias=False)
        self.MRLA = MRLA(self.dim_learn, att_kernel)  # MRLA is medium range Attention
        if stage > 2:
            self.GA = GA(self.dim_untouched)      # GA is long range Attention
            self.norm = nn.BatchNorm2d(self.dim_untouched)

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2, x3 = torch.split(x, [self.dim_learn, self.dim_learn, self.dim_untouched], dim=1)
        x1 = self.Conv(x1)
        x2 = self.MRLA(x2)
        if self.stage > 2:
            x3 = self.norm(x3 + self.GA(x3))
        x = torch.cat((x1, x2, x3), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 stage,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.MBFD = MBFD(
            dim,
            stage,
            att_kernel
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.MBFD(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.MBFD(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 stage,
                 depth,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer
                 ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                stage=stage,
                att_kernel=att_kernel,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x

@MODELS.register_module()
class DecoupleNet(nn.Module):
    def __init__(self,
                 in_chans=3,  # 输入图像通道数
                 num_classes=1000,  # 分类任务类别数
                 embed_dim=32,  # 初始嵌入维度
                 depths=(1, 6, 6, 2),  # 每个 stage 的 block 数量
                 att_kernel=(9, 9, 9, 9),  # 每个 stage 的注意力卷积核大小
                 mlp_ratio=2.,  # MLP 隐藏层扩展倍数
                 patch_size=4,  # patch 大小
                 patch_stride=4,  # patch 步长
                 patch_norm=True,  # 是否对 patch 嵌入做归一化
                 feature_dim=1280,  # 分类头中间特征维度
                 drop_path_rate=0.1,  # 随机深度丢弃率
                 layer_scale_init_value=0,  # 层缩放初始值
                 fork_feat=True,  # 是否输出多尺度特征用于检测
                 init_cfg=dict(type='Pretrained', checkpoint='/home/xnxt/code/mmrotate-dcfl-main/jcd/DecoupleNet_D2.pth'),  # 初始化配置
                 Pretrained=None,  # 预训练权重路径
                 **kwargs):  # 其他配置
        super().__init__()

        norm_layer = nn.BatchNorm2d  # 默认归一化层类型
        act_layer = nn.GELU  # 激活函数

        if not fork_feat:
            self.num_classes = num_classes  # 如果用于分类，则记录类别数
        self.num_stages = len(depths)  # stage 数量
        self.embed_dim = embed_dim  # 记录嵌入维度
        self.patch_norm = patch_norm  # 是否对 patch 嵌入做归一化
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))  # 最后一层特征维度
        self.mlp_ratio = mlp_ratio  # MLP 扩展比例
        self.depths = depths  # 每个 stage 的深度
        self.att_kernel = att_kernel  # 每个 stage 的注意力核大小
        self.fork_feat = True  # 输出四层特征
        self.out_indices = [0, 2, 4, 6]  # backbone输出四层特征
        self.fpn_indices = [2, 4, 6]  # FPN只使用后三层

        # 将图像划分成非重叠 patch 并嵌入
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # 随机深度丢弃率线性衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建每个 stage 的模块
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(
                dim=int(embed_dim * 2 ** i_stage),  # 当前 stage 的维度
                stage=i_stage,  # stage 编号
                depth=depths[i_stage],  # 当前 stage 的 block 数
                att_kernel=att_kernel[i_stage],  # 注意力核大小
                mlp_ratio=self.mlp_ratio,  # MLP 扩展比
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],  # 当前 stage 的 drop path 配置
                layer_scale_init_value=layer_scale_init_value,  # 层缩放初始化值
                norm_layer=norm_layer,  # 归一化层
                act_layer=act_layer  # 激活函数
            )
            stages_list.append(stage)

            # 除了最后一个 stage，每个 stage 后加一个 FID（Feature Interaction & Downsampling）层
            if i_stage < self.num_stages - 1:
                stages_list.append(FID(dim=int(embed_dim * 2 ** i_stage)))

        self.stages = nn.Sequential(*stages_list)  # 多个 stage 和 FID 层串联

        self.fork_feat = fork_feat  # 是否用于检测任务

        if self.fork_feat:
            self.forward = self.forward_det  # 如果是检测，使用检测前向
            self.out_indices = [0, 2, 4, 6]  # 每个 stage 输出的位置（FID 后的输出）
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):  # 环境变量控制是否使用最后3个特征
                    raise NotImplementedError
                else:
                    layer = norm_layer(int(embed_dim * 2 ** i_emb))  # 添加归一化层
                layer_name = f'norm{i_layer}'  # 层名如 norm0, norm2...
                self.add_module(layer_name, layer)  # 注册为模块
        else:
            self.forward = self.forward_cls  # 否则为分类前向
            # 分类头：全局平均池化 → 1×1卷积 → 激活
            self.avgpool_pre_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
                act_layer()
            )
            self.head = nn.Linear(feature_dim, num_classes) if num_classes > 0 else nn.Identity()  # 分类层

        self.apply(self.cls_init_weights)  # 应用初始化
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or Pretrained is not None):
            self.init_weights()  # 加载预训练权重

    def cls_init_weights(self, m):
        # 用于初始化分类头等模块中的权重
        if isinstance(m, nn.Linear):
            # 如果是全连接层，使用截断正态分布初始化权重
            trunc_normal_(m.weight, std=.02)
            # 如果全连接层有偏置，则初始化为 0
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            # 如果是一维或二维卷积层，使用截断正态分布初始化权重
            trunc_normal_(m.weight, std=.02)
            # 如果有偏置，初始化为 0
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            # 如果是 LayerNorm 或 GroupNorm，初始化偏置为 0，权重为 1
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def init_weights(self, Pretrained=None):
        logger = MMLogger.get_current_instance()

        # 初始化整个模型的权重（包括加载预训练权重或打印警告）
        # logger = get_root_logger()  # 获取日志记录器

        # 如果没有指定 init_cfg 和 Pretrained，则表示从头开始训练
        if self.init_cfg is None and Pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass  # 不加载任何预训练权重，直接使用默认初始化
        else:
            # 要么指定 init_cfg，要么指定 Pretrained，否则报错
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            # 获取 checkpoint 路径
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif Pretrained is not None:
                ckpt_path = Pretrained

            # 从 checkpoint 文件加载模型参数（字典）
            ckpt = load_checkpoint(
                model=self,
                filename=ckpt_path,
                map_location='cpu',
                logger=logger
            )

            # 根据 ckpt 内容提取真正的 state_dict（兼容不同保存方式）
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict  # 最终使用的权重字典

            # 将加载的 state_dict 应用到当前模型中，但不强制匹配所有 key
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # 打印缺失和未预期 key，便于调试或确认加载情况
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def forward_cls(self, x):
        # 分类任务前向传播
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)  # B C 1 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

    def forward_det(self, x: Tensor) -> Tensor:
        # 检测任务前向传播
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                if idx in self.fpn_indices:  # 只将后三层特征输出到FPN
                    outs.append(x_out)
        return outs