"""
Vision Transformer (ViT) 模型实现
适配不同数据集（CIFAR-10, GTSRB, Tiny ImageNet, MNIST等）

工程说明：
- ViT 使用 LayerNorm + Linear 层，与 ResNet/DenseNet 的 Conv+BN 结构不同
- 对于 CLP/ABI 机制，ViT 需要单独处理（基于 LayerNorm 和 Linear 权重）
- 支持 return_hidden 参数以兼容防御方法（如 AC）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class PatchEmbedding(nn.Module):
    """
    图像分块嵌入层
    将输入图像分割为固定大小的 patch，然后线性投影到嵌入空间
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 使用卷积实现 patch 嵌入（等价于将每个 patch 展平后乘以权重矩阵）
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size]
        x = self.proj(x)
        # [B, embed_dim, num_patches_h, num_patches_w] -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """
    多头自注意力模块
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        # [B, N, 3*C] -> [B, N, 3, num_heads, head_dim] -> [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    前馈神经网络（FFN）模块
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer 编码器块
    包含：LayerNorm -> Attention -> 残差连接 -> LayerNorm -> MLP -> 残差连接
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 模型
    
    参数:
        img_size: 输入图像尺寸（假设为正方形）
        patch_size: 每个 patch 的尺寸
        in_channels: 输入通道数
        num_classes: 分类类别数
        embed_dim: 嵌入维度
        depth: Transformer 块数量
        num_heads: 注意力头数量
        mlp_ratio: MLP 隐藏层维度倍数
        qkv_bias: QKV 是否使用偏置
        drop_rate: Dropout 比率
        attn_drop_rate: 注意力 Dropout 比率
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch 嵌入
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size,
                                          in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 可学习的分类 token 和位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer 编码器块
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        # 最终 LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        # 位置嵌入使用截断正态分布初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 线性层和 LayerNorm 使用默认初始化
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x, return_hidden=False):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            return_hidden: 是否返回隐藏层特征（用于 AC 等防御方法）
        
        Returns:
            如果 return_hidden=True: (logits, hidden_features)
            否则: logits
        """
        B = x.shape[0]
        
        # Patch 嵌入
        x = self.patch_embed(x)
        
        # 添加分类 token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer 编码器
        for block in self.blocks:
            x = block(x)
        
        # 最终 LayerNorm
        x = self.norm(x)
        
        # 提取分类 token 作为 hidden features
        hidden = x[:, 0]
        
        # 分类头
        out = self.head(hidden)
        
        if return_hidden:
            return out, hidden
        else:
            return out
    
    def freeze_feature(self):
        """冻结除分类头外的所有参数"""
        for name, para in self.named_parameters():
            if 'head' not in name:
                para.requires_grad = False
    
    def unfreeze_feature(self):
        """解冻所有参数"""
        for name, para in self.named_parameters():
            para.requires_grad = True


# ========== 各数据集的 ViT 模型定义 ==========

def ViT_cifar10(num_classes=10):
    """
    ViT-Small for CIFAR-10 (32x32 input)
    
    配置: patch_size=4, embed_dim=192, depth=12, num_heads=3
    参数量约 ~22M（比 ResNet18 大，但在小数据集上表现良好）
    """
    return VisionTransformer(
        img_size=32, patch_size=4, in_channels=3, num_classes=num_classes,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
        drop_rate=0.1, attn_drop_rate=0.
    )


def ViT_gtsrb(num_classes=43):
    """
    ViT-Small for GTSRB (32x32 input)
    """
    return VisionTransformer(
        img_size=32, patch_size=4, in_channels=3, num_classes=num_classes,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
        drop_rate=0.1, attn_drop_rate=0.
    )


def ViT_imagenette(num_classes=10):
    """
    ViT-Small for ImageNette (224x224 input)
    
    注意: 对于 224x224 输入，使用 patch_size=16，得到 14x14=196 个 patch
    """
    return VisionTransformer(
        img_size=224, patch_size=16, in_channels=3, num_classes=num_classes,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True,
        drop_rate=0.1, attn_drop_rate=0.
    )


def ViT_tiny_imagenet(num_classes=200):
    """
    ViT-Small for Tiny ImageNet (64x64 input)
    
    配置: patch_size=8, embed_dim=256, depth=12, num_heads=4
    """
    return VisionTransformer(
        img_size=64, patch_size=8, in_channels=3, num_classes=num_classes,
        embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True,
        drop_rate=0.1, attn_drop_rate=0.
    )


def ViT_mnist(num_classes=10):
    """
    ViT-Tiny for MNIST (28x28 input, 3 channels after repeat)
    
    配置: patch_size=4, embed_dim=128, depth=6, num_heads=4
    更小的模型适合 MNIST
    """
    return VisionTransformer(
        img_size=28, patch_size=4, in_channels=3, num_classes=num_classes,
        embed_dim=128, depth=6, num_heads=4, mlp_ratio=4., qkv_bias=True,
        drop_rate=0.1, attn_drop_rate=0.
    )


def ViT_imagenet(num_classes=1000):
    """
    ViT-Base for ImageNet (224x224 input)
    
    配置: patch_size=16, embed_dim=768, depth=12, num_heads=12
    """
    return VisionTransformer(
        img_size=224, patch_size=16, in_channels=3, num_classes=num_classes,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
        drop_rate=0.1, attn_drop_rate=0.
    )


# ========== 测试代码 ==========
if __name__ == '__main__':
    # 测试 CIFAR-10 ViT
    model = ViT_cifar10()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"CIFAR-10 ViT output shape: {out.shape}")
    
    out, hidden = model(x, return_hidden=True)
    print(f"CIFAR-10 ViT hidden shape: {hidden.shape}")
    
    # 测试 Tiny ImageNet ViT
    model = ViT_tiny_imagenet()
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    print(f"Tiny ImageNet ViT output shape: {out.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
