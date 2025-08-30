"""
WildFire CNN Model
专用于野火传播预测的时空CNN模型，结合U-Net和LSTM进行多尺度时空建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention

class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module)"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ConvBlock(nn.Module):
    """基础卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        if self.use_attention:
            x = self.attention(x)
        
        return x

class UNetEncoder(nn.Module):
    """U-Net编码器"""
    
    def __init__(self, in_channels: int, features: List[int] = [64, 128, 256, 512]):
        super(UNetEncoder, self).__init__()
        self.features = features
        
        # 编码路径
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # 输入层
        self.encoder_blocks.append(ConvBlock(in_channels, features[0]))
        
        # 其他编码层
        for i in range(1, len(features)):
            self.encoder_blocks.append(ConvBlock(features[i-1], features[i]))
    
    def forward(self, x):
        skip_connections = []
        
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i < len(self.encoder_blocks) - 1:  # 不对最后一层进行池化
                skip_connections.append(x)
                x = self.pool(x)
        
        return x, skip_connections

class UNetDecoder(nn.Module):
    """U-Net解码器"""
    
    def __init__(self, features: List[int] = [512, 256, 128, 64], out_channels: int = 1):
        super(UNetDecoder, self).__init__()
        self.features = features
        
        # 上采样
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(features) - 1):
            self.upsamples.append(
                nn.ConvTranspose2d(features[i], features[i+1], 2, stride=2)
            )
            # 拼接后的通道数是 features[i+1] + features[i+1] (来自skip connection)
            concat_channels = features[i+1] + features[i+1]
            self.decoder_blocks.append(
                ConvBlock(concat_channels, features[i+1])
            )
        
        # 输出层
        self.final_conv = nn.Conv2d(features[-1], out_channels, 1)
    
    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]  # 反转skip connections
        
        for i, (upsample, block) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            x = upsample(x)
            skip = skip_connections[i]
            
            # 确保尺寸匹配
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        
        return self.final_conv(x)

class ConvLSTMCell(nn.Module):
    """卷积LSTM单元"""
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, 
                       dtype=torch.float, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width,
                       dtype=torch.float, device=self.conv.weight.device)
        )

class ConvLSTM(nn.Module):
    """多层卷积LSTM"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], kernel_size: int = 3, 
                 num_layers: int = 1, batch_first: bool = True, bias: bool = True, 
                 return_all_layers: bool = False):
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = self._extend_for_multilayer(hidden_dims, num_layers)
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=self.kernel_size,
                    bias=self.bias
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)
    
    def _extend_for_multilayer(self, param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, seq_len, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

class WildfireCNN(nn.Module):
    """
    野火传播预测CNN模型
    结合U-Net空间特征提取和ConvLSTM时序建模
    """
    
    def __init__(
        self,
        input_channels: int = 23,
        sequence_length: int = 5,
        unet_features: List[int] = [64, 128, 256, 512],
        lstm_hidden_dims: List[int] = [128, 64],
        num_classes: int = 1,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super(WildfireCNN, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # 特征预处理层（处理土地覆盖类别特征）
        self.landcover_embedding = nn.Embedding(17, 8)  # 16个类别+padding
        self.feature_projection = nn.Conv2d(
            input_channels - 1 + 8, unet_features[0], 1  # -1 for landcover +8 for embedding
        )
        
        # U-Net编码器（用于空间特征提取）
        self.unet_encoder = UNetEncoder(unet_features[0], unet_features)
        
        # ConvLSTM（用于时序建模）
        self.conv_lstm = ConvLSTM(
            input_dim=unet_features[-1],
            hidden_dims=lstm_hidden_dims,
            kernel_size=3,
            num_layers=len(lstm_hidden_dims),
            batch_first=True,
            return_all_layers=False
        )
        
        # U-Net解码器（用于空间重建）
        decoder_features = [lstm_hidden_dims[-1]] + unet_features[::-1][1:]
        self.unet_decoder = UNetDecoder(decoder_features, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout)
        
        # 输出激活
        self.output_activation = nn.Sigmoid()
    
    def _process_features(self, x):
        """预处理输入特征"""
        b, t, c, h, w = x.shape
        
        # 分离土地覆盖特征（通道16）和其他特征
        landcover = x[:, :, 16, :, :].long()  # (B, T, H, W)
        other_features = torch.cat([x[:, :, :16, :, :], x[:, :, 17:, :, :]], dim=2)  # (B, T, C-1, H, W)
        
        # 确保土地覆盖值在有效范围内 (1-16 -> 0-15)
        landcover = torch.clamp(landcover - 1, 0, 15)
        
        # 土地覆盖embedding
        landcover_embedded = self.landcover_embedding(landcover)  # (B, T, H, W, 8)
        landcover_embedded = landcover_embedded.permute(0, 1, 4, 2, 3)  # (B, T, 8, H, W)
        
        # 合并特征
        processed_features = torch.cat([other_features, landcover_embedded], dim=2)  # (B, T, C-1+8, H, W)
        
        return processed_features
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, sequence_length, channels, height, width)
        
        Returns:
            output: 预测结果 (batch_size, num_classes, height, width)
        """
        b, t, c, h, w = x.shape
        
        # 预处理特征
        x = self._process_features(x)
        
        # 特征投影
        x = x.view(b * t, -1, h, w)
        x = self.feature_projection(x)
        x = x.view(b, t, -1, h, w)
        
        # 空间特征提取（逐帧处理）
        spatial_features = []
        skip_connections_list = []
        
        for i in range(t):
            frame = x[:, i, :, :, :]  # (B, C, H, W)
            features, skip_connections = self.unet_encoder(frame)
            spatial_features.append(features)
            if i == t - 1:  # 只保存最后一帧的skip connections用于解码
                final_skip_connections = skip_connections
        
        # 时序建模
        spatial_features = torch.stack(spatial_features, dim=1)  # (B, T, C, H', W')
        temporal_features, _ = self.conv_lstm(spatial_features)
        temporal_features = temporal_features[0]  # 取最后一层的输出
        
        # 取最后一个时间步的特征进行解码
        final_features = temporal_features[:, -1, :, :, :]  # (B, C, H', W')
        
        # 空间重建
        final_features = self.dropout(final_features)
        output = self.unet_decoder(final_features, final_skip_connections)
        
        # 输出激活
        output = self.output_activation(output)
        
        return output

class WildfireResNet(nn.Module):
    """
    基于ResNet的简化版野火预测模型（用于对比实验）
    """
    
    def __init__(self, input_channels: int = 23, num_classes: int = 1):
        super(WildfireResNet, self).__init__()
        
        # 特征预处理
        self.landcover_embedding = nn.Embedding(17, 8)
        self.feature_projection = nn.Conv2d(input_channels - 1 + 8, 64, 1)
        
        # ResNet-like backbone
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 上采样和输出
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
            nn.Sigmoid()
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(self._basic_block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._basic_block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _basic_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 只使用最后一个时间步
        if len(x.shape) == 5:  # (B, T, C, H, W)
            x = x[:, -1, :, :, :]  # (B, C, H, W)
        
        # 处理特征（与WildfireCNN相同）
        b, c, h, w = x.shape
        landcover = x[:, 16, :, :].long()
        other_features = torch.cat([x[:, :16, :, :], x[:, 17:, :, :]], dim=1)
        
        landcover_embedded = self.landcover_embedding(landcover)
        landcover_embedded = landcover_embedded.permute(0, 3, 1, 2)
        
        x = torch.cat([other_features, landcover_embedded], dim=1)
        x = self.feature_projection(x)
        
        # 前向传播
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upsample(x)
        
        return x

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试模型
    print("🧪 测试野火预测CNN模型...")
    
    # 创建模型
    model = WildfireCNN(
        input_channels=23,
        sequence_length=5,
        num_classes=1
    )
    
    # 创建测试输入
    batch_size = 2
    sequence_length = 5
    height, width = 128, 128
    
    x = torch.randn(batch_size, sequence_length, 23, height, width)
    
    # 前向传播
    print(f"输入形状: {x.shape}")
    output = model(x)
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {count_parameters(model):,}")
    
    # 测试ResNet模型
    resnet_model = WildfireResNet(input_channels=23, num_classes=1)
    resnet_output = resnet_model(x)
    print(f"ResNet输出形状: {resnet_output.shape}")
    print(f"ResNet参数数量: {count_parameters(resnet_model):,}")
    
    print("✅ 模型测试完成！") 