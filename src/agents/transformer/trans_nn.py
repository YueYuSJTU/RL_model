import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    基于Transformer的特征提取器，适用于输入为时序数据的情形。
    假设观测数据形状为 (batch_size, seq_length, obs_dim)
    """
    def __init__(self, observation_space, features_dim=256, seq_length=10, 
                 d_model=64, nhead=8, num_layers=2, dropout=0.1):
        """
        :param observation_space: 环境的观测空间，其shape应该为 (seq_length, obs_dim)
        :param features_dim: 最终输出特征的维度，需与PPO网络后续MLP的输入尺寸匹配
        :param seq_length: 序列长度，即时间步数
        :param d_model: Transformer中每个token的特征维度
        :param nhead: 多头注意力机制的头数
        :param num_layers: Transformer Encoder层数
        :param dropout: dropout概率
        """
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # 假设observation_space的shape为 (seq_length, obs_dim)
        self.seq_length = seq_length
        self.obs_dim = observation_space.shape[-1]
        
        # 1. 嵌入层：将原始观测投影到d_model维度
        self.embedding = nn.Linear(self.obs_dim, d_model)
        
        # 2. 位置编码：使用函数计算位置编码
        self.pos_embedding = self._generate_positional_encoding(seq_length, d_model)
        
        # 3. Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 池化层：对Transformer输出的时序信息做聚合
        # 这里采用平均池化，可以根据需要调整为其他方式（如取CLS token、最大池化等）
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # 5. 最终投影：将d_model维度映射到features_dim，使其兼容后续MLP部分
        self.fc = nn.Linear(d_model, features_dim)
    
    def _generate_positional_encoding(self, seq_length, d_model):
        """
        生成位置编码
        :param seq_length: 序列长度
        :param d_model: 特征维度
        :return: 位置编码张量，形状为 (1, seq_length, d_model)
        """
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding = torch.zeros((seq_length, d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        :param observations: shape (batch_size, seq_length, obs_dim)
        :return: shape (batch_size, features_dim)
        """
        # 嵌入
        x = self.embedding(observations)  # (batch_size, seq_length, d_model)
        
        # 添加位置编码（广播相加）
        x = x + self.pos_embedding  # (batch_size, seq_length, d_model)
        
        # Transformer Encoder处理时序信息
        x = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # 池化：先变换维度为 (batch_size, d_model, seq_length)
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)  # (batch_size, d_model)
        
        # 最终投影
        features = self.fc(x)  # (batch_size, features_dim)
        return features
