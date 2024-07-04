import torch
import torch.jit
import torch.nn as nn
import torch.utils.checkpoint

class LoRA(nn.Module):
    def __init__(self, linear_layer, rank):
        super(LoRA, self).__init__()
        self.linear_layer = linear_layer
        
        # 获取原始线性层的输入和输出维度
        self.input_dim = linear_layer.in_features
        self.output_dim = linear_layer.out_features
        
        # 定义LoRA低秩分解的两个矩阵
        self.A = nn.Parameter(torch.randn(self.input_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, self.output_dim))
        
        # 冻结原始线性层的参数
        for param in self.linear_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # 通过原始线性层
        original_output = self.linear_layer(x)
        
        # 通过LoRA低秩矩阵分解
        lora_output = x @ self.A @ self.B
        
        # 将两个输出相加
        return original_output + lora_output

class MlpLoRA(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, rank=4):
        super(MlpLoRA, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

        # 添加LoRA
        self.lora_fc1 = LoRA(self.fc1, rank)
        self.lora_fc2 = LoRA(self.fc2, rank)

    def forward(self, x):
        x = self.lora_fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.lora_fc2(x)
        x = self.drop2(x)
        return x