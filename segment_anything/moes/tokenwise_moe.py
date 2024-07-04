import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modeling.mlp import Mlp

class SparseMoE(nn.Module):
    # In this case, a sequence is [1, HW, C], a token is [1, 1, C]
    def __init__(self, input_dim, output_dim, num_experts, hidden_dim, k):
        super(SparseMoE, self).__init__()
        self.num_experts = num_experts
        self.k = k
        
        # Initialize experts (MLPs)
        self.experts = nn.ModuleList([
            Mlp(in_features=input_dim, hidden_features=hidden_dim, out_features=output_dim) for _ in range(num_experts)
        ])
        
        # Initialize gating network
        self.gating_network = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size, seq_length, _ = x.size() # [B, HW, C]
        
        # Flatten the input for gating network
        x_flat = x.view(batch_size * seq_length, -1) # [BHW, C]
        
        # Get the gating scores
        gate_scores = self.gating_network(x_flat) # [BHW, N]
        
        # Get the top-k expert indices
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=1) # [BHW, K]
        
        # Normalize the top-k scores to get the routing weights
        routing_weights = F.softmax(topk_scores, dim=1)
        
        # Initialize output tensor
        output = torch.zeros(batch_size * seq_length, self.experts[0].out_features, device=x.device)
        
        # Route input to the top-k experts
        for i in range(self.k):
            expert_index = topk_indices[:, i] # [BHW]
            expert_weight = routing_weights[:, i].unsqueeze(1) # [BHW, 1]
            expert_output = torch.zeros_like(output)
            for j in range(self.num_experts):
                mask = expert_index == j
                if mask.sum() > 0:
                    expert_output[mask] = self.experts[j](x_flat[mask])
            output += expert_weight * expert_output
        
        # Reshape the output to the original sequence shape
        output = output.view(batch_size, seq_length, -1)
        
        return output


# # Example usage
# input_dim = 128
# output_dim = 128
# num_experts = 4
# hidden_dim = 64
# k = 2

# sparse_moe = SparseMoE(input_dim, output_dim, num_experts, hidden_dim, k)
# print(sparse_moe)
# x = torch.randn(1, 256, input_dim)  # Batch of 8 samples, each with 20 tokens
# output = sparse_moe(x)
# print(output.shape)  # Should be (8, 20, output_dim)
