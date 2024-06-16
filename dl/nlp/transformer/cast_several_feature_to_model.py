import torch
import torch.nn as nn

# 假设有10个样本，每个样本7个时间步，每个时间步包含2个连续特征
history_continuous_features = torch.randn(10, 7, 2)


# 假设有100种产品类别和50个地区
product_embedding_dim = 10
region_embedding_dim = 5

product_embedding = nn.Embedding(100, product_embedding_dim)
region_embedding = nn.Embedding(50, region_embedding_dim)

# 假设有10个样本，每个样本7个时间步，每个时间步都有产品类别和地区
product_indices = torch.randint(0, 100, (10, 7))
region_indices = torch.randint(0, 50, (10, 7))

product_embedded = product_embedding(product_indices)
region_embedded = region_embedding(region_indices)

# 将所有特征在最后一个维度上拼接
combined_features = torch.cat(
    (history_continuous_features, product_embedded, region_embedded), dim=-1
)

print(combined_features.shape)
# 输出形状应为 (10, 7, 2 + product_embedding_dim + region_embedding_dim)
