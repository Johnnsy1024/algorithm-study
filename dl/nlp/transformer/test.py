import matplotlib.pyplot as plt
import seaborn as sns
import torch
from Encoder import PositionalEncoding

pos_encod = PositionalEncoding(512)
print(pos_encod(torch.randint(low=0, high=32, size=(128, 32))).shape)
plt.figure(dpi=300)
sns.heatmap(
    pos_encod(torch.randint(low=0, high=32, size=(128, 32))).detach().numpy()[0, ...],
    cmap="viridis",
)
plt.savefig("./vector_false.png")
pos_encod = PositionalEncoding(512)
print(pos_encod(torch.randint(low=0, high=32, size=(128, 32))).shape)
plt.figure(dpi=300)
sns.heatmap(
    pos_encod(torch.randint(low=0, high=32, size=(128, 32))).detach().numpy()[0, ...],
    cmap="viridis",
)
plt.savefig("./vector_true.png")


embed_demo = torch.nn.Embedding(32, 512)
print(embed_demo.weight.shape)
