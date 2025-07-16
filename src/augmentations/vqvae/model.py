import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(hidden_channels, latent_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=128, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_channels, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.net(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        # Flatten input
        z_flattened = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        flat_z = z_flattened.view(-1, self.embedding_dim)

        # Compute distances
        distances = (flat_z ** 2).sum(dim=1, keepdim=True) - \
                    2 * torch.matmul(flat_z, self.embedding.weight.t()) + \
                    (self.embedding.weight ** 2).sum(dim=1)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Quantized
        quantized = torch.matmul(encodings, self.embedding.weight).view(z_flattened.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Preserve gradients
        quantized = z + (quantized - z).detach()

        return quantized, loss

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(latent_dim=embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(latent_dim=embedding_dim)

    def encode(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.vq(z)
        return quantized, vq_loss

    def decode(self, quantized):
        return self.decoder(quantized)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.vq(z)
        x_recon = self.decoder(quantized)
        recon_loss = F.mse_loss(x_recon, x)
        return x_recon, recon_loss + vq_loss
