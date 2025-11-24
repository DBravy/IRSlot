"""
Loss functions for instance recognition training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorEntropyLoss(nn.Module):
    """
    Regularization loss that encourages each slot to attend to color-coherent regions.

    For each slot, computes the weighted color distribution based on attention weights,
    then penalizes high entropy (encouraging slots to specialize on fewer colors).
    """
    def __init__(self, num_colors=10, eps=1e-8):
        """
        Args:
            num_colors: Number of possible colors (10 for ARC: 0-9)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.num_colors = num_colors
        self.eps = eps

    def forward(self, attn_weights, grids):
        """
        Compute color entropy regularization loss.

        Args:
            attn_weights: [B, num_slots, H, W] - Attention masks from slot attention
            grids: [B, H, W] - Original grids with color values 0-9

        Returns:
            loss: Scalar tensor - Mean entropy across all slots (to minimize)
            metrics: Dict with additional metrics for logging
        """
        B, num_slots, H, W = attn_weights.shape

        # Flatten spatial dimensions
        # attn: [B, num_slots, H*W]
        attn_flat = attn_weights.reshape(B, num_slots, -1)

        # grids: [B, H*W]
        grids_flat = grids.reshape(B, -1)

        # Create one-hot encoding of colors: [B, H*W, num_colors]
        colors_onehot = F.one_hot(grids_flat.long(), num_classes=self.num_colors).float()

        # For each slot, compute weighted color distribution
        # attn_flat: [B, num_slots, H*W]
        # colors_onehot: [B, H*W, num_colors]
        # Result: [B, num_slots, num_colors]
        #
        # We want: for each slot, sum(attn * color_onehot) across spatial positions
        # attn_flat.unsqueeze(-1): [B, num_slots, H*W, 1]
        # colors_onehot.unsqueeze(1): [B, 1, H*W, num_colors]
        # Product: [B, num_slots, H*W, num_colors]
        # Sum over H*W: [B, num_slots, num_colors]

        color_dist = torch.einsum('bsh,bhc->bsc', attn_flat, colors_onehot)

        # Normalize to get probability distribution (attention is already normalized,
        # but we normalize again to be safe)
        color_dist = color_dist / (color_dist.sum(dim=-1, keepdim=True) + self.eps)

        # Compute entropy: -sum(p * log(p))
        # Add eps to avoid log(0)
        entropy = -torch.sum(color_dist * torch.log(color_dist + self.eps), dim=-1)

        # Mean entropy across batch and slots
        mean_entropy = entropy.mean()

        # Max possible entropy for reference (uniform over num_colors)
        max_entropy = torch.log(torch.tensor(self.num_colors, dtype=torch.float32, device=grids.device))

        # Normalized entropy (0 = perfect coherence, 1 = uniform)
        normalized_entropy = mean_entropy / max_entropy

        metrics = {
            'color_entropy': mean_entropy.item(),
            'color_entropy_normalized': normalized_entropy.item(),
        }

        return mean_entropy, metrics


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Normalized Temperature-scaled Cross Entropy) Loss.

    Used for contrastive learning / instance discrimination.

    Formula:
        L = -log(exp(sim(q, k+) / τ) / (exp(sim(q, k+) / τ) + Σ exp(sim(q, k-) / τ)))

    where:
    - q: query embedding (current batch embedding)
    - k+: positive key (stored embedding from memory bank)
    - k-: negative keys (random samples from memory bank)
    - τ: temperature parameter
    """
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
                        Lower temperature = sharper distribution
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeddings, positive_embeddings, negative_embeddings):
        """
        Compute InfoNCE loss.

        Args:
            query_embeddings: [B, D] - Current batch embeddings
            positive_embeddings: [B, D] - Stored embeddings from memory bank
            negative_embeddings: [N, D] - Negative samples from memory bank
                                 where N is number of negatives

        Returns:
            loss: Scalar tensor
            metrics: Dict with additional metrics for logging
        """
        B, D = query_embeddings.shape
        N = negative_embeddings.shape[0]

        # Normalize all embeddings
        query_embeddings = F.normalize(query_embeddings, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, dim=1)

        # Compute positive similarities (one per batch item)
        # [B, D] * [B, D] -> [B]
        positive_sim_raw = (query_embeddings * positive_embeddings).sum(dim=1)
        positive_sim = positive_sim_raw / self.temperature

        # Compute negative similarities
        # [B, D] @ [D, N] -> [B, N]
        negative_sim_raw = torch.mm(query_embeddings, negative_embeddings.t())
        negative_sim = negative_sim_raw / self.temperature

        # Combine positive and negative similarities
        # [B, 1 + N] where first column is positive
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)

        # Labels are 0 (first position = positive)
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)

        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy (is positive the highest similarity?)
        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == labels).float().mean()

            # Average positive and negative similarities (use raw values before temperature scaling)
            # These are actual cosine similarities in range [-1, 1] (typically [0, 1] for normalized embeddings)
            avg_positive_sim = positive_sim_raw.mean()
            avg_negative_sim = negative_sim_raw.mean()

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'avg_positive_sim': avg_positive_sim.item(),
            'avg_negative_sim': avg_negative_sim.item(),
        }

        return loss, metrics
