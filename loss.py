"""
Loss functions for instance recognition training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
