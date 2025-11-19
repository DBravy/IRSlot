"""
Memory bank for instance recognition with momentum updates.
"""
import torch
import torch.nn as nn


class MemoryBank(nn.Module):
    """
    Memory bank that stores embeddings for each grid in the dataset.

    Uses momentum updates to slowly incorporate new embeddings.
    """
    def __init__(self, num_grids, embedding_dim, momentum=0.5):
        """
        Args:
            num_grids: Total number of unique grids in dataset
            embedding_dim: Dimension of embedding vectors
            momentum: Momentum coefficient for updates (0 = no update, 1 = full update)
        """
        super().__init__()
        self.num_grids = num_grids
        self.embedding_dim = embedding_dim
        self.momentum = momentum

        # Initialize memory bank with random unit vectors
        initial_bank = torch.randn(num_grids, embedding_dim)
        initial_bank = torch.nn.functional.normalize(initial_bank, dim=1)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('bank', initial_bank)

    @torch.no_grad()
    def update(self, grid_ids, new_embeddings):
        """
        Update memory bank with new embeddings using momentum.

        Args:
            grid_ids: [B] - Grid identifiers
            new_embeddings: [B, embedding_dim] - New embeddings from current forward pass

        Formula:
            bank[id] = momentum * bank[id] + (1 - momentum) * new_embedding
        """
        # Normalize new embeddings
        new_embeddings = torch.nn.functional.normalize(new_embeddings, dim=1)

        # Get stored embeddings
        stored_embeddings = self.bank[grid_ids]

        # Momentum update
        updated = self.momentum * stored_embeddings + (1 - self.momentum) * new_embeddings
        updated = torch.nn.functional.normalize(updated, dim=1)

        # Update bank
        self.bank[grid_ids] = updated

    def get(self, grid_ids):
        """
        Retrieve embeddings from memory bank.

        Args:
            grid_ids: [B] - Grid identifiers

        Returns:
            embeddings: [B, embedding_dim]
        """
        return self.bank[grid_ids]

    def sample_negatives(self, num_negatives, exclude_ids=None):
        """
        Sample random negative embeddings from the memory bank.

        Args:
            num_negatives: Number of negatives to sample
            exclude_ids: Optional tensor of grid IDs to exclude from sampling

        Returns:
            negative_embeddings: [num_negatives, embedding_dim]
        """
        # Create pool of valid indices
        if exclude_ids is not None:
            # Create mask of valid indices
            mask = torch.ones(self.num_grids, dtype=torch.bool, device=self.bank.device)
            mask[exclude_ids] = False
            valid_indices = torch.where(mask)[0]

            # Sample from valid indices
            if len(valid_indices) < num_negatives:
                # If not enough valid indices, sample with replacement
                sampled_indices = valid_indices[torch.randint(0, len(valid_indices), (num_negatives,))]
            else:
                # Sample without replacement
                sampled_indices = valid_indices[torch.randperm(len(valid_indices))[:num_negatives]]
        else:
            # Sample random indices
            sampled_indices = torch.randperm(self.num_grids, device=self.bank.device)[:num_negatives]

        return self.bank[sampled_indices]
