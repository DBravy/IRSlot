"""
Mask supervision for slot attention training.

Self-contained module for:
1. Ground truth mask generation from grids using connectivity analysis
2. Dice loss for mask comparison
3. Hungarian matching for slot-to-object assignment
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Optional, List


# =============================================================================
# Constants
# =============================================================================

NUM_COLORS = 10

# Connectivity structures for scipy.ndimage.label
STRUCTURE_8CONN = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]], dtype=np.int32)

STRUCTURE_4CONN = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.int32)


# =============================================================================
# Background Detection
# =============================================================================

def detect_background_color(
    grid: np.ndarray,
    min_bg_coverage: float = 0.5,
    prefer_black: bool = True,
    black_min_coverage: float = 0.4
) -> Optional[int]:
    """
    Detect the background color using the largest edge-connected component heuristic.

    The background is identified as the color with the largest connected component
    that touches any edge of the grid, provided it covers at least min_bg_coverage
    of the total grid area.

    Args:
        grid: (H, W) integer color values 0-9
        min_bg_coverage: Minimum fraction of grid that must be covered to be
                        considered background (default 0.5 = 50%)
        prefer_black: If True, prefer black (0) as background when it touches
                     edges and covers at least black_min_coverage (default True)
        black_min_coverage: Minimum coverage for black to be preferred (default 0.4)

    Returns:
        The background color (0-9), or None if no clear background is detected
    """
    H, W = grid.shape
    total_cells = H * W

    # If the grid has only one color, there's no background
    unique_colors = np.unique(grid)
    if len(unique_colors) <= 1:
        return None

    # Get colors that touch any edge
    edge_colors = set()
    edge_colors.update(grid[0, :].tolist())      # top
    edge_colors.update(grid[H-1, :].tolist())    # bottom
    edge_colors.update(grid[:, 0].tolist())      # left
    edge_colors.update(grid[:, W-1].tolist())    # right

    # Prefer black (0) if it touches edges and covers enough of the grid
    if prefer_black and 0 in edge_colors:
        black_mask = (grid == 0)
        black_coverage = black_mask.sum() / total_cells
        if black_coverage >= black_min_coverage:
            return 0

    best_color = None
    best_size = 0

    for color in edge_colors:
        mask = (grid == color)
        labeled, num_features = ndimage.label(mask)

        # Find components that touch the edge
        for comp_id in range(1, num_features + 1):
            comp_mask = (labeled == comp_id)

            touches_edge = (
                comp_mask[0, :].any() or comp_mask[H-1, :].any() or
                comp_mask[:, 0].any() or comp_mask[:, W-1].any()
            )

            if touches_edge:
                size = comp_mask.sum()
                if size > best_size:
                    best_size = size
                    best_color = color

    # Only return as background if it covers enough of the grid
    if best_color is not None and best_size / total_cells >= min_bg_coverage:
        return best_color

    return None


# =============================================================================
# Object Mask Extraction
# =============================================================================

def extract_object_masks(
    grid: np.ndarray,
    num_slots: int,
    background_color: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Extract object masks using connectivity analysis.

    Uses 8-connectivity for foreground colors and 4-connectivity for background
    (allowing diagonal lines to act as barriers).

    Args:
        grid: (H, W) integer color values 0-9
        num_slots: Number of slots (slot 0 = background, slots 1+ = objects)
        background_color: If provided, use this as background. Otherwise auto-detect.

    Returns:
        masks: (num_slots, H, W) float32 masks
               - masks[0] = background mask (all background pixels)
               - masks[1:] = foreground object masks sorted by area (largest first)
               - Unused slots are zero masks
        num_objects: Number of foreground objects found (not counting background)
    """
    H, W = grid.shape

    # Detect background if not provided
    if background_color is None:
        background_color = detect_background_color(grid)

    # Initialize output masks
    masks = np.zeros((num_slots, H, W), dtype=np.float32)

    # Slot 0: background mask
    if background_color is not None:
        masks[0] = (grid == background_color).astype(np.float32)

    # Extract foreground objects
    objects: List[Tuple[np.ndarray, int]] = []  # List of (mask, area)

    for color in range(NUM_COLORS):
        # Skip background color
        if color == background_color:
            continue

        color_mask = (grid == color)
        if not color_mask.any():
            continue

        # Use 8-connectivity for foreground objects
        labeled, num_features = ndimage.label(color_mask, structure=STRUCTURE_8CONN)

        for comp_id in range(1, num_features + 1):
            comp_mask = (labeled == comp_id).astype(np.float32)
            area = int(comp_mask.sum())
            objects.append((comp_mask, area))

    # Sort objects by area (largest first)
    objects.sort(key=lambda x: x[1], reverse=True)

    # Assign objects to slots 1+
    num_objects = len(objects)
    max_foreground_slots = num_slots - 1  # Slot 0 is background

    for i, (obj_mask, _) in enumerate(objects[:max_foreground_slots]):
        masks[i + 1] = obj_mask

    return masks, num_objects


# =============================================================================
# Dice Loss
# =============================================================================

def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute Dice coefficient between predicted and target masks.

    Args:
        pred: (*, H, W) predicted soft masks
        target: (*, H, W) target binary masks
        eps: Small constant for numerical stability

    Returns:
        Dice coefficient (higher is better, 1.0 = perfect match)
    """
    # Flatten spatial dimensions
    pred_flat = pred.flatten(-2)  # (*, H*W)
    target_flat = target.flatten(-2)  # (*, H*W)

    intersection = (pred_flat * target_flat).sum(dim=-1)
    union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)

    dice = (2 * intersection + eps) / (union + eps)
    return dice


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute Dice loss (1 - Dice coefficient).

    Args:
        pred: (*, H, W) predicted soft masks
        target: (*, H, W) target binary masks
        eps: Small constant for numerical stability
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Dice loss (lower is better, 0.0 = perfect match)
    """
    dice = 1 - dice_coefficient(pred, target, eps)
    if reduction == 'mean':
        return dice.mean()
    elif reduction == 'sum':
        return dice.sum()
    return dice


# =============================================================================
# Hungarian Matching
# =============================================================================

def hungarian_match(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    num_objects: torch.Tensor
) -> List[Tuple[List[int], List[int]]]:
    """
    Find optimal assignment between predicted slots and ground truth objects.

    Uses Hungarian algorithm on Dice similarity matrix.

    Args:
        pred_masks: (B, num_pred_slots, H, W) predicted slot attention
        gt_masks: (B, num_gt_slots, H, W) ground truth object masks
        num_objects: (B,) number of valid GT objects per sample

    Returns:
        List of (pred_indices, gt_indices) tuples for each sample in batch
    """
    B = pred_masks.shape[0]
    num_pred = pred_masks.shape[1]
    num_gt = gt_masks.shape[1]

    assignments = []

    for b in range(B):
        n_obj = int(num_objects[b].item())

        # No objects to match or no prediction slots
        if n_obj == 0 or num_pred == 0:
            assignments.append(([], []))
            continue

        # Clamp n_obj to available GT slots
        n_obj = min(n_obj, num_gt)

        # Compute pairwise Dice scores
        # pred_masks[b]: (num_pred, H, W)
        # gt_masks[b, :n_obj]: (n_obj, H, W)
        pred_b = pred_masks[b].detach()  # (num_pred, H, W)
        gt_b = gt_masks[b, :n_obj].detach()  # (n_obj, H, W)

        # Compute cost matrix (negative Dice = we want to maximize Dice)
        cost_matrix = np.zeros((num_pred, n_obj), dtype=np.float32)

        for i in range(num_pred):
            for j in range(n_obj):
                # Get masks and ensure they're valid
                pred_mask = pred_b[i]  # (H, W)
                gt_mask = gt_b[j]  # (H, W)

                # Compute Dice directly to avoid shape issues
                pred_flat = pred_mask.flatten()  # (H*W,)
                gt_flat = gt_mask.flatten()  # (H*W,)

                # Handle empty masks
                if pred_flat.numel() == 0 or gt_flat.numel() == 0:
                    cost_matrix[i, j] = -0.0  # No overlap
                    continue

                intersection = (pred_flat * gt_flat).sum()
                union = pred_flat.sum() + gt_flat.sum()

                eps = 1e-6
                dice = (2 * intersection + eps) / (union + eps)
                cost_matrix[i, j] = -dice.item()  # Negative because we minimize cost

        # Hungarian algorithm
        pred_idx, gt_idx = linear_sum_assignment(cost_matrix)

        assignments.append((pred_idx.tolist(), gt_idx.tolist()))

    return assignments


# =============================================================================
# Mask Supervision Loss
# =============================================================================

class MaskSupervisionLoss(nn.Module):
    """
    Supervise slot attention masks against ground truth using Dice loss.

    Slot 0 is fixed to background. Slots 1+ are matched to foreground
    objects using Hungarian algorithm.
    """

    def __init__(self, bg_weight: float = 1.0, fg_weight: float = 1.0):
        """
        Args:
            bg_weight: Weight for background slot loss
            fg_weight: Weight for foreground slot losses
        """
        super().__init__()
        self.bg_weight = bg_weight
        self.fg_weight = fg_weight

    def forward(
        self,
        pred_attn: torch.Tensor,
        gt_masks: torch.Tensor,
        num_objects: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute mask supervision loss.

        Args:
            pred_attn: (B, num_slots, H, W) predicted slot attention
            gt_masks: (B, num_slots, H, W) ground truth masks
                      - gt_masks[:, 0] = background
                      - gt_masks[:, 1:] = foreground objects (sorted by area)
            num_objects: (B,) number of foreground objects per sample

        Returns:
            loss: Scalar loss value
            metrics: Dict with detailed metrics
        """
        B, S, H, W = pred_attn.shape

        # Slot 0: Background loss (fixed assignment)
        bg_pred = pred_attn[:, 0]  # (B, H, W)
        bg_gt = gt_masks[:, 0]  # (B, H, W)
        bg_loss = dice_loss(bg_pred, bg_gt, reduction='mean')  # Scalar

        # Slots 1+: Foreground with Hungarian matching
        fg_pred = pred_attn[:, 1:]  # (B, S-1, H, W)
        fg_gt = gt_masks[:, 1:]  # (B, S-1, H, W)

        # Get optimal assignment
        assignments = hungarian_match(fg_pred, fg_gt, num_objects)

        # Compute foreground loss based on assignments
        fg_losses = []
        total_matched = 0

        for b in range(B):
            pred_idx, gt_idx = assignments[b]

            if len(pred_idx) == 0:
                continue

            for pi, gi in zip(pred_idx, gt_idx):
                loss_i = dice_loss(fg_pred[b, pi], fg_gt[b, gi], reduction='mean')
                fg_losses.append(loss_i)
                total_matched += 1

        if fg_losses:
            fg_loss = torch.stack(fg_losses).mean()
        else:
            fg_loss = torch.tensor(0.0, device=pred_attn.device)

        # Combined loss
        total_loss = self.bg_weight * bg_loss + self.fg_weight * fg_loss

        # Metrics
        metrics = {
            'loss': total_loss.item(),
            'bg_loss': bg_loss.item(),
            'fg_loss': fg_loss.item() if fg_losses else 0.0,
            'num_matched': total_matched,
            'avg_objects': num_objects.float().mean().item(),
        }

        return total_loss, metrics


# =============================================================================
# Utility Functions
# =============================================================================

def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute IoU between predicted and target masks.

    Args:
        pred: (*, H, W) predicted soft masks
        target: (*, H, W) target binary masks
        threshold: Threshold for binarizing predictions
        eps: Small constant for numerical stability

    Returns:
        IoU score
    """
    pred_binary = (pred > threshold).float()

    pred_flat = pred_binary.flatten(-2)
    target_flat = target.flatten(-2)

    intersection = (pred_flat * target_flat).sum(dim=-1)
    union = ((pred_flat + target_flat) > 0).float().sum(dim=-1)

    iou = (intersection + eps) / (union + eps)
    return iou


def visualize_masks(
    grid: np.ndarray,
    masks: np.ndarray,
    title: str = "Object Masks"
) -> None:
    """
    Visualize grid and extracted masks (for debugging).

    Args:
        grid: (H, W) integer grid
        masks: (num_slots, H, W) object masks
        title: Plot title
    """
    import matplotlib.pyplot as plt

    num_slots = masks.shape[0]
    fig, axes = plt.subplots(1, num_slots + 1, figsize=(3 * (num_slots + 1), 3))

    # Original grid
    axes[0].imshow(grid, cmap='tab10', vmin=0, vmax=9)
    axes[0].set_title("Grid")
    axes[0].axis('off')

    # Masks
    slot_names = ["Background"] + [f"Object {i}" for i in range(1, num_slots)]
    for i, (ax, name) in enumerate(zip(axes[1:], slot_names)):
        ax.imshow(masks[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test on a simple grid
    grid = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 2, 2, 0],
        [0, 1, 1, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0, 4, 0],
        [0, 3, 3, 3, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int64)

    print("Test grid:")
    print(grid)
    print()

    # Extract masks
    num_slots = 7
    masks, num_objects = extract_object_masks(grid, num_slots)

    print(f"Detected background color: {detect_background_color(grid)}")
    print(f"Number of objects: {num_objects}")
    print(f"Masks shape: {masks.shape}")
    print()

    for i in range(num_slots):
        area = masks[i].sum()
        if area > 0:
            slot_type = "Background" if i == 0 else f"Object {i}"
            print(f"Slot {i} ({slot_type}): area = {int(area)}")

    # Test Dice loss
    print("\nTesting Dice loss:")
    pred = torch.from_numpy(masks).unsqueeze(0)  # (1, num_slots, H, W)
    target = pred.clone()

    # Perfect match
    loss = dice_loss(pred[:, 0], target[:, 0])
    print(f"Perfect match Dice loss: {loss.item():.4f}")

    # Random prediction
    random_pred = torch.rand_like(pred)
    loss = dice_loss(random_pred[:, 0], target[:, 0])
    print(f"Random prediction Dice loss: {loss.item():.4f}")

    print("\nAll tests passed!")
