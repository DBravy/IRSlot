"""
Per-puzzle ARC solver.

Approach: Train a small network to memorize one puzzle's transformation rule,
then test if it generalizes to the held-out test input.

The architecture is intentionally compact:
- Slot attention for object perception
- Small transformer for reasoning
- Cross-attention decoder for output generation

Training uses heavy augmentation to prevent pixel-level memorization
and encourage learning the actual rule.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import List, Tuple, Optional, Dict


# =============================================================================
# Core Architecture
# =============================================================================

class SlotEncoder(nn.Module):
    """
    Lightweight slot attention encoder.
    Extracts object slots with explicit properties.
    """
    def __init__(
        self,
        num_slots: int = 10,
        slot_dim: int = 48,
        num_colors: int = 10,
        num_iterations: int = 3,
        max_size: int = 30
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_colors = num_colors
        
        # Input embedding
        self.color_embed = nn.Embedding(num_colors, slot_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_size, max_size, slot_dim) * 0.02)
        
        # Slot attention
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
        self.slot_sigma = nn.Parameter(torch.ones(1, 1, slot_dim) * 0.1)
        
        self.norm_in = nn.LayerNorm(slot_dim)
        self.norm_slot = nn.LayerNorm(slot_dim)
        
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)
        
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim)
        )
        
        self.num_iterations = num_iterations
        
        # Property dimensions: size(1) + centroid(2) + bbox(4) + colors(num_colors)
        self.prop_dim = 7 + num_colors
        
    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            grid: [B, H, W] input grid
            
        Returns:
            slots: [B, num_slots, slot_dim]
            attn: [B, num_slots, H, W]
            props: [B, num_slots, prop_dim]
        """
        B, H, W = grid.shape
        device = grid.device
        
        # Embed input
        x = self.color_embed(grid) + self.pos_embed[:, :H, :W, :]
        x = x.reshape(B, H * W, -1)
        x = self.norm_in(x)
        
        # Initialize slots
        slots = self.slot_mu + self.slot_sigma * torch.randn(
            B, self.num_slots, self.slot_dim, device=device
        )
        
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Iterate
        for _ in range(self.num_iterations):
            slots_prev = slots
            q = self.to_q(self.norm_slot(slots))
            
            attn_logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.slot_dim)
            attn = F.softmax(attn_logits, dim=1)  # Competition over slots
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            
            updates = torch.bmm(attn, v)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(B, self.num_slots, -1)
            slots = slots + self.mlp(slots)
        
        # Final attention
        q = self.to_q(self.norm_slot(slots))
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.slot_dim)
        attn = F.softmax(attn_logits, dim=1)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        attn = attn.reshape(B, self.num_slots, H, W)
        
        # Compute properties
        props = self._compute_props(attn, grid)
        
        return slots, attn, props
    
    def _compute_props(self, attn: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Compute explicit object properties from attention masks."""
        B, num_slots, H, W = attn.shape
        device = attn.device
        
        # Coordinate grids (normalized to [0, 1])
        yy = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, num_slots, H, W)
        xx = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, num_slots, H, W)
        
        # Size (attention mass)
        size = attn.sum(dim=[2, 3])  # [B, num_slots]
        size_norm = size / (H * W)
        
        # Centroid
        attn_safe = attn + 1e-8
        centroid_y = (attn_safe * yy).sum(dim=[2, 3]) / attn_safe.sum(dim=[2, 3])
        centroid_x = (attn_safe * xx).sum(dim=[2, 3]) / attn_safe.sum(dim=[2, 3])
        
        # Bounding box (approximate via weighted std)
        var_y = (attn_safe * (yy - centroid_y.unsqueeze(-1).unsqueeze(-1))**2).sum(dim=[2, 3]) / attn_safe.sum(dim=[2, 3])
        var_x = (attn_safe * (xx - centroid_x.unsqueeze(-1).unsqueeze(-1))**2).sum(dim=[2, 3]) / attn_safe.sum(dim=[2, 3])
        height = 2 * torch.sqrt(var_y + 1e-8)  # Approximate extent
        width = 2 * torch.sqrt(var_x + 1e-8)
        
        # Color distribution
        colors_onehot = F.one_hot(grid, self.num_colors).float()  # [B, H, W, C]
        colors_onehot = colors_onehot.permute(0, 3, 1, 2).unsqueeze(1)  # [B, 1, C, H, W]
        attn_exp = attn.unsqueeze(2)  # [B, num_slots, 1, H, W]
        color_dist = (attn_exp * colors_onehot).sum(dim=[3, 4])  # [B, num_slots, C]
        color_dist = color_dist / (color_dist.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Concatenate all properties
        props = torch.cat([
            size_norm.unsqueeze(-1),
            centroid_y.unsqueeze(-1),
            centroid_x.unsqueeze(-1),
            height.unsqueeze(-1),
            width.unsqueeze(-1),
            (height * width).unsqueeze(-1),  # Area proxy
            (height / (width + 1e-8)).unsqueeze(-1),  # Aspect ratio
            color_dist
        ], dim=-1)
        
        return props


class Reasoner(nn.Module):
    """
    Small transformer for slot-to-slot reasoning.
    """
    def __init__(self, slot_dim: int = 48, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=slot_dim,
                nhead=num_heads,
                dim_feedforward=slot_dim * 2,
                dropout=0.0,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            slots = layer(slots)
        return slots


class Decoder(nn.Module):
    """
    Cross-attention decoder: output positions query slots.
    """
    def __init__(self, slot_dim: int = 48, num_colors: int = 10, max_size: int = 30):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_colors = num_colors
        
        self.pos_embed = nn.Parameter(torch.randn(1, max_size, max_size, slot_dim) * 0.02)
        
        self.cross_attn = nn.MultiheadAttention(
            slot_dim, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(slot_dim)
        
        self.head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, num_colors)
        )
        
    def forward(self, slots: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            slots: [B, num_slots, slot_dim]
            output_size: (H, W)
            
        Returns:
            logits: [B, num_colors, H, W]
        """
        B = slots.shape[0]
        H, W = output_size
        
        # Position queries
        pos = self.pos_embed[:, :H, :W, :].expand(B, -1, -1, -1)
        pos = pos.reshape(B, H * W, self.slot_dim)
        
        # Cross-attention
        out, _ = self.cross_attn(pos, slots, slots)
        out = self.norm(pos + out)
        
        # Predict colors
        logits = self.head(out)
        logits = logits.reshape(B, H, W, self.num_colors).permute(0, 3, 1, 2)
        
        return logits


class PuzzleSolver(nn.Module):
    """
    Complete puzzle solver: encode → reason → decode.
    """
    def __init__(
        self,
        num_slots: int = 10,
        slot_dim: int = 48,
        num_colors: int = 10,
        num_reasoning_layers: int = 2,
        max_size: int = 30
    ):
        super().__init__()
        self.encoder = SlotEncoder(num_slots, slot_dim, num_colors, max_size=max_size)
        self.reasoner = Reasoner(slot_dim, num_layers=num_reasoning_layers)
        self.decoder = Decoder(slot_dim, num_colors, max_size)
        
    def forward(
        self, 
        input_grid: torch.Tensor, 
        output_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_grid: [B, H, W]
            output_size: (H_out, W_out), defaults to input size
            
        Returns:
            dict with: logits, slots, attn, props
        """
        B, H, W = input_grid.shape
        if output_size is None:
            output_size = (H, W)
            
        slots, attn, props = self.encoder(input_grid)
        slots = self.reasoner(slots)
        logits = self.decoder(slots, output_size)
        
        return {
            'logits': logits,
            'slots': slots,
            'attn': attn,
            'props': props
        }
    
    def predict(self, input_grid: torch.Tensor, output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Get predicted grid."""
        out = self.forward(input_grid, output_size)
        return out['logits'].argmax(dim=1)


# =============================================================================
# Augmentations (Critical for per-puzzle training!)
# =============================================================================

class AugmentPair:
    """
    Augmentations that apply consistently to input-output pairs.
    """
    def __init__(self, p_flip: float = 0.5, p_rot: float = 0.5, p_color_perm: float = 0.5):
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.p_color_perm = p_color_perm
        
    def __call__(
        self, 
        input_grid: torch.Tensor, 
        output_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply same augmentation to input and output.
        
        Args:
            input_grid: [H_in, W_in]
            output_grid: [H_out, W_out]
        """
        # Horizontal flip
        if random.random() < self.p_flip:
            input_grid = input_grid.flip(-1)
            output_grid = output_grid.flip(-1)
            
        # Vertical flip
        if random.random() < self.p_flip:
            input_grid = input_grid.flip(-2)
            output_grid = output_grid.flip(-2)
            
        # Rotation (0, 90, 180, 270)
        if random.random() < self.p_rot:
            k = random.randint(0, 3)
            input_grid = torch.rot90(input_grid, k, dims=[-2, -1])
            output_grid = torch.rot90(output_grid, k, dims=[-2, -1])
            
        # Color permutation (preserving 0 as background)
        if random.random() < self.p_color_perm:
            # Create random permutation for colors 1-9, keep 0 fixed
            perm = torch.cat([
                torch.tensor([0]),
                torch.randperm(9) + 1
            ])
            input_grid = perm[input_grid]
            output_grid = perm[output_grid]
            
        return input_grid, output_grid


# =============================================================================
# Per-Puzzle Training
# =============================================================================

class PuzzleTrainer:
    """
    Train a model on a single puzzle's examples.
    """
    def __init__(
        self,
        model: PuzzleSolver,
        lr: float = 1e-3,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.augment = AugmentPair()
        
    def train_epoch(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]],
        augment_factor: int = 10
    ) -> float:
        """
        Train one epoch on puzzle examples.
        
        Args:
            examples: List of (input_grid, output_grid) pairs
            augment_factor: How many augmented versions per example
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        count = 0
        
        for _ in range(augment_factor):
            for input_grid, output_grid in examples:
                # Augment
                inp, out = self.augment(input_grid.clone(), output_grid.clone())
                
                # Move to device and add batch dim
                inp = inp.unsqueeze(0).to(self.device)
                out = out.unsqueeze(0).to(self.device)
                
                # Forward
                output_size = (out.shape[1], out.shape[2])
                result = self.model(inp, output_size)
                
                # Loss
                loss = F.cross_entropy(result['logits'], out)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                count += 1
                
        return total_loss / count
    
    @torch.no_grad()
    def evaluate(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[float, float]:
        """
        Evaluate on examples (no augmentation).
        
        Returns:
            (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_pixels = 0
        
        for input_grid, output_grid in examples:
            inp = input_grid.unsqueeze(0).to(self.device)
            out = output_grid.unsqueeze(0).to(self.device)
            
            output_size = (out.shape[1], out.shape[2])
            result = self.model(inp, output_size)
            
            loss = F.cross_entropy(result['logits'], out)
            total_loss += loss.item()
            
            pred = result['logits'].argmax(dim=1)
            total_correct += (pred == out).sum().item()
            total_pixels += out.numel()
            
        avg_loss = total_loss / len(examples)
        accuracy = total_correct / total_pixels
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def predict(
        self,
        input_grid: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Get prediction for test input."""
        self.model.eval()
        inp = input_grid.unsqueeze(0).to(self.device)
        return self.model.predict(inp, output_size).squeeze(0).cpu()
    
    @torch.no_grad()
    def get_slot_visualization(
        self,
        input_grid: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get slot attention masks and properties for visualization."""
        self.model.eval()
        inp = input_grid.unsqueeze(0).to(self.device)
        result = self.model(inp)
        
        return {
            'attn': result['attn'].squeeze(0).cpu(),  # [num_slots, H, W]
            'props': result['props'].squeeze(0).cpu(),  # [num_slots, prop_dim]
        }


def fit_puzzle(
    train_examples: List[Tuple[torch.Tensor, torch.Tensor]],
    test_input: torch.Tensor,
    test_output_size: Optional[Tuple[int, int]] = None,
    num_epochs: int = 500,
    num_slots: int = 10,
    slot_dim: int = 48,
    lr: float = 1e-3,
    augment_factor: int = 10,
    device: str = 'cuda',
    verbose: bool = True
) -> Tuple[torch.Tensor, PuzzleSolver]:
    """
    Fit a model to one puzzle and return prediction.
    
    Args:
        train_examples: List of (input, output) tensor pairs
        test_input: Test input grid
        test_output_size: Size of expected output (if different from input)
        num_epochs: Training epochs
        num_slots: Number of slots
        slot_dim: Slot dimension
        lr: Learning rate
        augment_factor: Augmentations per example per epoch
        device: 'cuda' or 'cpu'
        verbose: Print progress
        
    Returns:
        (predicted_output, trained_model)
    """
    # Create model
    model = PuzzleSolver(
        num_slots=num_slots,
        slot_dim=slot_dim,
        num_colors=10,
        num_reasoning_layers=2,
        max_size=30
    )
    
    trainer = PuzzleTrainer(model, lr=lr, device=device)
    
    # Train
    for epoch in range(num_epochs):
        loss = trainer.train_epoch(train_examples, augment_factor=augment_factor)
        
        if verbose and (epoch + 1) % 50 == 0:
            train_loss, train_acc = trainer.evaluate(train_examples)
            print(f"Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}")
    
    # Final evaluation
    if verbose:
        train_loss, train_acc = trainer.evaluate(train_examples)
        print(f"Final: loss={train_loss:.4f}, acc={train_acc:.4f}")
    
    # Predict test
    prediction = trainer.predict(test_input, test_output_size)
    
    return prediction, model


# =============================================================================
# Visualization
# =============================================================================

def grid_to_str(grid: torch.Tensor) -> List[str]:
    """Convert a grid to list of strings, one per row."""
    return [' '.join(str(int(c)) for c in row) for row in grid]


def visualize_prediction(
    test_input: torch.Tensor,
    prediction: torch.Tensor,
    test_output: Optional[torch.Tensor] = None,
    title: str = "ARC Puzzle Result"
):
    """
    Print grids side by side in the console.

    Args:
        test_input: [H, W] input grid
        prediction: [H, W] predicted output
        test_output: [H, W] ground truth (optional)
    """
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)

    # Convert grids to string rows
    input_rows = grid_to_str(test_input)
    pred_rows = grid_to_str(prediction)
    gt_rows = grid_to_str(test_output) if test_output is not None else None

    # Calculate widths for alignment
    input_width = max(len(r) for r in input_rows)
    pred_width = max(len(r) for r in pred_rows)

    # Headers
    if gt_rows is not None:
        gt_width = max(len(r) for r in gt_rows)
        print(f"{'Test Input':<{input_width}}   {'Prediction':<{pred_width}}   {'Ground Truth'}")
        print(f"{'-' * input_width}   {'-' * pred_width}   {'-' * gt_width}")
    else:
        print(f"{'Test Input':<{input_width}}   {'Prediction'}")
        print(f"{'-' * input_width}   {'-' * pred_width}")

    # Find max height
    max_rows = max(len(input_rows), len(pred_rows), len(gt_rows) if gt_rows else 0)

    # Print rows side by side
    for i in range(max_rows):
        inp = input_rows[i] if i < len(input_rows) else ''
        pred = pred_rows[i] if i < len(pred_rows) else ''
        if gt_rows is not None:
            gt = gt_rows[i] if i < len(gt_rows) else ''
            print(f"{inp:<{input_width}}   {pred:<{pred_width}}   {gt}")
        else:
            print(f"{inp:<{input_width}}   {pred}")

    print()


# =============================================================================
# Puzzle Loading
# =============================================================================

def load_arc_puzzle(puzzle_id: str, data_dir: str = 'kaggle/combined', arc_version: int = 1):
    """
    Load an ARC puzzle from JSON files.

    Returns:
        train_examples: List of (input_tensor, output_tensor) pairs
        test_input: Test input tensor
        test_output: Test output tensor (if available, for evaluation)
    """
    import json
    import os

    # Determine which files to search
    if arc_version == 1:
        challenge_files = ['arc-agi_training_challenges.json', 'arc-agi_evaluation_challenges.json']
        solution_files = ['arc-agi_training_solutions.json', 'arc-agi_evaluation_solutions.json']
    else:
        challenge_files = ['arc-agi_training_challenges.json', 'arc-agi_training2_challenges.json',
                          'arc-agi_evaluation_challenges.json', 'arc-agi_evaluation2_challenges.json']
        solution_files = ['arc-agi_training_solutions.json', 'arc-agi_training2_solutions.json',
                         'arc-agi_evaluation_solutions.json', 'arc-agi_evaluation2_solutions.json']

    # Find the puzzle
    puzzle_data = None
    test_solution = None

    for cf, sf in zip(challenge_files, solution_files):
        cf_path = os.path.join(data_dir, cf)
        if not os.path.exists(cf_path):
            continue
        with open(cf_path, 'r') as f:
            challenges = json.load(f)
        if puzzle_id in challenges:
            puzzle_data = challenges[puzzle_id]
            sf_path = os.path.join(data_dir, sf)
            if os.path.exists(sf_path):
                with open(sf_path, 'r') as f:
                    solutions = json.load(f)
                if puzzle_id in solutions:
                    test_solution = solutions[puzzle_id]
            break

    if puzzle_data is None:
        raise ValueError(f"Puzzle '{puzzle_id}' not found")

    # Convert to tensors
    train_examples = []
    for ex in puzzle_data['train']:
        inp = torch.tensor(ex['input'], dtype=torch.long)
        out = torch.tensor(ex['output'], dtype=torch.long)
        train_examples.append((inp, out))

    test_input = torch.tensor(puzzle_data['test'][0]['input'], dtype=torch.long)
    test_output = None
    if test_solution:
        test_output = torch.tensor(test_solution[0], dtype=torch.long)

    return train_examples, test_input, test_output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Per-puzzle ARC solver')
    parser.add_argument('puzzle_id', type=str, help='ARC puzzle ID (e.g., 23b5c85d)')
    parser.add_argument('--data-dir', type=str, default='kaggle/combined', help='Directory containing ARC JSON files')
    parser.add_argument('--arc-version', type=int, default=1, choices=[1, 2], help='ARC dataset version')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--num-slots', type=int, default=10, help='Number of slots')
    parser.add_argument('--slot-dim', type=int, default=48, help='Slot dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--augment-factor', type=int, default=10, help='Augmentations per example per epoch')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load puzzle
    print(f"Loading puzzle: {args.puzzle_id}")
    train_examples, test_input, test_output = load_arc_puzzle(
        args.puzzle_id, args.data_dir, args.arc_version
    )

    print(f"Train examples: {len(train_examples)}")
    print(f"Test input shape: {test_input.shape}")
    if test_output is not None:
        print(f"Test output shape: {test_output.shape}")

    # Fit the model
    prediction, model = fit_puzzle(
        train_examples,
        test_input,
        test_output_size=test_output.shape if test_output is not None else None,
        num_epochs=args.epochs,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        lr=args.lr,
        augment_factor=args.augment_factor,
        device=args.device,
    )

    print(f"\nPrediction shape: {prediction.shape}")
    if test_output is not None:
        accuracy = (prediction == test_output).float().mean().item()
        exact_match = (prediction == test_output).all().item()
        print(f"Pixel accuracy: {accuracy:.4f}")
        print(f"Exact match: {exact_match}")

    # Visualize results
    visualize_prediction(test_input, prediction, test_output, title=f"Puzzle: {args.puzzle_id}")