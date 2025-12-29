"""
Simplified transformer-only ARC solver (no slot attention).

This is a baseline to test whether a plain 1-2 layer transformer
can learn ARC transformations by treating each pixel as a token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


class PixelEncoder(nn.Module):
    """
    Simple encoder: embed colors + positions, no slot attention.
    Each pixel becomes one token.
    """
    def __init__(self, num_colors: int = 10, hidden_dim: int = 64, max_size: int = 30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.color_embed = nn.Embedding(num_colors, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_size, max_size, hidden_dim) * 0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: [B, H, W] discrete color indices
        Returns:
            tokens: [B, H*W, hidden_dim]
        """
        B, H, W = grid.shape
        
        # Color embeddings + positional embeddings
        x = self.color_embed(grid) + self.pos_embed[:, :H, :W, :]
        x = x.reshape(B, H * W, self.hidden_dim)
        x = self.norm(x)
        
        return x


class TransformerReasoner(nn.Module):
    """
    Standard transformer encoder layers.
    This is the core "attention + MLP" you want to test.
    """
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


class CrossAttentionDecoder(nn.Module):
    """
    Output positions query the encoded tokens via cross-attention.
    """
    def __init__(self, hidden_dim: int = 64, num_colors: int = 10, max_size: int = 30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_colors = num_colors
        
        self.pos_embed = nn.Parameter(torch.randn(1, max_size, max_size, hidden_dim) * 0.02)
        
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_colors)
        )
        
    def forward(self, encoded: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            encoded: [B, N, hidden_dim] - encoded input tokens
            output_size: (H_out, W_out)
        Returns:
            logits: [B, num_colors, H_out, W_out]
        """
        B = encoded.shape[0]
        H, W = output_size
        
        # Position queries for output grid
        queries = self.pos_embed[:, :H, :W, :].expand(B, -1, -1, -1)
        queries = queries.reshape(B, H * W, self.hidden_dim)
        
        # Cross-attention: output positions attend to encoded input
        out, _ = self.cross_attn(queries, encoded, encoded)
        out = self.norm(queries + out)
        
        # Predict color at each output position
        logits = self.head(out)
        logits = logits.reshape(B, H, W, self.num_colors).permute(0, 3, 1, 2)
        
        return logits


class SizePredictor(nn.Module):
    """Predicts output grid size from encoded representation."""
    def __init__(self, hidden_dim: int = 64, max_size: int = 30):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_h = nn.Linear(hidden_dim, max_size)
        self.head_w = nn.Linear(hidden_dim, max_size)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = tokens.mean(dim=1)  # Global average pool
        pooled = self.pool(pooled)
        return self.head_h(pooled), self.head_w(pooled)

    def predict_size(self, tokens: torch.Tensor) -> Tuple[int, int]:
        h_logits, w_logits = self.forward(tokens)
        h = h_logits.argmax(dim=-1).item() + 1
        w = w_logits.argmax(dim=-1).item() + 1
        return (h, w)


class TransformerARCSolver(nn.Module):
    """
    Complete solver using just transformer layers (no slot attention).
    
    Architecture:
        Input Grid → Pixel Embedding → Transformer Layers → Cross-Attention Decoder → Output Grid
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        num_colors: int = 10,
        num_layers: int = 2,
        num_heads: int = 4,
        max_size: int = 30,
        dropout: float = 0.0
    ):
        super().__init__()
        self.encoder = PixelEncoder(num_colors, hidden_dim, max_size)
        self.transformer = TransformerReasoner(hidden_dim, num_heads, num_layers, dropout)
        self.decoder = CrossAttentionDecoder(hidden_dim, num_colors, max_size)
        self.size_predictor = SizePredictor(hidden_dim, max_size)
        
    def forward(
        self,
        input_grid: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_grid: [B, H, W]
            output_size: (H_out, W_out) or None to use input size
        Returns:
            dict with 'logits', 'h_logits', 'w_logits'
        """
        if output_size is None:
            output_size = (input_grid.shape[1], input_grid.shape[2])
            
        # Encode: pixels → tokens
        tokens = self.encoder(input_grid)
        
        # Transform: self-attention + MLP
        encoded = self.transformer(tokens)
        
        # Decode: cross-attention to output positions
        logits = self.decoder(encoded, output_size)
        
        # Size prediction
        h_logits, w_logits = self.size_predictor(encoded)
        
        return {
            'logits': logits,
            'h_logits': h_logits,
            'w_logits': w_logits,
            'encoded': encoded
        }

    def predict(self, input_grid: torch.Tensor, output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Get predicted output grid."""
        with torch.no_grad():
            if output_size is None:
                tokens = self.encoder(input_grid)
                encoded = self.transformer(tokens)
                output_size = self.size_predictor.predict_size(encoded)
            
            out = self.forward(input_grid, output_size)
            return out['logits'].argmax(dim=1)


# =============================================================================
# Training utilities (simplified from original)
# =============================================================================

def augment_grid(grid: torch.Tensor) -> torch.Tensor:
    """Random augmentation: flips and rotations."""
    if torch.rand(1).item() < 0.5:
        grid = grid.flip(-1)  # Horizontal flip
    if torch.rand(1).item() < 0.5:
        grid = grid.flip(-2)  # Vertical flip
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        grid = torch.rot90(grid, k, dims=[-2, -1])
    return grid


def augment_pair(inp: torch.Tensor, out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply same augmentation to input/output pair."""
    if torch.rand(1).item() < 0.5:
        inp, out = inp.flip(-1), out.flip(-1)
    if torch.rand(1).item() < 0.5:
        inp, out = inp.flip(-2), out.flip(-2)
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        inp = torch.rot90(inp, k, dims=[-2, -1])
        out = torch.rot90(out, k, dims=[-2, -1])
    return inp, out


class Trainer:
    def __init__(
        self,
        model: TransformerARCSolver,
        lr: float = 1e-3,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train_step(self, inputs: torch.Tensor, outputs: torch.Tensor) -> float:
        self.model.train()
        inputs = inputs.to(self.device)
        outputs = outputs.to(self.device)
        
        B = inputs.shape[0]
        output_size = (outputs.shape[1], outputs.shape[2])
        
        result = self.model(inputs, output_size)
        
        # Pixel classification loss
        pixel_loss = F.cross_entropy(result['logits'], outputs)
        
        # Size prediction loss
        h_target = torch.full((B,), output_size[0] - 1, device=self.device, dtype=torch.long)
        w_target = torch.full((B,), output_size[1] - 1, device=self.device, dtype=torch.long)
        size_loss = F.cross_entropy(result['h_logits'], h_target) + F.cross_entropy(result['w_logits'], w_target)
        
        loss = pixel_loss + 0.1 * size_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_pixels = 0
        
        with torch.no_grad():
            for inp, out in examples:
                inp = inp.unsqueeze(0).to(self.device)
                out = out.unsqueeze(0).to(self.device)
                output_size = (out.shape[1], out.shape[2])
                
                result = self.model(inp, output_size)
                loss = F.cross_entropy(result['logits'], out)
                total_loss += loss.item()
                
                pred = result['logits'].argmax(dim=1)
                total_correct += (pred == out).sum().item()
                total_pixels += out.numel()
        
        return total_loss / len(examples), total_correct / total_pixels
    
    def predict(self, input_grid: torch.Tensor, output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        self.model.eval()
        inp = input_grid.unsqueeze(0).to(self.device)
        pred = self.model.predict(inp, output_size)
        return pred.squeeze(0).cpu()
    
    def predict_size(self, input_grid: torch.Tensor) -> Tuple[int, int]:
        self.model.eval()
        inp = input_grid.unsqueeze(0).to(self.device)
        with torch.no_grad():
            tokens = self.model.encoder(inp)
            encoded = self.model.transformer(tokens)
            return self.model.size_predictor.predict_size(encoded)


def fit_puzzle(
    train_examples: List[Tuple[torch.Tensor, torch.Tensor]],
    test_input: torch.Tensor,
    num_epochs: int = 500,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    lr: float = 1e-3,
    augment_factor: int = 50,
    device: str = 'cuda',
    verbose: bool = True
) -> Tuple[torch.Tensor, TransformerARCSolver, Tuple[int, int]]:
    """
    Fit model to a single puzzle.
    """
    model = TransformerARCSolver(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    trainer = Trainer(model, lr=lr, device=device)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        # Create augmented batch
        inputs, outputs = [], []
        for inp, out in train_examples:
            for _ in range(augment_factor):
                aug_inp, aug_out = augment_pair(inp.clone(), out.clone())
                inputs.append(aug_inp)
                outputs.append(aug_out)
        
        # Batch by output size (simplified: just iterate)
        for inp, out in zip(inputs, outputs):
            loss = trainer.train_step(inp.unsqueeze(0), out.unsqueeze(0))
            epoch_loss += loss
        
        if verbose and (epoch + 1) % 100 == 0:
            train_loss, pixel_acc = trainer.evaluate(train_examples)
            print(f"Epoch {epoch+1}: loss={train_loss:.4f}, pixel_acc={pixel_acc:.4f}")
    
    # Predict
    predicted_size = trainer.predict_size(test_input)
    prediction = trainer.predict(test_input, output_size=None)
    
    if verbose:
        print(f"Predicted size: {predicted_size}")
    
    return prediction, model, predicted_size


# =============================================================================
# Puzzle loading (same as original)
# =============================================================================

def load_arc_puzzle(puzzle_id: str, data_dir: str = 'kaggle/combined', arc_version: int = 1):
    import json
    import os
    
    if arc_version == 1:
        challenge_files = ['arc-agi_training_challenges.json', 'arc-agi_evaluation_challenges.json']
        solution_files = ['arc-agi_training_solutions.json', 'arc-agi_evaluation_solutions.json']
    else:
        challenge_files = ['arc-agi_training_challenges.json', 'arc-agi_training2_challenges.json',
                          'arc-agi_evaluation_challenges.json', 'arc-agi_evaluation2_challenges.json']
        solution_files = ['arc-agi_training_solutions.json', 'arc-agi_training2_solutions.json',
                         'arc-agi_evaluation_solutions.json', 'arc-agi_evaluation2_solutions.json']
    
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
    
    parser = argparse.ArgumentParser(description='Transformer-only ARC solver (no slot attention)')
    parser.add_argument('puzzle_id', type=str, help='ARC puzzle ID')
    parser.add_argument('--data-dir', type=str, default='kaggle/combined')
    parser.add_argument('--arc-version', type=int, default=1, choices=[1, 2])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=5, help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--augment-factor', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Loading puzzle: {args.puzzle_id}")
    train_examples, test_input, test_output = load_arc_puzzle(
        args.puzzle_id, args.data_dir, args.arc_version
    )
    
    print(f"Train examples: {len(train_examples)}")
    print(f"Test input shape: {test_input.shape}")
    
    prediction, model, predicted_size = fit_puzzle(
        train_examples,
        test_input,
        num_epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lr=args.lr,
        augment_factor=args.augment_factor,
        device=args.device,
    )
    
    print(f"\nPrediction shape: {prediction.shape}")
    
    if test_output is not None:
        size_correct = (predicted_size[0] == test_output.shape[0] and
                       predicted_size[1] == test_output.shape[1])
        print(f"Size prediction: {'CORRECT' if size_correct else 'WRONG'}")

        if size_correct:
            accuracy = (prediction == test_output).float().mean().item()
            exact_match = (prediction == test_output).all().item()
            print(f"Pixel accuracy: {accuracy:.4f}")
            print(f"Exact match: {exact_match}")

        # Print grids to console
        print("\n" + "="*40)
        print("PREDICTION:")
        print("="*40)
        for row in prediction.tolist():
            print(" ".join(str(c) for c in row))

        print("\n" + "="*40)
        print("CORRECT ANSWER:")
        print("="*40)
        for row in test_output.tolist():
            print(" ".join(str(c) for c in row))