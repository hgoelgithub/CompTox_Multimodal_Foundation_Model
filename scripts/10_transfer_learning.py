"""STEP 10 -- Transfer-learning helper: reuses the step-03 foundation model's
pretrained embedding as a starting point for a small task head predicting a
real downstream endpoint (MEA, hERG, DILI, or another registered endpoint
from step 09), instead of training that task from scratch.

This file intentionally does not invent endpoint labels. It provides the three
fine-tuning strategies and the low-data experiment matrix. Once a downstream
dataset contains DTXSID/SMILES plus a real label, connect it here.
"""
import argparse
import torch.nn as nn

FRACTIONS = [0.10, 0.25, 0.50, 1.00]
STRATEGIES = ["scratch", "head_only", "partial", "full"]


class TaskHead(nn.Module):
    """Small MLP bolted onto the pretrained model's shared embedding to predict
    one downstream endpoint (default: a single scalar/logit output)."""

    def __init__(self, latent_dim=256, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, output_dim),
        )

    def forward(self, embedding):
        return self.net(embedding)


def configure_finetuning(model, task_head, strategy="head_only", last_n_layers=2):
    """Freeze/unfreeze `model`'s parameters in place according to `strategy`:
    head_only trains just the new TaskHead; partial also unfreezes the last
    `last_n_layers` transformer layers + the fusion block; full unfreezes
    everything. The task head's own parameters are always trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    if strategy == "partial":
        for layer in model.transformer.layers[-last_n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in model.fusion.parameters():
            parameter.requires_grad = True
    elif strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif strategy != "head_only":
        raise ValueError("strategy must be head_only, partial, or full")

    for parameter in task_head.parameters():
        parameter.requires_grad = True


def main():
    """CLI entry point: print the chosen strategy and, with --show-plan, the
    full low-data experiment matrix (see module docstring)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["head_only", "partial", "full"], default="head_only")
    parser.add_argument("--show-plan", action="store_true")
    args = parser.parse_args()
    print("Selected fine-tuning strategy:", args.strategy)
    if args.show_plan:
        print("\nRecommended low-data comparison:")
        for fraction in FRACTIONS:
            for strategy in STRATEGIES:
                print(f"fraction={fraction:>4.0%}  strategy={strategy}")
    print("\nNo synthetic labels are generated. Use real MEA/hERG/DILI/other endpoint labels.")


if __name__ == "__main__":
    main()
