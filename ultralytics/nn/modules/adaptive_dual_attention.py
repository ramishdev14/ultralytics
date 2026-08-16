"""
Adaptive Dual Attention module for Ultralytics YOLO.

This module combines Coordinate Attention and CBAM in parallel. It learns
the relative contribution of both attention branches and integrates the
fused representation through a positive, bounded residual gate.

A2 ablation:
    P4-only Adaptive Dual Attention with a positive identity-preserving
    residual gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import CBAM
from .coordatt import CoordAtt


class AdaptiveDualAttention(nn.Module):
    """
    Adaptive fusion of Coordinate Attention and CBAM.

    Args:
        c1 (int):
            Number of input and output channels.

        reduction (int):
            Reduction ratio used by Coordinate Attention.

        kernel_size (int):
            Spatial-attention kernel size used by CBAM.

    The input and output tensor dimensions are identical.

    The module contains two attention branches:

        1. Coordinate Attention
        2. CBAM

    Their outputs are combined using learnable softmax-normalized
    branch weights.

    The fused attention representation is then integrated with the
    original input using a sigmoid-constrained residual gate.

    The residual gate is initialized to -4.0. Therefore,

        sigmoid(-4.0) ≈ 0.018

    and the module initially behaves very close to an identity mapping.
    This allows the pretrained YOLO representation to dominate during
    the early stages of training while still providing a small gradient
    path through the attention branches.
    """

    def __init__(
        self,
        c1: int,
        reduction: int = 32,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        if c1 <= 0:
            raise ValueError(
                f"c1 must be positive, received {c1}."
            )

        if reduction <= 0:
            raise ValueError(
                f"reduction must be positive, received {reduction}."
            )

        if kernel_size not in {3, 7}:
            raise ValueError(
                "CBAM kernel_size must be either 3 or 7, "
                f"received {kernel_size}."
            )

        self.c1 = c1
        self.reduction = reduction
        self.kernel_size = kernel_size

        # -------------------------------------------------------------
        # Coordinate Attention branch
        # -------------------------------------------------------------

        self.coordinate_attention = CoordAtt(
            c1,
            c1,
            reduction,
        )

        # -------------------------------------------------------------
        # CBAM branch
        # -------------------------------------------------------------

        self.cbam = CBAM(
            c1,
            kernel_size,
        )

        # -------------------------------------------------------------
        # Learnable branch fusion
        # -------------------------------------------------------------
        #
        # Two trainable logits control the relative contributions of
        # Coordinate Attention and CBAM.
        #
        # Both are initialized to zero:
        #
        # softmax([0, 0]) = [0.5, 0.5]
        #
        # Therefore, neither attention mechanism is preferred at
        # initialization.
        # -------------------------------------------------------------

        self.branch_logits = nn.Parameter(
            torch.zeros(2)
        )

        # -------------------------------------------------------------
        # Positive residual gate
        # -------------------------------------------------------------
        #
        # The gate is represented as a logit and transformed using
        # sigmoid during the forward pass.
        #
        # Initial value:
        #
        # sigmoid(-4.0) ≈ 0.018
        #
        # Therefore, at initialization:
        #
        # output ≈
        #     0.982 * original_features
        #     +
        #     0.018 * attention_features
        #
        # Unlike the previous tanh gate, this gate is constrained to
        # [0, 1] and cannot assign a negative contribution to the
        # attention-refined representation.
        # -------------------------------------------------------------

        self.residual_logit = nn.Parameter(
            torch.tensor(-4.0)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Adaptive Dual Attention.

        Args:
            x:
                Input feature tensor with shape [B, C, H, W].

        Returns:
            Feature tensor with the same shape as the input.
        """

        # -------------------------------------------------------------
        # Attention branches
        # -------------------------------------------------------------

        coordinate_features = (
            self.coordinate_attention(x)
        )

        cbam_features = (
            self.cbam(x)
        )

        # -------------------------------------------------------------
        # Adaptive branch weighting
        # -------------------------------------------------------------
        #
        # The two learned logits are converted into normalized positive
        # weights whose sum equals one.
        #
        # Example:
        #
        # branch_weights[0] -> Coordinate Attention contribution
        # branch_weights[1] -> CBAM contribution
        # -------------------------------------------------------------

        branch_weights = torch.softmax(
            self.branch_logits,
            dim=0,
        )

        # -------------------------------------------------------------
        # Fuse Coordinate Attention and CBAM representations
        # -------------------------------------------------------------

        fused_features = (
            branch_weights[0]
            * coordinate_features
            +
            branch_weights[1]
            * cbam_features
        )

        # -------------------------------------------------------------
        # Positive bounded residual gate
        # -------------------------------------------------------------

        gate = torch.sigmoid(
            self.residual_logit
        )

        # -------------------------------------------------------------
        # Residual integration
        # -------------------------------------------------------------
        #
        # Equivalent to:
        #
        # output =
        #     (1 - gate) * x
        #     +
        #     gate * fused_features
        #
        # This formulation ensures a smooth transition between the
        # pretrained YOLO representation and the attention-refined
        # representation.
        # -------------------------------------------------------------

        output = (
            x
            + gate
            * (
                fused_features
                - x
            )
        )

        return output