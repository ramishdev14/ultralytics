"""
Adaptive Dual Attention module for Ultralytics YOLO.

This module combines Coordinate Attention and CBAM in parallel. It learns
the relative contribution of both attention branches and integrates the
fused result through a learnable residual gate.
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

        # Coordinate Attention branch.
        self.coordinate_attention = CoordAtt(
            c1,
            c1,
            reduction,
        )

        # CBAM branch.
        self.cbam = CBAM(
            c1,
            kernel_size,
        )

        # Two trainable scores controlling the relative contribution
        # of Coordinate Attention and CBAM.
        #
        # Both are initialized equally.
        self.branch_logits = nn.Parameter(
            torch.zeros(2)
        )

        # The residual gate starts at zero because tanh(0) = 0.
        #
        # Therefore, the initial module behaves exactly as:
        #
        # output = input
        #
        # This helps preserve pretrained YOLO representations during
        # the initial training stage.
        self.residual_gate = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply adaptive dual attention.

        Args:
            x:
                Input feature tensor with shape [B, C, H, W].

        Returns:
            Feature tensor with the same shape as the input.
        """

        coordinate_features = self.coordinate_attention(x)

        cbam_features = self.cbam(x)

        # Convert branch scores into normalized positive weights.
        branch_weights = torch.softmax(
            self.branch_logits,
            dim=0,
        )

        # Adaptive fusion of both attention branches.
        fused_features = (
            branch_weights[0] * coordinate_features
            + branch_weights[1] * cbam_features
        )

        # tanh constrains the gate between -1 and 1 and starts at zero.
        gate = torch.tanh(
            self.residual_gate
        )

        # Residual integration.
        output = x + gate * (
            fused_features - x
        )

        return output