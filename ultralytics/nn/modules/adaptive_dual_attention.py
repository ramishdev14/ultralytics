"""
Adaptive Dual Attention module for Ultralytics YOLO.

Backward-compatible implementation supporting the three ablations:

A1:
    P4-only Adaptive Dual Attention with static learned branch fusion
    and a signed tanh residual gate.

A2:
    P4-only Adaptive Dual Attention with static learned branch fusion
    and a positive bounded residual gate.

A3:
    P4-only Adaptive Dual Attention with input-dependent dynamic fusion
    and the A1 signed tanh residual gate.

The compatibility logic is required because the A1 and A2 checkpoints
were saved before the A3 dynamic-fusion attributes were introduced.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import CBAM
from .coordatt import CoordAtt


class AdaptiveDualAttention(nn.Module):
    """
    Adaptive fusion of Coordinate Attention and CBAM.

    New instances use the A3 dynamic-fusion architecture.

    Previously saved A1 and A2 checkpoints remain executable because
    forward() detects their serialized parameter structure.
    """

    def __init__(
        self,
        c1: int,
        reduction: int = 32,
        kernel_size: int = 7,
        fusion_reduction: int = 16,
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

        if fusion_reduction <= 0:
            raise ValueError(
                "fusion_reduction must be positive, "
                f"received {fusion_reduction}."
            )

        if kernel_size not in {3, 7}:
            raise ValueError(
                "CBAM kernel_size must be either 3 or 7, "
                f"received {kernel_size}."
            )

        self.c1 = c1
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.fusion_reduction = fusion_reduction

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
        # A3: input-dependent dynamic fusion
        # -------------------------------------------------------------

        hidden_channels = max(
            c1 // fusion_reduction,
            8,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fusion_mlp = nn.Sequential(
            nn.Conv2d(
                c1,
                hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(
                inplace=True,
            ),
            nn.Conv2d(
                hidden_channels,
                2,
                kernel_size=1,
                bias=True,
            ),
        )

        # Start A3 with equal branch weighting.
        nn.init.zeros_(
            self.fusion_mlp[-1].weight
        )

        nn.init.zeros_(
            self.fusion_mlp[-1].bias
        )

        # A1-style signed residual gate.
        self.residual_gate = nn.Parameter(
            torch.zeros(1)
        )

    def _static_fusion(
        self,
        coordinate_features: torch.Tensor,
        cbam_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the static learned branch fusion used by A1 and A2.
        """

        branch_weights = torch.softmax(
            self.branch_logits,
            dim=0,
        )

        coordinate_weight = branch_weights[0]
        cbam_weight = branch_weights[1]

        fused_features = (
            coordinate_weight
            * coordinate_features
            +
            cbam_weight
            * cbam_features
        )

        return fused_features

    def _dynamic_fusion(
        self,
        x: torch.Tensor,
        coordinate_features: torch.Tensor,
        cbam_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the input-dependent dynamic fusion used by A3.
        """

        descriptor = self.global_pool(
            x
        )

        fusion_logits = self.fusion_mlp(
            descriptor
        )

        branch_weights = torch.softmax(
            fusion_logits,
            dim=1,
        )

        coordinate_weight = branch_weights[
            :,
            0:1,
            :,
            :,
        ]

        cbam_weight = branch_weights[
            :,
            1:2,
            :,
            :,
        ]

        fused_features = (
            coordinate_weight
            * coordinate_features
            +
            cbam_weight
            * cbam_features
        )

        return fused_features

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the correct Adaptive Dual Attention variant.

        Variant detection:

        A3:
            fusion_mlp and global_pool exist.

        A2:
            branch_logits and residual_logit exist.

        A1:
            branch_logits and residual_gate exist.
        """

        # -------------------------------------------------------------
        # Shared attention branches
        # -------------------------------------------------------------

        coordinate_features = (
            self.coordinate_attention(x)
        )

        cbam_features = (
            self.cbam(x)
        )

        # =============================================================
        # A3
        # Dynamic sample-dependent branch fusion
        # + signed tanh residual gate
        # =============================================================

        if (
            hasattr(self, "global_pool")
            and hasattr(self, "fusion_mlp")
        ):
            fused_features = self._dynamic_fusion(
                x,
                coordinate_features,
                cbam_features,
            )

            if not hasattr(
                self,
                "residual_gate",
            ):
                raise RuntimeError(
                    "A3 AdaptiveDualAttention is missing "
                    "'residual_gate'."
                )

            gate = torch.tanh(
                self.residual_gate
            )

            return (
                x
                + gate
                * (
                    fused_features
                    - x
                )
            )

        # =============================================================
        # Older checkpoint variants require branch_logits
        # =============================================================

        if not hasattr(
            self,
            "branch_logits",
        ):
            raise RuntimeError(
                "AdaptiveDualAttention checkpoint does not contain "
                "either A3 dynamic-fusion attributes or the "
                "A1/A2 'branch_logits' parameter."
            )

        fused_features = self._static_fusion(
            coordinate_features,
            cbam_features,
        )

        # =============================================================
        # A2
        # Static learned branch fusion
        # + positive bounded residual gate
        # =============================================================

        if hasattr(
            self,
            "residual_logit",
        ):
            gate = torch.sigmoid(
                self.residual_logit
            )

            return (
                x
                + gate
                * (
                    fused_features
                    - x
                )
            )

        # =============================================================
        # A1
        # Static learned branch fusion
        # + signed tanh residual gate
        # =============================================================

        if hasattr(
            self,
            "residual_gate",
        ):
            gate = torch.tanh(
                self.residual_gate
            )

            return (
                x
                + gate
                * (
                    fused_features
                    - x
                )
            )

        raise RuntimeError(
            "Unable to determine AdaptiveDualAttention variant. "
            "Expected A1 residual_gate, A2 residual_logit, "
            "or A3 dynamic-fusion attributes."
        )