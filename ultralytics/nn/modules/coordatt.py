# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Coordinate Attention module."""

from __future__ import annotations

import torch
import torch.nn as nn


class HSigmoid(nn.Module):
    """Computationally efficient hard-sigmoid activation."""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hard-sigmoid activation."""
        return self.relu6(x + 3.0) / 6.0


class HSwish(nn.Module):
    """Computationally efficient hard-swish activation."""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.hsigmoid = HSigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hard-swish activation."""
        return x * self.hsigmoid(x)


class CoordAtt(nn.Module):
    """
    Coordinate Attention block.

    The module separately encodes spatial information along the height and
    width dimensions and generates direction-aware channel attention maps.

    Args:
        c1: Number of input channels.
        c2: Number of output channels. Coordinate Attention preserves the
            channel count, so c1 and c2 must be equal.
        reduction: Channel-reduction ratio used inside the attention block.
        min_channels: Minimum number of intermediate channels.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        reduction: int = 32,
        min_channels: int = 8,
    ) -> None:
        super().__init__()

        if c1 != c2:
            raise ValueError(
                "CoordAtt preserves the channel dimension, but received "
                f"c1={c1} and c2={c2}."
            )

        if reduction <= 0:
            raise ValueError(
                f"reduction must be positive, received {reduction}."
            )

        self.c1 = c1
        self.c2 = c2
        self.reduction = reduction

        hidden_channels = max(
            min_channels,
            c1 // reduction,
        )

        self.conv1 = nn.Conv2d(
            c1,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            hidden_channels
        )

        self.act = HSwish()

        self.conv_h = nn.Conv2d(
            hidden_channels,
            c2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.conv_w = nn.Conv2d(
            hidden_channels,
            c2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply coordinate attention to an input feature map."""

        identity = x

        height = x.size(2)
        width = x.size(3)

        # Aggregate information independently along width and height.
        x_h = x.mean(
            dim=3,
            keepdim=True,
        )

        x_w = x.mean(
            dim=2,
            keepdim=True,
        ).permute(
            0,
            1,
            3,
            2,
        )

        # Combine the two directional descriptors.
        y = torch.cat(
            [x_h, x_w],
            dim=2,
        )

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Recover height- and width-specific descriptors.
        y_h, y_w = torch.split(
            y,
            [height, width],
            dim=2,
        )

        y_w = y_w.permute(
            0,
            1,
            3,
            2,
        )

        attention_h = self.conv_h(
            y_h
        ).sigmoid()

        attention_w = self.conv_w(
            y_w
        ).sigmoid()

        return (
            identity
            * attention_h
            * attention_w
        )