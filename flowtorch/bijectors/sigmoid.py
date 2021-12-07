# Copyright (c) Meta Platforms, Inc

from typing import Optional

import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.fixed import Fixed
from flowtorch.ops import clipped_sigmoid


class Sigmoid(Fixed):
    codomain = constraints.unit_interval

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return clipped_sigmoid(x)

    def _inverse(
        self,
        y: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return y.log() - torch.log1p(-y)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return -F.softplus(-x) - F.softplus(x)


class InverseSigmoid(Fixed):
    domain: constraints.Constraint = constraints._IndependentConstraint(
        constraints.unit_interval, 1
    )
    codomain: constraints.Constraint = constraints._IndependentConstraint(
        constraints.real, 1
    )

    # def __init__(self, shape):
    #     super().__init__(shape=shape)
    #     self.domain.event_dim = 1
    #     self.codomain.event_dim = 1

    def _forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        finfo = torch.finfo(x.dtype)
        x = x.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return x.log() - torch.log1p(-x)

    def _inverse(
        self,
        y: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return clipped_sigmoid(y)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return (F.softplus(x) + F.softplus(-x)).sum(1)
