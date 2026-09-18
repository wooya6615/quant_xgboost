"""
KAN(Kolmogorov-Arnold Network) 기반 이진분류기.

MLP와 다르게 "노드에 고정 활성화, 엣지에 숫자 가중치"가 아니라 "엣지에 학습되는
B-spline 곡선, 노드는 단순 합산" 구조. 엣지마다 자기만의 비선형 곡선을 학습해서
같은 파라미터 수 대비 부드러운 비선형 관계를 더 잘 표현한다는 게 핵심 주장
(Liu et al. 2024, "KAN: Kolmogorov-Arnold Networks"). 이 파일은 외부 패키지
의존 없이 순수 PyTorch로 구현(efficient-kan 스타일 B-spline 계산 재구현) —
GRU/TCN처럼 이 레포 안에서 직접 유지보수 가능하게 하기 위함.

⚠️ 시퀀스 구조가 아님: GRU/TCN처럼 20일 윈도우를 그대로 못 받음. 여기서는
XGBoost와 동일한 BASE 13개 스칼라 피처를 입력으로 씀 (윈도우 펼치기 안 함).

⚠️ spline은 입력이 grid 범위(기본 [-1,1]) 안에 있다고 가정 -- 학습 전 fold의
train 통계로 z-score 후 반드시 clip할 것 (run_kan_base_baseline.py의
normalize_fold_kan 참고). 정규화 없이 넣으면 grid 밖에서 외삽이 불안정해짐.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """엣지마다 B-spline(+base activation) 곡선을 학습하는 KAN의 한 층."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5,
                 spline_order: int = 3, scale_noise: float = 0.1, scale_base: float = 1.0,
                 scale_spline: float = 1.0, base_activation=nn.SiLU, grid_range=(-1, 1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                self._curve_to_coeff(self.grid.T[self.spline_order:-self.spline_order], noise)
            )
            nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features) -> (batch, in_features, grid_size + spline_order)
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def _curve_to_coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        A = self._b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def _scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self._b_splines(x).view(x.size(0), -1),
            self._scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    def regularization_loss(self, regularize_activation: float = 1.0,
                             regularize_entropy: float = 1.0) -> torch.Tensor:
        # 엣지 함수의 L1 크기가 소수 엣지에 몰리지 않게 하는 entropy 정규화
        # (spline이 노이즈에 과적합하는 걸 억제하는 용도 -- KAN 논문 3.2절)
        l1 = self.spline_weight.abs().mean(-1)
        act_loss = l1.sum()
        p = l1 / (act_loss + 1e-8)
        entropy_loss = -torch.sum(p * torch.log(p + 1e-8))
        return regularize_activation * act_loss + regularize_entropy * entropy_loss


class KANClassifier(nn.Module):
    """BASE 13개 스칼라 피처 -> 이진분류. GRU/TCN classifier들과 동일한 인터페이스
    (forward(x) -> (batch, 2) logits)라서 MODEL_BUILDERS 패턴에 그대로 끼울 수 있음."""

    def __init__(self, n_features: int = 13, hidden: int = 16, grid_size: int = 5,
                 spline_order: int = 3):
        super().__init__()
        self.layer1 = KANLinear(n_features, hidden, grid_size=grid_size, spline_order=spline_order)
        self.layer2 = KANLinear(hidden, 2, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        return self.layer2(x)

    def regularization_loss(self, regularize_activation: float = 1.0,
                             regularize_entropy: float = 1.0) -> torch.Tensor:
        return (
            self.layer1.regularization_loss(regularize_activation, regularize_entropy)
            + self.layer2.regularization_loss(regularize_activation, regularize_entropy)
        )