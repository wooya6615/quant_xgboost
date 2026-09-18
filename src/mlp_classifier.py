"""
BASE 13개 스칼라 피처용 MLP 이진분류기.

KAN 실험(src/kan_classifier.py)과 정확히 같은 구조([13, 16, 2])를 씀 --
"엣지에 학습되는 곡선(KAN)" vs "노드에 고정 활성화 + 엣지에 숫자 가중치(MLP)"라는
표현력 형태 차이 하나만 비교하려는 목적이라, hidden width를 다르게 주면 그 차이가
용량 차이인지 형태 차이인지 구분이 안 됨. dropout도 KAN 쪽에 없었으므로 여기도
안 넣음 (공정한 통제 비교).
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, n_features: int = 13, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)