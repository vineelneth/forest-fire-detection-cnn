import torch.nn as nn


class FireCNN(nn.Module):
    """
    Binary CNN classifier for fire detection.

    Input:  (N, 3, 128, 128) RGB image tensor, pixel values in [0, 1].
    Output: (N,) sigmoid probability — near 0 → fire, near 1 → no fire.

    Architecture
    ------------
    Conv2d(3→32, 3x3) → ReLU → MaxPool2d(2)   # 128x128 → 63x63
    Conv2d(32→64, 3x3) → ReLU → MaxPool2d(2)  # 63x63  → 30x30
    Flatten → Linear(57600→128) → ReLU
    Linear(128→1) → Sigmoid
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 30 * 30, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
