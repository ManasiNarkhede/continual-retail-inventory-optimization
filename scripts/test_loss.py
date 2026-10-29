import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.tft_model import AsymmetricQuantileLoss

# Test that AsymmetricQuantileLoss computes different losses for different quantiles
loss_fn = AsymmetricQuantileLoss(
    quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
    under_penalty_factor=1.5,
)

# Create synthetic data where predictions are the same for all quantiles
# but target varies
batch_size = 4
horizon = 2
n_quantiles = 7

y_pred = torch.ones(batch_size, horizon, n_quantiles) * 5.0  # all quantiles predict 5.0
y_pred.requires_grad = True

target = torch.tensor([[3.0, 7.0], [4.0, 6.0], [5.0, 5.0], [2.0, 8.0]])

loss = loss_fn.loss(y_pred, target)
print(f"Loss shape: {loss.shape}")
print(f"Loss per quantile:\n{loss.mean(dim=(0,1))}")

# Check gradients
loss.mean().backward()
print(f"\nGradient shape: {y_pred.grad.shape}")
print(f"Gradients (should differ across quantiles):\n{y_pred.grad[0, 0, :]}")

# Now test with different predictions per quantile
y_pred2 = torch.zeros(batch_size, horizon, n_quantiles)
for i, q in enumerate(loss_fn.quantiles):
    y_pred2[:, :, i] = q * 10.0  # different prediction per quantile
y_pred2.requires_grad = True

loss2 = loss_fn.loss(y_pred2, target)
print(f"\nLoss2 shape: {loss2.shape}")
print(f"Loss2 per quantile:\n{loss2.mean(dim=(0,1))}")

loss2.mean().backward()
print(f"\nGradient2 shape: {y_pred2.grad.shape}")
print(f"Gradients2:\n{y_pred2.grad[0, 0, :]}")
