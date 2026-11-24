
import torch


def generate_mask(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert segmentation logits into pixel-wise class predictions.

    Args:
        logits: Tensor of shape [B, C, H, W].

    Returns:
        Predicted mask of shape [B, H, W].
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")

    if logits.ndim != 4:
        raise ValueError(
            "Expected logits with shape [B, C, H, W]."
        )

    if any(size == 0 for size in logits.shape):
        raise ValueError("Logits dimensions must be non-zero.")

    if logits.is_complex():
        raise TypeError("Complex-valued logits are not supported.")

    predicted_mask = torch.argmax(logits, dim=1)

    return predicted_mask
