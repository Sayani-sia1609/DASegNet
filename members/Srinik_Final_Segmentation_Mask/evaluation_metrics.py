
import torch
import numpy as np

from torchmetrics.classification import MulticlassConfusionMatrix


class SegmentationMetrics:
    """
    Calculate segmentation evaluation metrics using TorchMetrics.

    Supported inputs:
        Predicted masks: [H, W] or [B, H, W]
        Ground-truth masks: [H, W] or [B, H, W]
    """

    def __init__(self, num_classes, ignore_index=None):

        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, (int, np.integer))
            or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer.")

        if ignore_index is not None:
            if (
                isinstance(ignore_index, bool)
                or not isinstance(ignore_index, (int, np.integer))
            ):
                raise TypeError("ignore_index must be an integer.")

        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index

        self.confusion_matrix = MulticlassConfusionMatrix(
            num_classes=self.num_classes
        )

    def _convert_to_tensor(self, values, name):

        if isinstance(values, torch.Tensor):
            tensor = values.detach().cpu()

        elif isinstance(values, np.ndarray):
            if not np.issubdtype(values.dtype, np.integer):
                raise TypeError(f"{name} must contain integer class IDs.")

            tensor = torch.from_numpy(values.copy())

        else:
            raise TypeError(
                f"{name} must be a PyTorch tensor or NumPy array."
            )

        if tensor.dtype == torch.bool or tensor.is_floating_point():
            raise TypeError(f"{name} must contain integer class IDs.")

        if tensor.is_complex():
            raise TypeError(f"{name} cannot contain complex values.")

        return tensor.long()

    def update(self, predictions, targets):
        """
        Accumulate predictions and ground-truth labels.
        """

        predictions = self._convert_to_tensor(
            predictions, "predictions"
        )

        targets = self._convert_to_tensor(
            targets, "targets"
        )

        if predictions.ndim not in (2, 3):
            raise ValueError(
                "Predictions must have shape [H, W] or [B, H, W]."
            )

        if predictions.shape != targets.shape:
            raise ValueError(
                "Prediction and target shapes must match."
            )

        if predictions.numel() == 0:
            raise ValueError("Prediction masks cannot be empty.")

        if torch.any(predictions < 0) or torch.any(
            predictions >= self.num_classes
        ):
            raise ValueError("Predictions contain invalid class IDs.")

        # Exclude ignored ground-truth pixels
        if self.ignore_index is not None:
            valid_pixels = targets != self.ignore_index

            predictions = predictions[valid_pixels]
            targets = targets[valid_pixels]

        if targets.numel() == 0:
            return

        if torch.any(targets < 0) or torch.any(
            targets >= self.num_classes
        ):
            raise ValueError("Targets contain invalid class IDs.")

        self.confusion_matrix.update(
            predictions.reshape(-1),
            targets.reshape(-1)
        )

    def compute(self):
        """
        Calculate pixel accuracy, mean IoU and mean Dice score.
        """

        matrix = self.confusion_matrix.compute().double()

        true_positives = torch.diag(matrix)

        false_positives = matrix.sum(dim=0) - true_positives

        false_negatives = matrix.sum(dim=1) - true_positives

        total_pixels = matrix.sum()

        if total_pixels == 0:
            return {
                "pixel_accuracy": float("nan"),
                "mean_iou": float("nan"),
                "mean_dice": float("nan"),
                "per_class_iou": {},
                "per_class_dice": {},
                "confusion_matrix": matrix.tolist()
            }

        # Pixel Accuracy
        pixel_accuracy = (
            true_positives.sum() / total_pixels
        )

        # Intersection over Union
        iou_denominator = (
            true_positives + false_positives + false_negatives
        )

        iou = torch.full_like(true_positives, float("nan"))

        present_iou = iou_denominator > 0

        iou[present_iou] = (
            true_positives[present_iou]
            / iou_denominator[present_iou]
        )

        mean_iou = torch.nanmean(iou)

        # Dice Score
        dice_denominator = (
            2 * true_positives
            + false_positives
            + false_negatives
        )

        dice = torch.full_like(true_positives, float("nan"))

        present_dice = dice_denominator > 0

        dice[present_dice] = (
            2 * true_positives[present_dice]
            / dice_denominator[present_dice]
        )

        mean_dice = torch.nanmean(dice)

        return {
            "pixel_accuracy": pixel_accuracy.item(),
            "mean_iou": mean_iou.item(),
            "mean_dice": mean_dice.item(),

            "per_class_iou": {
                i: iou[i].item()
                for i in range(self.num_classes)
            },

            "per_class_dice": {
                i: dice[i].item()
                for i in range(self.num_classes)
            },

            "confusion_matrix": matrix.to(torch.int64).tolist()
        }

    def reset(self):
        """Reset accumulated evaluation results."""

        self.confusion_matrix.reset()
