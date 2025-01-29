# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "beartype",
#     "grad-cam",
#     "numpy",
#     "opencv-python",
#     "timm",
#     "torch",
#     "tyro",
# ]
# ///
"""
Downloads the Begula whale dataset from lila.science.
"""

import tyro
import beartype
import dataclasses
import cv2
import numpy as np
import torch

from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad,
)

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

methods = {
    "gradcam": GradCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad,
}


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    device: str = "cpu"
    """Torch device to use."""

    image_path: str = "./examples/both.png"
    """Input image path"""

    aug_smooth: bool = False
    """Apply test time augmentation to smooth the CAM"""

    eigen_smooth: bool = False
    """Reduce noise by taking the first principle component of cam_weights*activations"""

    method: str = "gradcam"
    """Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam"""


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == "__main__":
    """
    Example usage of using cam-methods on a VIT network.
    """

    args = tyro.cli(Args)

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model = (
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        .to(torch.device(args.device))
        .eval()
    )

    target_layers = [model.blocks[-1].norm1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
            ablation_layer=AblationLayerVit(),
        )
    else:
        cam = methods[args.method](
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(
        rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    ).to(args.device)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
        eigen_smooth=args.eigen_smooth,
        aug_smooth=args.aug_smooth,
    )

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f"{args.method}_cam.jpg", cam_image)
