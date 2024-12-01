import marimo

__generated_with = "0.9.20"
app = marimo.App(width="full")


@app.cell
def __():
    import sys

    pkg_root = "/home/stevens.994/projects/saev-live"
    if pkg_root not in sys.path:
        sys.path.append(pkg_root)

    import random

    import beartype
    import einops
    import marimo as mo
    import numpy as np
    import torch
    from jaxtyping import Int, UInt8, jaxtyped
    from PIL import Image
    from torchvision.transforms import v2

    import contrib.semseg.training
    import saev.config

    return (
        Image,
        Int,
        UInt8,
        beartype,
        contrib,
        einops,
        jaxtyped,
        mo,
        np,
        pkg_root,
        random,
        saev,
        sys,
        torch,
        v2,
    )


@app.cell
def __(torch):
    ckpt_fpath = (
        "/home/stevens.994/projects/saev-live/checkpoints/semseg/model_ep99_step2000.pt"
    )
    model = torch.nn.Linear(768, 151)
    model.load_state_dict(torch.load(ckpt_fpath, weights_only=True, map_location="cpu"))
    model.eval()
    return ckpt_fpath, model


@app.cell
def __(contrib, saev):
    data_cfg = saev.config.DataLoad(
        shard_root="/local/scratch/stevens.994/cache/saev/a860104bf29d6093dd18b8e2dccd2e7efdfcd9fac35dceb932795af05187cb9f",
        scale_mean=False,
        scale_norm=False,
    )
    imgs_cfg = saev.config.Ade20kDataset(
        root="/research/nfs_su_809/workspace/stevens.994/datasets/ade20k/"
    )
    dataset = contrib.semseg.training.Dataset(data_cfg, imgs_cfg)
    return data_cfg, dataset, imgs_cfg


@app.cell
def __(v2):
    def make_img_transform():
        return v2.Compose([
            v2.Resize(size=224, interpolation=v2.InterpolationMode.NEAREST),
            v2.CenterCrop(size=(224, 224)),
            # v2.ToImage(),
            # v2.ToDtype(torch.uint8),
        ])

    img_transform = make_img_transform()
    return img_transform, make_img_transform


@app.cell
def __(Image, UInt8, beartype, jaxtyped, np, random):
    @jaxtyped(typechecker=beartype.beartype)
    def make_colors(seed: int = 42) -> UInt8[np.ndarray, "n 3"]:
        values = (0, 31, 63, 95, 127, 159, 191, 223, 255)
        colors = []
        for r in values:
            for g in values:
                for b in values:
                    colors.append((r, g, b))
        random.seed(seed)
        random.shuffle(colors)
        colors = np.array(colors, dtype=np.uint8)
        return colors

    @jaxtyped(typechecker=beartype.beartype)
    def color_map(map: UInt8[np.ndarray, "width height"]) -> Image.Image:
        colored = np.zeros((224, 224, 3), dtype=np.uint8)
        for i, color in enumerate(make_colors()):
            colored[map == i, :] = color

        return Image.fromarray(colored)

    return color_map, make_colors


@app.cell
def __(mo):
    number = mo.ui.number(start=0, stop=20_000, step=1)
    return (number,)


@app.cell
def __(number):
    number
    return


@app.cell
def __(number):
    image_i = number.value
    return (image_i,)


@app.cell
def __():
    # img_transform(dataset.imgs[image_i]["image"])
    return


@app.cell
def __(color_map, dataset, image_i):
    color_map(dataset.imgs[image_i]["segmentation"].numpy().reshape(224, 224))
    return


@app.cell
def __(color_map, dataset, image_i, model, np, torch):
    example = dataset[image_i]
    pred = (
        torch.nn.functional.interpolate(
            model(example["acts"]).max(axis=-1)[1].view((1, 1, 14, 14)).float(),
            scale_factor=16,
        )
        .view((224, 224))
        .int()
        .numpy()
        .astype(np.uint8)
    )
    color_map(pred)
    return example, pred


@app.cell
def __(color_map, dataset, einops, image_i, model, torch):
    def make_interpolated_pred(i):
        example = dataset[i]
        acts_WHD = example["acts"]
        logits_WHC = model(acts_WHD)
        logits_CWH = einops.rearrange(
            logits_WHC, "width height classes -> classes width height"
        )
        upsampled_CWH = torch.nn.functional.interpolate(
            logits_CWH.contiguous().unsqueeze(0), size=(224, 224), mode="bilinear"
        )[0]
        pred_WH = upsampled_CWH.argmax(axis=0).cpu().type(torch.uint8)
        return pred_WH

    make_interpolated_pred(image_i).numpy().shape
    color_map(make_interpolated_pred(image_i).numpy())
    return (make_interpolated_pred,)


@app.cell
def __(Image, dataset, image_i):
    Image.fromarray(dataset.imgs[image_i]["segmentation"].numpy().reshape(224, 224))
    return


@app.cell
def __(Int, Tensor, beartype, jaxtyped, torch, y_pred, y_true):
    @jaxtyped(typechecker=beartype.beartype)
    def mean_iou(
        y_pred: Int[Tensor, "batch width height"],
        y_true: Int[Tensor, "batch width height"],
        n_classes: int,
        ignore_class: int | None = 0,
        eps: float = 1e-12,
    ) -> float:
        """
        Calculate mean IoU for predicted masks.

        Arguments:
            y_pred:
            y_true:
            n_classes: Number of classes.

        Returns:
            Mean IoU as a float.
        """

        # Convert to one-hot encoded format
        pred_one_hot = torch.nn.functional.one_hot(y_pred.long(), n_classes)
        true_one_hot = torch.nn.functional.one_hot(y_true.long(), n_classes)

        if ignore_class is not None:
            pred_one_hot = torch.cat(
                (
                    pred_one_hot[..., :ignore_class],
                    pred_one_hot[..., ignore_class + 1 :],
                ),
                axis=-1,
            )
            true_one_hot = torch.cat(
                (
                    true_one_hot[..., :ignore_class],
                    true_one_hot[..., ignore_class + 1 :],
                ),
                axis=-1,
            )

        # Calculate intersection and union for all classes at once
        # Sum over height and width dimensions
        intersection = torch.logical_and(pred_one_hot, true_one_hot).sum(axis=(0, 1))
        union = torch.logical_or(pred_one_hot, true_one_hot).sum(axis=(0, 1))

        # Handle division by zero
        return ((intersection + eps) / (union + eps)).mean().item()

    mean_iou(y_pred, y_true, 151)
    return (mean_iou,)


@app.cell
def __(example, model, torch):
    y_pred = torch.nn.functional.interpolate(
        model(example["vit_acts"]).max(axis=-1)[1].view((1, 1, 14, 14)).float(),
        scale_factor=16,
    ).view((224, 224))
    return (y_pred,)


@app.cell
def __(dataset, image_i):
    y_true = dataset.segs[image_i]["segmentation"][0]
    return (y_true,)


@app.cell
def __(Dict, Int, Optional, beartype, jaxtyped, np, pred_label):
    @jaxtyped(typechecker=beartype.beartype)
    def intersect_and_union(
        y_pred: Int[np.ndarray, "w h"],
        label,
        num_labels,
        ignore_index: bool,
        label_map: Optional[Dict[int, int]] = None,
        reduce_labels: bool = False,
    ):
        """Calculate intersection and Union.
        Args:
            y_pred: (`ndarray`):
                Prediction segmentation map of shape (height, width).
            label (`ndarray`):
                Ground truth segmentation map of shape (height, width).
            num_labels (`int`):
                Number of categories.
            ignore_index (`int`):
                Index that will be ignored during evaluation.
            label_map (`dict`, *optional*):
                Mapping old labels to new labels. The parameter will work only when label is str.
            reduce_labels (`bool`, *optional*, defaults to `False`):
                Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
                and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
         Returns:
             area_intersect (`ndarray`):
                The intersection of prediction and ground truth histogram on all classes.
             area_union (`ndarray`):
                The union of prediction and ground truth histogram on all classes.
             area_pred_label (`ndarray`):
                The prediction histogram on all classes.
             area_label (`ndarray`):
                The ground truth histogram on all classes.
        """
        if label_map is not None:
            for old_id, new_id in label_map.items():
                label[label == old_id] = new_id

        # turn into Numpy arrays
        pred_label = np.array(pred_label)
        label = np.array(label)

        if reduce_labels:
            label[label == 0] = 255
            label = label - 1
            label[label == 254] = 255

        mask = label != ignore_index
        mask = np.not_equal(label, ignore_index)
        pred_label = pred_label[mask]
        label = np.array(label)[mask]

        intersect = pred_label[pred_label == label]

        area_intersect = np.histogram(
            intersect, bins=num_labels, range=(0, num_labels - 1)
        )[0]
        area_pred_label = np.histogram(
            pred_label, bins=num_labels, range=(0, num_labels - 1)
        )[0]
        area_label = np.histogram(label, bins=num_labels, range=(0, num_labels - 1))[0]

        area_union = area_pred_label + area_label - area_intersect

        return area_intersect, area_union, area_pred_label, area_label

    @jaxtyped(typechecker=beartype.beartype)
    def total_intersect_and_union(
        results,
        gt_seg_maps,
        num_labels,
        ignore_index: bool,
        label_map: Optional[Dict[int, int]] = None,
        reduce_labels: bool = False,
    ):
        """Calculate Total Intersection and Union, by calculating `intersect_and_union` for each (predicted, ground truth) pair.
        Args:
            results (`ndarray`):
                List of prediction segmentation maps, each of shape (height, width).
            gt_seg_maps (`ndarray`):
                List of ground truth segmentation maps, each of shape (height, width).
            num_labels (`int`):
                Number of categories.
            ignore_index (`int`):
                Index that will be ignored during evaluation.
            label_map (`dict`, *optional*):
                Mapping old labels to new labels. The parameter will work only when label is str.
            reduce_labels (`bool`, *optional*, defaults to `False`):
                Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
                and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
         Returns:
             total_area_intersect (`ndarray`):
                The intersection of prediction and ground truth histogram on all classes.
             total_area_union (`ndarray`):
                The union of prediction and ground truth histogram on all classes.
             total_area_pred_label (`ndarray`):
                The prediction histogram on all classes.
             total_area_label (`ndarray`):
                The ground truth histogram on all classes.
        """
        total_area_intersect = np.zeros((num_labels,), dtype=np.float64)
        total_area_union = np.zeros((num_labels,), dtype=np.float64)
        total_area_pred_label = np.zeros((num_labels,), dtype=np.float64)
        total_area_label = np.zeros((num_labels,), dtype=np.float64)
        for result, gt_seg_map in zip(results, gt_seg_maps):
            area_intersect, area_union, area_pred_label, area_label = (
                intersect_and_union(
                    result,
                    gt_seg_map,
                    num_labels,
                    ignore_index,
                    label_map,
                    reduce_labels,
                )
            )
            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_pred_label += area_pred_label
            total_area_label += area_label
        return (
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
        )

    @jaxtyped(typechecker=beartype.beartype)
    def mean_iou(
        results,
        gt_seg_maps,
        num_labels,
        ignore_index: bool,
        nan_to_num: Optional[int] = None,
        label_map: Optional[Dict[int, int]] = None,
        reduce_labels: bool = False,
    ):
        """Calculate Mean Intersection and Union (mIoU).
        Args:
            results (`ndarray`):
                List of prediction segmentation maps, each of shape (height, width).
            gt_seg_maps (`ndarray`):
                List of ground truth segmentation maps, each of shape (height, width).
            num_labels (`int`):
                Number of categories.
            ignore_index (`int`):
                Index that will be ignored during evaluation.
            nan_to_num (`int`, *optional*):
                If specified, NaN values will be replaced by the number defined by the user.
            label_map (`dict`, *optional*):
                Mapping old labels to new labels. The parameter will work only when label is str.
            reduce_labels (`bool`, *optional*, defaults to `False`):
                Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
                and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
        Returns:
            `Dict[str, float | ndarray]` comprising various elements:
            - *mean_iou* (`float`):
                Mean Intersection-over-Union (IoU averaged over all categories).
            - *mean_accuracy* (`float`):
                Mean accuracy (averaged over all categories).
            - *overall_accuracy* (`float`):
                Overall accuracy on all images.
            - *per_category_accuracy* (`ndarray` of shape `(num_labels,)`):
                Per category accuracy.
            - *per_category_iou* (`ndarray` of shape `(num_labels,)`):
                Per category IoU.
        """
        (
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
        ) = total_intersect_and_union(
            results, gt_seg_maps, num_labels, ignore_index, label_map, reduce_labels
        )

        # compute metrics
        metrics = dict()

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        iou = total_area_intersect / total_area_union
        acc = total_area_intersect / total_area_label

        metrics["mean_iou"] = np.nanmean(iou)
        metrics["mean_accuracy"] = np.nanmean(acc)
        metrics["overall_accuracy"] = all_acc
        metrics["per_category_iou"] = iou
        metrics["per_category_accuracy"] = acc

        if nan_to_num is not None:
            metrics = dict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in metrics.items()
            })

        return metrics

    return intersect_and_union, mean_iou, total_intersect_and_union


@app.cell
def __():
    import tensordict

    return (tensordict,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
