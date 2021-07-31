from service import get_loaders
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
from catalyst.contrib.nn import DiceLoss, IoULoss
from torch.nn import BCEWithLogitsLoss
from catalyst.contrib.models.cv.segmentation.unet import Unet
from catalyst import dl
from Runner import CustomRunner

from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    image_dir = '../input/segmentation-full-body-mads-dataset/segmentation_full_body_mads_dataset_1192_img/images'
    mask_dir = '../input/segmentation-full-body-mads-dataset/segmentation_full_body_mads_dataset_1192_img/masks'
    image_height = 200
    image_width = 200
    batch_size = 4

    path_img = Path(image_dir)
    img_list = list(path_img.glob('*.png'))
    path_mask = Path(mask_dir)
    mask_list = list(path_mask.glob('*.png'))

    x_data, x_test, y_data, y_test = train_test_split(
                                                img_list,
                                                mask_list,
                                                test_size=0.1,
                                                random_state=42,
                                                shuffle=True)

    x_train, x_val, y_train, y_val = train_test_split(
                                                x_data,
                                                y_data,
                                                test_size=0.1,
                                                random_state=42,
                                                shuffle=True)

    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(),
            ToTensorV2(),
        ],
    )

    train_loader, val_loader, test_loader = get_loaders(
                                                x_train,
                                                y_train,
                                                x_val,
                                                y_val,
                                                x_test,
                                                y_test,
                                                batch_size,
                                                train_transform,
                                                val_transform,
                                                num_workers=2,
                                                pin_memory=True,
                                            )
    loaders = {"train": train_loader, "valid": val_loader}

    model = Unet()

    criterion = {
        "dice": DiceLoss(),
        "iou": IoULoss(),
        "bce": BCEWithLogitsLoss()
    }
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    # training
    runner = CustomRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir="./logdir",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        num_epochs=10,
        callbacks=[
            dl.CriterionCallback(
                input_key="binar",
                target_key="targets",
                metric_key="loss_dice",
                criterion_key="dice",
            ),
            dl.CriterionCallback(
                input_key="binar",
                target_key="targets",
                metric_key="loss_iou",
                criterion_key="iou",
            ),
            dl.CriterionCallback(
                input_key="logits",
                target_key="targets",
                metric_key="loss_bce",
                criterion_key="bce",
            ),
            # loss aggregation
            dl.MetricAggregationCallback(
                metric_key="loss",
                metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
                mode="weighted_sum",
            ),
            dl.OptimizerCallback(metric_key="loss"),
        ],
    )


if __name__ == '__main__':
    main()
