from sympy.parsing.sympy_parser import transformations

```python
UpgradedCNN(
  (features): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
    (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (12): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): AdaptiveAvgPool2d(output_size=1)
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Dropout(p=0.4, inplace=False)
    (2): Linear(in_features=256, out_features=1, bias=True)
  )
)
```

# validation, test = 50 malignant/1000 benign
external data = https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic
# Testing models with additional malignant data
> Done without any augmentations, oversampling, putting more attention to some classes. Raw data without modifications, only normalized.

# 1:1 ratio with additional data and best model checkpointing based on recall(2764 malignant,3000 benign) lr = 0.0001

Val Loss: 0.1894 | Acc: 95.05% | Precision: 0.472 | Recall: 0.340 | F1: 0.395
Test Loss: 0.1984 | Acc: 95.14% | Precision: 0.485 | Recall: 0.320 | F1: 0.386

![external1_mc.png](images%2Fexternal1_mc.png)
![external1_plot.png](images%2Fexternal1_plot.png)
# validation, test = 100 malignant/2000 benign

# 1:1 ratio with additional data and best model checkpointing based on recall(x malignant,x benign) lr = 0.0001

![external2_mc.png](images%2Fexternal2_mc.png)
![external2_plot.png](images%2Fexternal2_plot.png)

# now trying different threshholds


# validation, test = 50 malignant/1000 benign
# 1:1 ratio with additional data and best model checkpointing based on recall(2764 malignant,2764 benign) lr = 0.0001
# threshold = 0.3
Val Loss: 0.1920 | Acc: 89.43% | Precision: 0.239 | Recall: 0.560 | F1: 0.335\
Test Loss: 0.1890 | Acc: 88.95% | Precision: 0.216 | Recall: 0.500 | F1: 0.301
![external3_mc.jpg](images%2Fexternal3_mc.jpg)
![external3_plot.jpg](images%2Fexternal3_plot.jpg)


# Model EfficientNet-B3
# 1:1 ratio with additional data and best model checkpointing based on recall(2764 malignant,2764 benign) lr = 0.0001

Val Loss: 0.1046 | Acc: 96.10% | Precision: 0.585 | Recall: 0.620 | F1: 0.602
Test Loss: 0.1239 | Acc: 95.43% | Precision: 0.516 | Recall: 0.640 | F1: 0.571
![external4_mc.jpg](images%2Fexternal4_mc.jpg)
![external4_plot.jpg](images%2Fexternal4_plot.jpg)

```python
transformations 1

A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.75
        ),

        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(
                std_range=(0.22, 0.55),
                mean_range=(0.0, 0.0),
                per_channel=True,
                p=1.0
            )
        ], p=0.7),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.0),
            A.ElasticTransform(alpha=3),
        ], p=0.7),

        A.CLAHE(clip_limit=4.0, p=0.7),

        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.5
        ),

        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.85
        ),

        A.Resize(image_size, image_size),

        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(int(image_size * 0.375), int(image_size * 0.375)),
            hole_width_range=(int(image_size * 0.375), int(image_size * 0.375)),
            fill=0,
            p=0.7
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])
```

# 1:1 ratio with additional data, transformations 1, and best model checkpointing based on recall(2764 malignant,2764 benign) lr = 0.0001

Val Loss: 0.1203 | Acc: 95.24% | Precision: 0.500 | Recall: 0.300 | F1: 0.375\
Test Loss: 0.1239 | Acc: 95.33% | Precision: 0.517 | Recall: 0.300 | F1: 0.380
![external5_mc.jpg](images%2Fexternal5_mc.jpg)
![external5_plot.jpg](images%2Fexternal5_plot.jpg)

```python
transformations 2
A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=20,
            border_mode=0,
            p=0.7
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),

        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=10,
            val_shift_limit=5,
            p=0.3
        ),

        A.Resize(image_size, image_size),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])
```
# 1:1 ratio with additional data, transformations 2, and best model checkpointing based on recall(2764 malignant,2764 benign) lr = 0.0001
Val Loss: 0.1128 | Acc: 95.52% | Precision: 0.522 | Recall: 0.700 | F1: 0.598\
Test Loss: 0.1265 | Acc: 95.33% | Precision: 0.507 | Recall: 0.760 | F1: 0.608

![external6_mc.jpg](images%2Fexternal6_mc.jpg)
![external6_plot.jpg](images%2Fexternal6_plot.jpg)

# 1:1 ratio with additional data, transformations 2, and best model checkpointing based on recall(2764 malignant,2764 benign) lr = 0.00001

Val Loss: 0.1561 | Acc: 94.38% | Precision: 0.433 | Recall: 0.580 | F1: 0.496
Test Loss: 0.1474 | Acc: 94.48% | Precision: 0.423 | Recall: 0.440 | F1: 0.431
![external8_mc.jpg](images%2Fexternal8_mc.jpg)
![external8_plot.jpg](images%2Fexternal8_plot.jpg)

# 1:2 ratio with additional data, transformations 2, and best model checkpointing based on recall(2764 malignant,2764 benign) lr = 0.0001

Val Loss: 0.1155 | Acc: 96.57% | Precision: 0.694 | Recall: 0.500 | F1: 0.581
Test Loss: 0.1275 | Acc: 95.90% | Precision: 0.595 | Recall: 0.440 | F1: 0.506

![external7_mc.jpg](images%2Fexternal7_mc.jpg)
![external7_plot.jpg](images%2Fexternal7_plot.jpg)

# 1:2 ratio with additional data, transformations 2, and best model checkpointing based on recall(2764 malignant,2764 benign) lr = 0.0001, pos_weiht = 2.0

