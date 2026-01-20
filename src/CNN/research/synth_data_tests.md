validation, test = 50 malignant/1000 benign


# Testing models with synthetic data
> Done without any augmentations, oversampling, putting more attention to some classes. Raw data without modifications, only normalized.


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

# 1:2 ratio with synthetic data and best model checkpointing based on f1(293 (natural) + 3000 malignant,6586 benign) lr = 0.0001
Val Loss: 0.1788 | Acc: 95.52% | Precision: 0.560 | Recall: 0.280 | F1: 0.373\
Test Loss: 0.1721 | Acc: 95.33% | Precision: 0.517 | Recall: 0.300 | F1: 0.380

![synth1_mc.jpg](images%2Fsynth1_mc.jpg)
![synth1_plot.jpg](images%2Fsynth1_plot.jpg)

# 1:2 ratio with synthetic data and best model checkpointing based on recall(293 (natural) + 3000 malignant,6586 benign) lr = 0.0001

Val Loss: 0.4188 | Acc: 82.57% | Precision: 0.166 | Recall: 0.660 | F1: 0.265\
Test Loss: 0.3896 | Acc: 83.52% | Precision: 0.178 | Recall: 0.680 | F1: 0.282

![synth2_mc.jpg](images%2Fsynth2_mc.jpg)
![synth2_plot.jpg](images%2Fsynth2_plot.jpg)

# 1:1 ratio (293 + 15000 (synth) malignant, 15293 benign) lr = 0.0001
Val Loss: 4.3953 | Acc: 37.81% | Precision: 0.055 | Recall: 0.740 | F1: 0.102\
Test Loss: 4.4285 | Acc: 35.71% | Precision: 0.044 | Recall: 0.600 | F1: 0.082

![synth3_mc.jpg](images%2Fsynth3_mc.jpg)
![synth3_plot.jpg](images%2Fsynth3_plot.jpg)