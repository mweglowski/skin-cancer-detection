Data
* Training Set: 1793 images (293 Malignant)
* Val Set: 1050 images (50 Malignant)
* Test Set: 1050 images (50 Malignant)

Patients can have multiple lesions.
IP_1117889    9184
IP_5714646    6267
IP_3921915    5568
IP_7797815    4454
IP_9577633    3583
              ... 

# Testing models
> Done without any augmentations, syntetic data, oversampling, putting more attention to some classes. Raw data without modifications, only normalized.

Model
```python
SimpleCNN(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (batchNorm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu1): ReLU(inplace=True)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc): Sequential(
    (0): Linear(in_features=131072, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=1, bias=True)
  )
)
```

**10 epochs**

![metrics history](https://i.imgur.com/PB8qrBT.png)
![confusion matrix](https://i.imgur.com/P4pXaZM.png)

Test metrics:
* loss: 0.1592
* acc: 91.33%

next training
**30 epochs**

![metrics history](https://i.imgur.com/gRWZ81F.png)
![confusion matrix](https://i.imgur.com/285jAUB.png)

Test metrics:
* loss: 0.0145
* acc: 88.57%