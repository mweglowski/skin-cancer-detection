

# Testing different malignant / benign ratios
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
# 1:1 ratio (293 malignant, 293 benign)

Val Loss: 0.5680 | Acc: 83.33% | Precision: 0.198 | Recall: 0.820 | F1: 0.319\
Test Loss: 0.5137 | Acc: 82.76% | Precision: 0.187 | Recall: 0.780 | F1: 0.301
![balance_test_cm_1_to_1.jpg](images%2Fbalance_test_cm_1_to_1.jpg)
![balance_test_plots_1_to_1.jpg](images%2Fbalance_test_plots_1_to_1.jpg)

# 1:2 ratio (293 malignant, 586 benign)
Val Loss: 0.3223 | Acc: 89.52% | Precision: 0.246 | Recall: 0.580 | F1: 0.345\
Test Loss: 0.2757 | Acc: 92.00% | Precision: 0.337 | Recall: 0.700 | F1: 0.455
![balance_test_cm_1_to_1.jpg](images%2Fbalance_test_cm_1_to_2.jpg)
![balance_test_plots_1_to_1.jpg](images%2Fbalance_test_plots_1_to_2.jpg)
# 1:3 ratio (293 malignant, 879 benign)
Val Loss: 0.2121 | Acc: 94.19% | Precision: 0.400 | Recall: 0.440 | F1: 0.419\
Test Loss: 0.2608 | Acc: 92.86% | Precision: 0.329 | Recall: 0.480 | F1: 0.390
![balance_test_cm_1_to_1.jpg](images%2Fbalance_test_cm_1_to_3.jpg)
![balance_test_plots_1_to_1.jpg](images%2Fbalance_test_plots_1_to_3.jpg)
# 1:5 ratio (293 malignant, 1465 benign)
Val Loss: 0.2007 | Acc: 94.67% | Precision: 0.429 | Recall: 0.360 | F1: 0.391\
Test Loss: 0.2428 | Acc: 94.76% | Precision: 0.442 | Recall: 0.380 | F1: 0.409
![balance_test_cm_1_to_1.jpg](images%2Fbalance_test_cm_1_to_5.jpg)
![balance_test_plots_1_to_1.jpg](images%2Fbalance_test_plots_1_to_5.jpg)
