# Info about approach

### Group split (train, test)
Single patient can have multiple lesions. All images from patient x have to go into train or test, patient x cannot have lesions in both. It would be leakage.

```python
GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
```
> 20% of patients go to test set

Patient (unique) counts after splitting:
* df_train: `833`
* df_test: `209`

Malignant counts after splitting:
* df_train: `283`
* df_test: `110`

### Pipeline look
Each lgb pipeline uses undersampling and tuned params.

```python
Pipeline([
    ('sampler', RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=12)),
    ('classifier', lgb.LGBMClassifier(**lgb_params, random_state=12)),
])
```

### Cross-Validation using Stratified Group K-Folding
Groups are just patient_ids. This technique provides patients and classes balance across each fold.

* Patient x is never split across folds = lesions from patient x are not splitted across folds. No data leakage.
* Every fold has approximately the same percentage of 1's and 0's.
* Default threshold in predictions inside `cross_val_score` is `0.5`

```python
groups = df_train[group_col]

cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)

val_score = cross_val_score(
    estimator=estimator, 
    X=X_train, y=y_train, 
    cv=cv, 
    groups=groups,
    scoring=custom_metric,
)
```