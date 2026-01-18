## Gradient Boosting Approach


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
Each lgb pipeline uses undersampling and tuned params. `LightGBM` creates tree **leaf-wise**. `RandomForest` creates usually **level-wise** for example.

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


### Diagnosed using Partial AUC
Partial AUC is our `custom_metric()`. It calculates the area under ROC curve from `0.8` tpr (recall).

So that our metric values is in range `[0.0, 0.2]`.

![Partial AUC Graph](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4972760%2Ff9089439a6256c84a1d98a83910a46a0%2FJiang.png?generation=1717599679915410&alt=media)
> On the graph there is AUC starting from 0.9 tpr.


### LightGBM Tuning using Optuna

**How it was tuned?**
50 trials, in each trial model built 200 trees (`n_iter`).

```python
def lgb_objective(trial):
    params = {
        # type of task in classification
        'objective': 'binary',

        # whether info is shown
        'verbosity': -1, 

        # number of trees in the model
        'n_iter': 200, 

        # gradient boosting decistion tree (classic boosting)
        'boosting_type': 'gbdt', 

        # regularization (Lasso), adds penalty for non-zero weights, encourage to use fewer features
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True), 

        # (Ridge) adds penalty for large weights, encourages small, stable weights, prevents from domination of some features
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),

        # how much should new tree affect previous results
        'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True),

        # max depth of a tree
        'max_depth': trial.suggest_int('max_depth', 4, 8), 

        # num of leaves in whole model, it should be < 2**max_depth
        'num_leaves': trial.suggest_int('num_leaves', 16, 256), 

        # for each tree select only x% of features
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0), 

        # for each split (node) in a tree select x% of available features
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0), 

        # for each iteration use only x% trainig data rows
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0), 

        # per how many trees data should be reshuffled
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7), 

        # min num of samples required for leaf creation in a single tree
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100), 

        # weighting for positive class (malignant)
        'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0.8, 4.0), 
    }

    estimator = Pipeline([
        ('sampler', RandomUnderSampler(sampling_strategy=sampling_ratio)),
        ('classifier', lgb.LGBMClassifier(**params)),
    ])

    X = df_train[feature_cols]
    y = df_train[target_col]
    groups = df_train[group_col]
    cv = StratifiedGroupKFold(5, shuffle=True)

    val_score = cross_val_score(
        estimator=estimator, 
        X=X, y=y, 
        cv=cv, 
        groups=groups,
        scoring=custom_metric,
    )

    return np.mean(val_score)

study = optuna.create_study(direction='maximize')
study.optimize(lgb_objective, n_trials=50)
```

**Best params from our tuning**

Best value: `0.17595183302527378`
```python
{'lambda_l1': 0.050475696387308554,
 'lambda_l2': 0.004604923273849557,
 'learning_rate': 0.03706854419039463,
 'max_depth': 7,
 'num_leaves': 244,
 'colsample_bytree': 0.8964672653963122,
 'colsample_bynode': 0.4934008311812317,
 'bagging_fraction': 0.6447666943905135,
 'bagging_freq': 6,
 'min_data_in_leaf': 71,
 'scale_pos_weight': 1.616949356992221}
```

**Best params from kaggle**
```python
{
    'lambda_l1':        0.03335206514282942, 
    'lambda_l2':        0.005157393323802471, 
    'learning_rate':    0.030665870185795318, 
    'max_depth':        7, 
    'num_leaves':       239, 
    'colsample_bytree': 0.7573175155547233, 
    'colsample_bynode': 0.5005423904042993, 
    'bagging_fraction': 0.7937347683420382, 
    'bagging_freq':     4, 
    'min_data_in_leaf': 29, 
    'scale_pos_weight': 1.648349898918236,
}
```

**Optimization history**

![Optimization history chart](../../images/gbdt/optimization_history_lightgbm.jpg)

**Most important params**

![Most important params in study](../../images/gbdt/hyperparameter_importances_lightgbm.jpg)

### Results 

**Estimator**

Our estimator is a soft `VotingClassifier` which contains 5 pipelines. 

Every pipeline includes under sampling with `sampling_ratio=0.01` and `LightGBMClassifer`.

```python
estimator = VotingClassifier([
    # x5 of them
    ('lgb1', Pipeline([
        ('sampler', RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=12)),
        ('classifier', lgb.LGBMClassifier(**lgb_params, random_state=12)),
    ])),
], voting='soft')
```

**Params from kaggle**

- Test Partial AUC: `0.1709` ***Worse than ours!*** 🥳

| Precision | Recall | F1-Score |
| :--- | :--- | :--- |
| 0.2561 | 0.1909 | 0.2188 |

#### Confusion Matrix Test
![CM Fold Test](../../images/gbdt/11/cm_fold_test.jpg)

**Our tuned params**

Test Partial AUC: `0.1732` 🏆

* Threshold: `0.5`

    | Precision | Recall | F1-Score |
    | :--- | :--- | :--- |
    | 0.2360 | 0.1909 | 0.2111 |

    #### Confusion Matrix Test
    ![CM Fold Test](../../images/gbdt/12/cm_fold_test.jpg)

* Threshold: `0.1`

    | Precision | Recall | F1-Score |
    | :--- | :--- | :--- |
    | 0.1003 | 0.5545 | 0.1699 |

    #### Confusion Matrix Test
    ![CM Fold Test](../../images/gbdt/13/cm_fold_test.jpg)

* Threshold: `0.9`

    | Precision | Recall | F1-Score |
    | :--- | :--- | :--- |
    | 0.3333 | 0.0091 | 0.0177 |

    #### Confusion Matrix Test
    ![CM Fold Test](../../images/gbdt/14/cm_fold_test.jpg)