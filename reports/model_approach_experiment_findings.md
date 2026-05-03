# Model Approach Experiment Findings

## Goal

The goal was to test whether another modeling approach could improve the genre-family classifier after the genre mapping and feature engineering changes.

The specific question was whether a model-per-class approach, such as one-vs-rest classification, would improve accuracy.

## Data and Setup

Experiments used the existing processed Spotify split:

- Train: `data/processed/splits/train.csv`
- Validation: `data/processed/splits/validation.csv`
- Test: `data/processed/splits/test.csv`

The same feature list from `configs/project.yaml` was used for all experiments.

Models were compared first on the validation split. The strongest candidates were then checked on the held-out test split.

## Approaches Tested

The following approaches were tested:

| Approach | Purpose |
| --- | --- |
| Current balanced Random Forest | Existing best baseline |
| Unweighted Random Forest | Check whether class weighting was hurting accuracy |
| Depth-limited balanced Random Forest | Check whether a shallower forest improves generalization |
| Extra Trees | More randomized tree ensemble |
| One-vs-rest Logistic Regression | Model-per-class linear baseline |
| One-vs-rest Linear SVM | Model-per-class margin-based baseline |
| One-vs-rest Random Forest | Model-per-class tree baseline |
| Soft Voting Random Forest + Logistic Regression | Small ensemble of nonlinear and linear models |
| Prior-adjusted Random Forest probabilities | Improve class balance by adjusting predicted probabilities using class priors |
| Histogram Gradient Boosting | Tried, but failed in this environment |

## Validation Results

| Approach | Accuracy | Macro F1 | Weighted F1 | Top-3 Accuracy | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Current balanced Random Forest | 0.6063 | 0.5714 | 0.5994 | 0.8550 | Strong baseline |
| Unweighted Random Forest | **0.6104** | 0.5733 | 0.6030 | 0.8557 | Best plain validation accuracy |
| Balanced Random Forest, depth 20 | 0.6072 | 0.5767 | 0.6040 | **0.8580** | Best plain top-3 accuracy |
| Extra Trees, balanced | 0.5896 | 0.5555 | 0.5836 | 0.8424 | Faster, weaker |
| Extra Trees, unweighted | 0.5902 | 0.5510 | 0.5832 | 0.8450 | Faster, weaker |
| One-vs-rest Logistic Regression | 0.4232 | 0.3819 | 0.4320 | 0.7098 | Did not help |
| One-vs-rest Linear SVM | 0.4237 | 0.3850 | 0.4349 | N/A | Did not help |
| One-vs-rest Random Forest, small | 0.5891 | 0.5573 | 0.5887 | 0.8503 | Better than linear OvR, still below multiclass RF |
| Soft vote RF + Logistic Regression | 0.6004 | 0.5684 | 0.5951 | 0.8517 | Did not beat RF |
| Balanced RF + prior adjustment | 0.6085 | **0.5792** | **0.6051** | 0.8544 | Best validation macro F1 |

## Test Results for Top Candidates

The strongest validation candidates were checked on the held-out test split.

| Approach | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test Top-3 Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Current configured RF before this experiment | 0.5996 | 0.5613 | 0.5928 | **0.8557** |
| Unweighted Random Forest, 120 trees | **0.6062** | 0.5648 | 0.5984 | 0.8554 |
| Unweighted Random Forest + prior adjustment | 0.6039 | 0.5689 | **0.5998** | 0.8526 |
| Balanced RF + prior adjustment | 0.6014 | **0.5689** | 0.5983 | 0.8547 |
| Balanced Random Forest, depth 20 | 0.5956 | 0.5627 | 0.5919 | 0.8548 |

## Best Approach

The best approach depends on the metric:

- Best test accuracy: **Unweighted Random Forest with 120 trees**
- Best test macro F1: **Balanced Random Forest with prior adjustment**
- Best test weighted F1: **Unweighted Random Forest with prior adjustment**
- Best top-3 accuracy: The previous configured Random Forest was still slightly best

For this project, the best practical choice is:

```text
RandomForestClassifier
n_estimators: 120
max_depth: None
class_weight: None
```

This gives the best held-out test accuracy and keeps the production pipeline simple. The config was updated to use this setting.

## Did Model-Per-Class Help?

No, not for overall accuracy.

The one-vs-rest experiments tested the model-per-class idea directly:

- `ovr_logistic_regression`
- `ovr_linear_svm`
- `ovr_random_forest_small`

The one-vs-rest Random Forest was the best model-per-class approach, but it reached only:

```text
validation accuracy: 0.5891
validation macro F1: 0.5573
```

That is below the regular multiclass Random Forest:

```text
validation accuracy: 0.6104
validation macro F1: 0.5733
```

So the model-per-class approach did not increase accuracy here. It may still be useful later if the goal changes from overall accuracy to targeted recall for one weak class.

## What Worked

The unweighted Random Forest worked best for accuracy. Removing `class_weight="balanced_subsample"` improved validation and test accuracy.

Tree ensembles continued to outperform linear models. This suggests the engineered audio features contain nonlinear thresholds and interactions that trees can exploit better than linear classifiers.

Prior adjustment helped macro F1. It improved smaller-class balance by slightly reducing the dominance of larger classes, but it did not give the highest accuracy.

## What Did Not Work

One-vs-rest linear models did not work well. They were fast, but validation accuracy stayed around 42%.

One-vs-rest Random Forest did not beat the standard multiclass Random Forest. It was slower than the normal forest and had lower validation accuracy.

Soft voting did not improve performance. Adding Logistic Regression to Random Forest softened the predictions but reduced accuracy.

Extra Trees was faster than Random Forest but less accurate.

Histogram Gradient Boosting failed in this environment with:

```text
PermissionError(13, 'Access is denied', None, 5, None)
```

Because of that environment error, it could not be fairly compared.

## Recommendation

Use the unweighted Random Forest with 120 trees as the current best production approach.

Keep prior adjustment as an optional research path if the project wants to optimize macro F1 or improve smaller-class recall instead of pure accuracy.

Do not switch to a model-per-class setup right now. It adds complexity and did not improve validation accuracy in these experiments.

## Commands and Limitations

Experiments were run with direct Python execution using:

```powershell
$env:PYTHONPATH='src'
```

`poetry` and `pytest` are still unavailable in the current environment, so this experiment used direct module imports and the existing train/validation/test split files.
