# Genre Mapping and Feature Engineering Changes

## Summary

This note documents the changes made to improve the music genre classifier by changing the class grouping, adding engineered features, and validating the pipeline on the raw datasets.

The main result was an improvement in Spotify validation performance for the best model:

| Metric | Before | After |
| --- | ---: | ---: |
| Validation accuracy | 0.5227 | 0.6063 |
| Validation macro F1 | 0.3915 | 0.5714 |
| Validation weighted F1 | 0.4990 | 0.5994 |
| Validation top-3 accuracy | 0.7467 | 0.8550 |

The final test result after retraining was:

| Metric | Test result |
| --- | ---: |
| Accuracy | 0.5996 |
| Macro F1 | 0.5613 |
| Weighted F1 | 0.5928 |
| Top-3 accuracy | 0.8557 |
| High-confidence accuracy | 0.9190 |
| Costly misclassification rate | 0.0576 |

## Files Changed

- `configs/project.yaml`
- `src/music_genre/features.py`
- `src/music_genre/train.py`

## Genre Mapping Changes

The old mapping grouped some genres into broad families, but many Spotify labels were not included. Those unmapped labels became their own target classes after normalization, which made the classification target noisy and fragmented.

Examples of labels that previously escaped the main taxonomy included:

- `synth-pop`
- `punk-rock`
- `rock-n-roll`
- `metalcore`
- `progressive-house`
- `trip-hop`
- `reggaeton`
- `samba`
- `tango`
- `kids`
- `sleep`
- `mandopop`
- `j-idol`

The updated mapping assigns these labels into broader families. The cleaned dataset now has 9 target classes:

| Genre family | Rows |
| --- | ---: |
| `latin_world` | 20,974 |
| `electronic` | 19,819 |
| `rock` | 19,255 |
| `pop` | 10,413 |
| `mood_context` | 9,179 |
| `acoustic_folk_country` | 6,262 |
| `ambient_chill` | 4,930 |
| `jazz_blues_classical` | 4,914 |
| `hip_hop_rnb` | 4,262 |

This worked because the model no longer had to learn dozens of small raw-label classes. The broader labels are more stable and easier to predict from tabular Spotify audio features.

## Feature Engineering Changes

The original engineered features were:

- `energy_acoustic_ratio`
- `dance_valence_score`
- `loudness_normalized`
- `duration_bucket`
- `speechiness_tier`
- `major_minor_flag`

The updated pipeline adds numeric interaction features:

- `tempo_normalized`
- `energy_danceability_score`
- `acoustic_instrumental_score`
- `speech_energy_score`
- `valence_energy_score`
- `liveness_energy_score`
- `acoustic_energy_balance`
- `tempo_energy_score`
- `popularity_energy_score`
- `duration_minutes`

It also adds categorical bucket features:

- `tempo_bucket`
- `popularity_bucket`
- `energy_bucket`
- `acousticness_bucket`
- `instrumentalness_flag`
- `valence_bucket`

These features worked especially well with Random Forest because tree models can use thresholds and interactions naturally.

## Model Configuration Change

The Random Forest grid was updated from:

```yaml
n_estimators: [80]
max_depth: [12]
```

to:

```yaml
n_estimators: [100]
max_depth: [null]
```

This worked better because the previous `max_depth: 12` constraint limited the model. With the new mapping and richer features, the uncapped Random Forest had enough signal to improve validation performance.

## What Worked

The genre regrouping worked. It reduced the target space into 9 broader classes and removed many accidental raw-label classes.

The feature interactions worked. Accuracy, macro F1, weighted F1, and top-3 accuracy all improved in the validation benchmark.

The Random Forest model worked best among the configured models. After retraining, the best validation run was:

```text
random_forest_1
n_estimators: 100
max_depth: None
validation accuracy: 0.6063
validation macro F1: 0.5714
validation top-3 accuracy: 0.8550
```

The training command now works even when MLflow is not installed. In that case, it trains normally and skips experiment logging.

## What Did Not Work Well

The external benchmark did not improve to the same level as the Spotify test split. Final external accuracy was:

```text
external accuracy: 0.2352
external macro F1: 0.1188
external weighted F1: 0.2493
external top-3 accuracy: 0.5554
```

This is likely due to dataset shift. The model is trained mainly on Spotify-style tabular features, while FMA and GTZAN come from different datasets with different feature distributions, missing metadata, and different genre labeling rules.

The `hip_hop_rnb` class still has weak recall on the Spotify test split:

```text
precision: 0.7391
recall: 0.2439
f1-score: 0.3667
```

This means that when the model predicts `hip_hop_rnb`, it is often correct, but it misses many true `hip_hop_rnb` tracks.

`latin_world` is still a very broad group. It improved the class-count problem, but it mixes regional genres, language-based labels, and world music styles. This helps the model statistically, but it is not a perfect music taxonomy.

## Validation Commands Run

The following commands were run successfully:

```powershell
$env:PYTHONPATH='src'; python -m music_genre.data --config configs/project.yaml
$env:PYTHONPATH='src'; python -m music_genre.train --config configs/project.yaml
$env:PYTHONPATH='src'; python -m music_genre.evaluate --config configs/project.yaml
```

`poetry` was not available in the current shell, so direct module execution was used instead.

`pytest` was not available in the current Python environment:

```text
No module named pytest
```

Because of that, the automated unit test suite could not be run here. The pipeline itself was validated by running data preparation, full model training, and evaluation.

## Generated Outputs

The main generated reports are:

- `reports/validation_report.json`
- `reports/training_summary.json`
- `reports/evaluation_metrics.json`

The trained best model is:

- `models/best_model.joblib`

## Conclusion

The changes improved Spotify validation and test performance substantially. The biggest gains came from reducing noisy target classes and giving the model more useful audio-feature interactions.

The main remaining weakness is generalization to external datasets. Improving that would likely require stronger harmonization of FMA/GTZAN features, source-specific preprocessing, or training with more balanced external examples.
