# Recorded results

These CSV files contain historical measurements extracted from saved training text outputs. Training and evaluation have not been rerun with the standalone Python script. Values retain the precision printed by those runs.

- `training_history.csv`: baseline (10 epochs), tuned end-to-end model (10), VAE (15), and frozen-VAE prediction head (10).
- `test_metrics.csv`: final in-memory end-to-end and frozen-VAE models, evaluated on a recorded test set of 7,352 samples.

The baseline sums movement MSE and gripper cross-entropy; the tuned model and frozen-VAE head use their half-weighted sum. Model inputs, preprocessing, objectives, and loss scales differ between model families. In particular, the VAE reconstruction objective is not numerically comparable to the action-prediction objective. The frozen encoder receives differently scaled inputs during downstream training; the comparison does not isolate representation quality.

The training plots use only these recorded histories. Separate plots based on manually entered arrays were excluded because their measurement provenance could not be established. Checkpoints and underlying data are not included, so the metrics cannot be independently verified from the CSV files alone.
