# Rheed-Gaussian

Training and evaluation of the Gaussian-fitting CNN of the FastRHEED pipeline.

## Training on synthetic data

`train_synthetic.py` is the script version of `Generated_Flow.ipynb` (minus the
hls4ml conversion). All constants live in a YAML config:

```bash
python train_synthetic.py configs/synthetic.yaml
```

The run writes `Models/<output.model_name>/` containing the saved model, a copy
of the config, `history.json`, `test_metrics.json` and a few
image / label reconstruction / prediction plots under `plots/`.

To evaluate a saved model without retraining, set `training.train: false` and
`training.load_from: Models/<name>` in the config.

## Training on real data

`train_real.py` is the script version of `Real_Flow.ipynb` (minus the hls4ml
conversion), configured by `configs/real.yaml`:

```bash
python train_real.py configs/real.yaml
```

`data.path` points at the h5 file in `FastRHEED/data/` and is relative to the
directory the script is run from. Outputs go to `Models/<output.model_name>/`
as above.

## Training on a mix of real and synthetic data

`train_hybrid.py` combines the two: `hybrid.synthetic_fraction` of the training
and validation crops are synthetic Gaussians, the rest real crops. The real
crops are sampled and filtered first and the synthetic count is derived from
the surviving real count, so the share is exact. The model is tested separately
on a real and a synthetic test set.

```bash
python train_hybrid.py configs/hybrid.yaml
```

`configs/hybrid.yaml` takes its real-data, model, loss and training sections
from `configs/real.yaml` and the `gaussian` section from `configs/synthetic.yaml`.
Outputs go to `Models/<output.model_name>/` as above, with
`test_loss_recon_real` / `test_loss_recon_synthetic` in `test_metrics.json` and
plots under `plots/real/` and `plots/synthetic/`.
