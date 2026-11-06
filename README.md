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

## hls4ml conversion

`hls_convert.py` turns a saved model directory into a Vitis HLS project, following the
"HLS" cells of the notebooks (Vitis backend, `io_stream`, Latency strategy, 2 ns clock,
part `xcku035-fbva676-2-e`, project name `gaussian`). It converts the 7-parameter network
only; the reconstruction layer is a training-time wrapper.

```bash
python hls_convert.py Models/hybrid_flow_model            # project in gaussian_hls4ml_hybrid_flow_model/
python hls_convert.py Models/hybrid_flow_model --build    # also run Vitis HLS synthesis
```

Without the Xilinx tools it still loads the model, writes the project, compiles the C++
emulation and prints the QKeras-vs-HLS difference per parameter in physical units
(`params.scaling` from the model's `config.yaml`) on a few synthetic crops
(`compare.json`, `compare.npz`); testbench vectors go to `tb_data/`. `--inputs x.npy`
uses your own crops, `--no-compile` only writes the project, `--reuse-factor` and
`--fifo-opt` expose the notebook's optional settings (the FIFO-depth optimization, which
`Real_Flow.ipynb` enabled, runs a Vitis cosim during conversion, so it needs the tools).
The Xilinx install root defaults to the notebooks' path; pass `--xilinx <root>` or export
`XILINX_HLS`, `XILINX_VITIS` and `XILINX_VIVADO` yourself. The exact hls4ml config used is
saved as `hls_config.yaml` in the project. Note that hls4ml's default input precision is
`ap_fixed<16,6>`: the core expects crops scaled to [0, 1) like the training data, and its
outputs are the normalized parameters (multiply by `params.scaling`).
