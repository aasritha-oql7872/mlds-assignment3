# MLDS Assignment 3 – FedAvg + Laplace differential privacy on Federated EMNIST

| file | purpose |
|---|---|
| `fl.py` | data loading/splitting (+Laplace noise), 784-128-62 MLP, local training, Ray `Client` actor, `fedavg` |
| `run.py` | one training run -> `results/<name>.json` |
| `experiments.sh` | all sweeps (C, E, Ray run, noise scales b) |
| `plots.py` | all figures in `figures/` (incl. data distribution and a client-participation heatmap) |
| `validation_tests.py` | behavioural validation checks (see below) |

## Run on deepdish
    git clone https://github.com/aasritha-oql7872/mlds-assignment3.git && cd mlds-assignment3
    python3 -m venv ~/fl_env && source ~/fl_env/bin/activate && pip install -r requirements.txt
    nvidia-smi; export CUDA_VISIBLE_DEVICES=<idle gpu index>     # use one GPU only
    # put the Canvas data folder (train_data.npy, test_data.npy) at ./data
    bash experiments.sh data              # all experiments
    python plots.py --data data           # figures
    python validation_tests.py --data data
Single runs: `python run.py --data data --C 0.1 --E 10`, Ray: `--parallel [--gpus 0.25]` (4 clients/round, 1 CPU each, `--gpus` = GPU fraction per actor),
noise: `--b 0.2`. `--synthetic` replaces the data with random class prototypes for a quick smoke test.
Each client's local data is split 80% train / 20% validation; the provided `test_data.npy` is the held-out test set (never noised, never trained on). Noise is added to train/val images only (pixels are in [0,1]).
Part 1 (sequential) and Part 2 use the same `fedavg`; actors are created only for the selected clients each round and killed afterwards.

## Validation tests (`validation_tests.py`)
1. single client, full participation == plain local SGD; 2. `E=0` leaves the model unchanged; 3. global model == hand-computed n_k-weighted mean of client models;
4. Ray result == sequential result for the same seed; 5. mean |noise| ≈ b and test data stays clean; 6. train/val partition each client and the test set shares no images with client data;
7. trained model beats random init, and huge noise (b=50) hurts test accuracy; 8. more local epochs lower local training loss.
