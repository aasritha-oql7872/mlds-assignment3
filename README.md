# MLDS Assignment 3 – FedAvg + Laplace differential privacy on Federated EMNIST

| file | purpose |
|---|---|
| `fl.py` | data loading/splitting (+Laplace noise), 784-128-62 MLP, local training, Ray `Client` actor, `fedavg` |
| `run.py` | one training run -> `results/<name>.json` |
| `experiments.sh` | all sweeps (C, E, Ray run, noise scales b) |
| `plots.py` | all figures in `figures/` (incl. data distribution and a client-participation heatmap) |
| `experiments_seeds.sh` | seeds 1 and 2 of the noise sweep (error bars) |
| `validation_tests.py` | 15 behavioural validation checks (see below) |

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
1. single client == plain local SGD; 2. `E=0` leaves the model unchanged; 3. global model == hand-computed n_k-weighted mean;
4. Ray == sequential; 5. mean |noise| ~ b, labels unchanged; 6. train/val partition, test shares no images with clients;
7. trained beats random init, b=50 noise hurts; 8. more local epochs lower local loss; 9. `lr=0` leaves the model unchanged;
10. trace of the first 10 rounds (selected clients, weight change, loss); 11. IID >= real split >> label-sorted split;
12. clean/mild noise clearly beats heavy noise (b=2). Full-run seeds: `bash experiments_seeds.sh data` (after `experiments.sh`), then `python plots.py --data data`.
