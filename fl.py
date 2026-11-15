"""FedAvg on Federated EMNIST (62 classes) with Ray actors and optional Laplace input noise."""
import glob, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import ray


def load_clients(path, seed=0, synthetic=False):
    """Returns list of dicts {x:(n,784) float32, y:(n,) int64}. `path`: dir of LEAF-style json files
    ({"users": [...], "user_data": {u: {"x": [...], "y": [...]}}}); if it holds train/ and test/ subdirs both are merged."""
    users = {}
    if synthetic:
        r = np.random.RandomState(seed); P = np.random.RandomState(123).rand(62, 784)
        for u in range(100):
            y = r.randint(0, 62, 60); users[u] = ((P[y] + 0.3 * r.rand(60, 784)).astype("float32"), y)
    else:
        for f in sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True)):
            for u, d in json.load(open(f))["user_data"].items():
                x, y = np.array(d["x"], dtype="float32").reshape(len(d["y"]), -1), np.array(d["y"])
                users[u] = (np.concatenate([users[u][0], x]), np.concatenate([users[u][1], y])) if u in users else (x, y)
    return [{"x": x, "y": y.astype("int64")} for x, y in users.values()]


def split_clients(clients, b=0.0, seed=0):
    """Per client: 20% test (clean), remaining 80/20 train/val. Laplace(b) noise is added to train and val images only."""
    r = np.random.RandomState(seed); rn = np.random.RandomState(seed + 1); out = []
    for c in clients:
        p = r.permutation(len(c["y"])); n = len(p); nt = max(1, int(.2 * n)); nv = max(1, int(.2 * (n - nt)))
        parts = {"test": p[:nt], "val": p[nt:nt + nv], "train": p[nt + nv:]}
        d = {}
        for k, i in parts.items():
            x = c["x"][i].copy()
            if b > 0 and k != "test": x += rn.laplace(0, b, x.shape).astype("float32")
            d[k] = (torch.from_numpy(x), torch.from_numpy(c["y"][i]))
        out.append(d)
    return out


def make_model():
    return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 62))


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad(): o = model(x)
    return F.cross_entropy(o, y, reduction="sum").item(), (o.argmax(1) == y).sum().item()


def local_train(state, data, epochs, lr, bs, seed, dev="cpu"):
    torch.manual_seed(seed)
    m = make_model(); m.load_state_dict(state); m.to(dev).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr); x, y = data["train"][0].to(dev), data["train"][1].to(dev)
    for _ in range(epochs):
        for i in torch.randperm(len(y), device=dev).split(bs):
            opt.zero_grad(); F.cross_entropy(m(x[i]), y[i]).backward(); opt.step()
    vl, vc = evaluate(m, *(t.to(dev) for t in data["val"]))
    return {k: v.cpu() for k, v in m.state_dict().items()}, len(y), vc / len(data["val"][1])


@ray.remote(num_cpus=1)
class Client:
    """One actor per selected client per round; holds only that client's data."""
    def __init__(self, data): self.data = data
    def update(self, state, epochs, lr, bs, seed):
        dev = "cuda" if torch.cuda.is_available() and ray.get_gpu_ids() else "cpu"
        return local_train(state, self.data, epochs, lr, bs, seed, dev)


def fedavg(clients, rounds=100, C=0.1, epochs=1, lr=0.05, bs=32, seed=0, parallel=False, n_per_round=None, gpus=0.0):
    """clients: output of split_clients. Returns (global model, history dict)."""
    r = np.random.RandomState(seed); torch.manual_seed(seed)
    model = make_model(); k = n_per_round or max(1, int(round(C * len(clients))))
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "selected": [], "local_val_acc": []}
    ntr = sum(len(c["train"][1]) for c in clients); nva = sum(len(c["val"][1]) for c in clients)
    for t in range(rounds):
        sel = r.choice(len(clients), k, replace=False); state = {a: v.clone() for a, v in model.state_dict().items()}
        seeds = r.randint(0, 2**31 - 1, k)
        if parallel:
            actors = [Client.options(num_gpus=gpus).remote(clients[i]) for i in sel]
            res = ray.get([a.update.remote(state, epochs, lr, bs, int(s)) for a, s in zip(actors, seeds)])
            for a in actors: ray.kill(a)
        else:
            res = [local_train(state, clients[i], epochs, lr, bs, int(s)) for i, s in zip(sel, seeds)]
        n = np.array([x[1] for x in res], dtype="float64"); w = n / n.sum()
        model.load_state_dict({a: sum(wi * x[0][a] for wi, x in zip(w, res)) for a in state})
        # global-model metrics over all clients' train / val data
        for name, tot in (("train", ntr), ("val", nva)):
            l = c = 0
            for cl in clients: dl, dc = evaluate(model, *cl[name]); l += dl; c += dc
            hist[name + "_loss"].append(l / tot); hist[name + "_acc"].append(c / tot)
        hist["selected"].append(sel.tolist()); hist["local_val_acc"].append([x[2] for x in res])
    return model, hist


def test_accuracy(model, clients):
    c = sum(evaluate(model, *cl["test"])[1] for cl in clients)
    return c / sum(len(cl["test"][1]) for cl in clients)
