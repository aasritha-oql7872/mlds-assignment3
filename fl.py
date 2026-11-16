"""FedAvg on Federated EMNIST (62 classes) with Ray actors and optional Laplace input noise."""
import glob, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import ray


def load_data(path, seed=0, synthetic=False):
    """Returns (clients, test): clients = list of {x:(n,784), y:(n,)}, test = (x, y) held-out set.
    Expects train_data.npy / test_data.npy in `path` (train_data[i]['images'|'labels'], test_data[0][...])."""
    if synthetic:
        r = np.random.RandomState(seed); P = np.random.RandomState(123).rand(62, 784)
        def mk(n): y = r.randint(0, 62, n); return {"images": (P[y] + 0.3 * r.rand(n, 784)).astype("float32"), "labels": y}
        tr, te = [mk(60) for _ in range(100)], [mk(500)]
    else:
        tr = np.load(os.path.join(path, "train_data.npy"), allow_pickle=True)
        te = np.load(os.path.join(path, "test_data.npy"), allow_pickle=True)
    flat = lambda d: (np.asarray(d["images"], dtype="float32").reshape(len(d["labels"]), -1), np.asarray(d["labels"]).astype("int64"))
    clients = [dict(zip("xy", flat(d))) for d in tr]
    return clients, tuple(torch.from_numpy(a) for a in flat(te[0]))


def split_clients(clients, b=0.0, seed=0):
    """Per client 80/20 train/val split. Laplace(b) noise is added to the train and val images (test stays clean)."""
    r = np.random.RandomState(seed); rn = np.random.RandomState(seed + 1); out = []
    for c in clients:
        p = r.permutation(len(c["y"])); nv = max(1, int(.2 * len(p)))
        d = {}
        for k, i in (("val", p[:nv]), ("train", p[nv:])):
            x = c["x"][i].copy()
            if b > 0: x += rn.laplace(0, b, x.shape).astype("float32")
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


def test_accuracy(model, test):
    return evaluate(model, *test)[1] / len(test[1])
