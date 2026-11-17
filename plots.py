"""python plots.py [--data DIR] -> figures/*.png from results/*.json (+ data distribution plot)"""
import argparse, glob, json, os
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import fl

p = argparse.ArgumentParser(); p.add_argument("--data", default="data"); p.add_argument("--synthetic", action="store_true")
a = p.parse_args(); os.makedirs("figures", exist_ok=True)
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob("results/*.json")}

def curves(name, title):
    h = R[name]; fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for k in ("train", "val"):
        ax[0].plot(h[k + "_loss"], label=k); ax[1].plot(h[k + "_acc"], label=k)
    for x, t in zip(ax, ("loss", "accuracy")): x.set_xlabel("communication round"); x.set_ylabel(t); x.legend()
    fig.suptitle(f"{title} (test acc {h['test_acc']:.3f})"); fig.tight_layout(); fig.savefig(f"figures/{name}.png", dpi=150); plt.close(fig)

for n in R:
    if R[n]["args"]["seed"] == 0: curves(n, n)

# data distribution: overall + 5 users
cl, _ = fl.load_data(a.data, 0, a.synthetic); fig, ax = plt.subplots(2, 3, figsize=(12, 6))
allc = np.bincount(np.concatenate([c["y"] for c in cl]), minlength=62)
ax[0, 0].bar(range(62), allc); ax[0, 0].set_title("all 100 users")
for x, i in zip(ax.flat[1:], np.random.RandomState(0).choice(len(cl), 5, replace=False)):
    x.bar(range(62), np.bincount(cl[i]["y"], minlength=62)); x.set_title(f"user {i} (n={len(cl[i]['y'])})")
for x in ax.flat: x.set_xlabel("class"); x.set_ylabel("count")
fig.tight_layout(); fig.savefig("figures/data_distribution.png", dpi=150); plt.close(fig)

# C and E comparison (final accuracies)
ce = sorted(n for n in R if n.startswith("C") and "_b0.0_seq" in n)
if ce:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for n in ce:
        for x, k in zip(ax, ("train", "val")): x.plot(R[n][k + "_acc"], label=n.split("_b")[0] + f" test={R[n]['test_acc']:.3f}")
    for x, k in zip(ax, ("train", "validation")): x.set_xlabel("communication round"); x.set_ylabel(k + " accuracy")
    ax[1].legend(fontsize=6); fig.tight_layout(); fig.savefig("figures/part1_C_E.png", dpi=150); plt.close(fig)

# noise sweep: mean +- std over seeds
P2 = {}
for n in R:
    if n.startswith("part2_b"): P2.setdefault(R[n]["args"]["b"], []).append(R[n])
nb = sorted(P2.items())
if nb:
    ms = lambda b, f: (np.mean([f(h) for h in P2[b]]), np.std([f(h) for h in P2[b]]))
    plt.figure(figsize=(5.5, 3.8)); bs = [b for b, _ in nb]
    for lab, f, m in (("final train", lambda h: h["train_acc"][-1], "o-"), ("test", lambda h: h["test_acc"], "s-")):
        v = np.array([ms(b, f) for b in bs]); plt.errorbar(bs, v[:, 0], v[:, 1], fmt=m, capsize=3, label=lab)
    plt.xlabel("noise scale b"); plt.ylabel(f"accuracy (mean $\\pm$ std, {len(nb[0][1])} seeds)"); plt.legend(); plt.tight_layout(); plt.savefig("figures/part2_noise.png", dpi=150); plt.close()

# non-standard plot: client participation heatmap (rows=clients, cols=rounds, colour=client's local val acc after local training)
n = "C0.1_E10_b0.0_seq_s0"
if n in R:
    h = R[n]; M = np.full((100, len(h["selected"])), np.nan)
    for t, (s, v) in enumerate(zip(h["selected"], h["local_val_acc"])): M[s, t] = v
    plt.figure(figsize=(9, 4)); plt.imshow(M, aspect="auto", cmap="viridis", interpolation="nearest"); plt.colorbar(label="local val acc")
    plt.xlabel("communication round"); plt.ylabel("client"); plt.title("Client participation & local quality"); plt.tight_layout()
    plt.savefig("figures/participation_heatmap.png", dpi=150); plt.close()

# Part 2 non-standard plot: one test image under increasing noise (top) and train-val accuracy gap vs b (bottom)
if nb:
    _, test = fl.load_data(a.data, 0, a.synthetic); img = test[0][0].numpy(); bs = [x[0] for x in nb]
    fig = plt.figure(figsize=(11, 4)); rn = np.random.RandomState(0)
    for k, bb in enumerate(bs):
        ax = fig.add_subplot(2, len(bs), k + 1); ax.imshow(np.clip(img + rn.laplace(0, bb, 784), 0, 1).reshape(28, 28), cmap="gray"); ax.set_title(f"b={bb}", fontsize=8); ax.axis("off")
    ax = fig.add_subplot(2, 1, 2); ax.plot(bs, [np.mean([h["train_acc"][-1] - h["val_acc"][-1] for h in x[1]]) for x in nb], "o-")
    ax.set_xlabel("noise scale b"); ax.set_ylabel("train acc - val acc"); fig.tight_layout(); fig.savefig("figures/part2_noise_examples_gap.png", dpi=150); plt.close(fig)
