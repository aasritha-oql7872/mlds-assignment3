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

for n in R: curves(n, n)

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
    plt.figure(figsize=(8, 3.5))
    for n in ce: plt.plot(R[n]["val_acc"], label=n.split("_b")[0] + f" test={R[n]['test_acc']:.3f}")
    plt.xlabel("communication round"); plt.ylabel("validation accuracy"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig("figures/part1_C_E.png", dpi=150); plt.close()

# noise sweep
nb = sorted((R[n]["args"]["b"], R[n]) for n in R if n.startswith("part2_b"))
if nb:
    plt.figure(figsize=(5, 3.5)); b = [x[0] for x in nb]
    plt.plot(b, [x[1]["train_acc"][-1] for x in nb], "o-", label="final train"); plt.plot(b, [x[1]["test_acc"] for x in nb], "s-", label="test")
    plt.xlabel("noise scale b"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout(); plt.savefig("figures/part2_noise.png", dpi=150); plt.close()

# non-standard plot: client participation heatmap (rows=clients, cols=rounds, colour=client's local val acc after local training)
n = "C0.1_E10_b0.0_seq_s0"
if n in R:
    h = R[n]; M = np.full((100, len(h["selected"])), np.nan)
    for t, (s, v) in enumerate(zip(h["selected"], h["local_val_acc"])): M[s, t] = v
    plt.figure(figsize=(9, 4)); plt.imshow(M, aspect="auto", cmap="viridis", interpolation="nearest"); plt.colorbar(label="local val acc")
    plt.xlabel("communication round"); plt.ylabel("client"); plt.title("Client participation & local quality"); plt.tight_layout()
    plt.savefig("figures/participation_heatmap.png", dpi=150); plt.close()
