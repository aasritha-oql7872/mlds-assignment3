"""python validation_tests.py [--data DIR] [--synthetic]  -- behavioural checks of FedAvg / DP noise (not unit tests)."""
import argparse, numpy as np, ray, torch
import fl

p = argparse.ArgumentParser(); p.add_argument("--data", default="data"); p.add_argument("--synthetic", action="store_true")
a = p.parse_args(); raw, test = fl.load_data(a.data, 0, a.synthetic)
cl = fl.split_clients(raw, 0.0, 0); ok = []
def check(name, cond, info=""): ok.append(bool(cond)); print(("PASS" if cond else "FAIL"), name, info)
def dist(m1, m2): return max((x - y).abs().max().item() for x, y in zip(m1.state_dict().values(), m2.state_dict().values()))
def acc(m): return fl.test_accuracy(m, test)
def init(seed): torch.manual_seed(seed); return fl.make_model()
def seeds(seed, n_clients, k):  # mimic fedavg's RNG use for round 1
    r = np.random.RandomState(seed); sel = r.choice(n_clients, k, replace=False); return sel, r.randint(0, 2**31 - 1, k)

# 1. FedAvg with a single client == plain local SGD on that client
m, _ = fl.fedavg(cl[:1], rounds=1, n_per_round=1, epochs=2, seed=3)
_, sd = seeds(3, 1, 1); st, _, _ = fl.local_train(init(3).state_dict(), cl[0], 2, 0.03, 32, int(sd[0]))
ref = fl.make_model(); ref.load_state_dict(st)
check("single-client FedAvg == local SGD", dist(m, ref) < 1e-6)

# 2. zero local epochs -> global model never changes
m, _ = fl.fedavg(cl, rounds=3, epochs=0, seed=0)
check("E=0 leaves initial model unchanged", dist(m, init(0)) < 1e-7)

# 3. aggregation is sample-size weighted: recompute the one-round average by hand for the two largest clients
big = sorted(range(len(cl)), key=lambda i: len(cl[i]["train"][1]))[-1], sorted(range(len(cl)), key=lambda i: len(cl[i]["train"][1]))[0]
two = [cl[big[0]], cl[big[1]]]; sel, sd = seeds(5, 2, 2)
m, _ = fl.fedavg(two, rounds=1, n_per_round=2, epochs=1, seed=5)
res = [fl.local_train(init(5).state_dict(), two[i], 1, 0.03, 32, int(s)) for i, s in zip(sel, sd)]
w = np.array([r[1] for r in res], dtype=float); w /= w.sum()
exp = fl.make_model(); exp.load_state_dict({k: sum(wi * r[0][k] for wi, r in zip(w, res)) for k in res[0][0]})
check("global == n_k-weighted mean of client models (largest + smallest client)", dist(m, exp) < 1e-6, f"weights {w.round(3)}")

# 4. Ray (parallel) result == sequential result for same seed (exact on CPU; small float differences on GPU)
ray.init(ignore_reinit_error=True, include_dashboard=False)
ms, _ = fl.fedavg(cl, rounds=2, n_per_round=4, seed=1); mp, _ = fl.fedavg(cl, rounds=2, n_per_round=4, seed=1, parallel=True)
check("Ray actors == sequential", dist(ms, mp) < 1e-4, f"max diff {dist(ms, mp):.2e}")

# 5. Laplace noise: mean |noise| ~ b, same split as clean run, b=0 adds nothing
b = 0.5; nz = fl.split_clients(raw, b, 0)
diff = torch.cat([(n["train"][0] - c["train"][0]).flatten() for n, c in zip(nz, cl)])
check("mean |noise| ~= b", abs(diff.abs().mean().item() - b) < 0.02, f"{diff.abs().mean():.3f}")
check("labels identical with/without noise", all(torch.equal(n["train"][1], c["train"][1]) for n, c in zip(nz, cl)))

# 6. train/val partition each client; no sample is in both (test set is a separate file)
check("train+val sizes == client sizes", all(len(c["train"][1]) + len(c["val"][1]) == len(r["y"]) for c, r in zip(cl, raw)))
check("test set disjoint from client data (no identical images)",
      not set(map(bytes, (t.numpy().tobytes() for t in test[0][:200]))) & set(c["x"][i].tobytes() for c in raw for i in range(len(c["y"]))))

# 7. accuracy behaves as expected: trained >> random init (~1/62); huge noise -> degraded
mt, _ = fl.fedavg(cl, rounds=30, seed=0); mn, _ = fl.fedavg(fl.split_clients(raw, 50.0, 0), rounds=30, seed=0)
check("trained model beats random init", acc(mt) > acc(init(0)) + 0.05, f"{acc(mt):.3f} vs {acc(init(0)):.3f}")
check("b=50 noise degrades test accuracy", acc(mn) < acc(mt) - 0.05, f"{acc(mn):.3f}")

# 8. more local epochs on one client lower its training loss
l = []
for E in (1, 10):
    st, *_ = fl.local_train(init(0).state_dict(), cl[0], E, 0.03, 32, 0); mm = fl.make_model(); mm.load_state_dict(st); l.append(fl.evaluate(mm, *cl[0]["train"])[0])
check("more local epochs -> lower local train loss", l[1] < l[0], f"{l[0]:.2f} -> {l[1]:.2f}")

# 9. lr=0 (no learning) leaves the model unchanged even with many epochs
m, _ = fl.fedavg(cl, rounds=3, epochs=5, lr=0.0, seed=0)
check("lr=0 leaves initial model unchanged", dist(m, init(0)) < 1e-7)

# 10. trace of the first 10 rounds: selected clients are distinct, global model moves every round, loss goes down overall
prev, steps, losses = init(0), [], []
for t in range(1, 11):
    mt_, h = fl.fedavg(cl, rounds=t, seed=0); steps.append(dist(mt_, prev)); prev = mt_; losses.append(h["train_loss"][-1])
    print(f"  round {t:2d}: clients {sorted(h['selected'][-1])} max|dW| vs previous round {steps[-1]:.4f} train loss {losses[-1]:.3f}")
check("first 10 rounds: 10 distinct clients/round, model changes every round, loss decreases", all(len(set(sel)) == 10 for sel in h["selected"]) and min(steps) > 0 and losses[-1] < losses[0])

# 11. heterogeneity: re-deal the pooled data (a) IID and (b) sorted by label (~1-2 classes per client) over the same client sizes.
#     Expected: IID >= real by-writer split >> label-sorted split (FedAvg degrades with label skew).
allx = np.concatenate([c["x"] for c in raw]); ally = np.concatenate([c["y"] for c in raw])
def redeal(order):
    out, o = [], 0
    for c in raw: n = len(c["y"]); out.append({"x": allx[order[o:o + n]], "y": ally[order[o:o + n]]}); o += n
    return out
def mean_acc(d): return np.mean([acc(fl.fedavg(fl.split_clients(d, 0.0, 0), rounds=100, seed=s)[0]) for s in (0, 1)])
a_real, a_iid, a_sort = mean_acc(raw), mean_acc(redeal(np.random.RandomState(0).permutation(len(ally)))), mean_acc(redeal(np.argsort(ally, kind="stable")))
check("IID >= real split >> label-sorted split (100 rounds, 2 seeds)", a_iid >= a_real > a_sort + 0.1, f"IID {a_iid:.3f}, real {a_real:.3f}, sorted {a_sort:.3f}")

# 12. noise sweep monotonicity (30 rounds): test accuracy decreases with b
accs = [acc(fl.fedavg(fl.split_clients(raw, bb, 0), rounds=30, seed=0)[0]) for bb in (0.0, 0.5, 2.0)]
check("test accuracy decreases with noise scale b (0, 0.5, 2)", accs[0] > accs[1] > accs[2], str(np.round(accs, 3)))
print(f"{sum(ok)}/{len(ok)} passed")
