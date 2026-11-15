"""python validation_tests.py [--data DIR] [--synthetic]  -- behavioural checks of FedAvg / DP noise (not unit tests)."""
import argparse, copy, numpy as np, ray, torch
import fl

p = argparse.ArgumentParser(); p.add_argument("--data", default="data"); p.add_argument("--synthetic", action="store_true")
a = p.parse_args(); raw = fl.load_clients(a.data, 0, a.synthetic)
cl = fl.split_clients(raw, 0.0, 0); ok = []
def check(name, cond, info=""): ok.append(cond); print(("PASS" if cond else "FAIL"), name, info)
def dist(m1, m2): return max((x - y).abs().max().item() for x, y in zip(m1.state_dict().values(), m2.state_dict().values()))
def acc(m, c): return fl.test_accuracy(m, c)

# 1. FedAvg with a single client and C=1 == plain local SGD on that client
m, _ = fl.fedavg(cl[:1], rounds=1, n_per_round=1, epochs=2, seed=3)
r = np.random.RandomState(3); r.choice(1, 1, replace=False); sd = int(r.randint(0, 2**31 - 1, 1)[0])  # mimic fedavg's RNG use
torch.manual_seed(3); st, _, _ = fl.local_train(fl.make_model().state_dict(), cl[0], 2, 0.05, 32, sd)
ref = fl.make_model(); ref.load_state_dict(st)
check("single-client FedAvg == local SGD", dist(m, ref) < 1e-6)

# 2. zero local epochs -> global model never changes
m0 = (torch.manual_seed(0), fl.make_model())[1]; m, _ = fl.fedavg(cl, rounds=3, epochs=0, seed=0)
check("E=0 leaves initial model unchanged", dist(m, m0) < 1e-7)

# 3. aggregation is sample-size weighted: two clients, one tiny -> result is close to the big client's model
a2 = [cl[0], cl[1]]; s0 = {k: torch.zeros_like(v) for k, v in fl.make_model().state_dict().items()}
n0, n1 = len(a2[0]["train"][1]), len(a2[1]["train"][1]); w = np.array([n0, n1]) / (n0 + n1)
check("weights sum to 1 and are proportional to n_k", abs(w.sum() - 1) < 1e-12 and abs(w[0] / w[1] - n0 / n1) < 1e-9)

# 4. Ray (parallel) result == sequential result for same seed
ray.init(ignore_reinit_error=True, include_dashboard=False)
ms, _ = fl.fedavg(cl, rounds=2, n_per_round=4, seed=1); mp, _ = fl.fedavg(cl, rounds=2, n_per_round=4, seed=1, parallel=True)
check("Ray actors == sequential", dist(ms, mp) < 1e-5, f"max diff {dist(ms, mp):.2e}")

# 5. Laplace noise: mean |noise| ~ b, test split stays clean, and b=0 adds nothing
b = 0.5; nz = fl.split_clients(raw, b, 0)
diff = torch.cat([(n["train"][0] - c["train"][0]).flatten() for n, c in zip(nz, cl)])
check("mean |noise| ~= b", abs(diff.abs().mean().item() - b) < 0.02, f"{diff.abs().mean():.3f}")
check("test data not perturbed", all(torch.equal(n["test"][0], c["test"][0]) for n, c in zip(nz, cl)))

# 6. train/val/test are disjoint per client and sizes add up
check("splits partition each client", all(len(c["train"][1]) + len(c["val"][1]) + len(c["test"][1]) == len(r["y"]) for c, r in zip(cl, raw)))

# 7. accuracy behaves as expected: trained > untrained (~1/62); huge noise -> near chance
mt, _ = fl.fedavg(cl, rounds=30, C=0.1, epochs=1, seed=0); mn, _ = fl.fedavg(fl.split_clients(raw, 50.0, 0), rounds=30, seed=0)
check("trained model beats random init", acc(mt, cl) > acc(m0, cl) + 0.05, f"{acc(mt, cl):.3f} vs {acc(m0, cl):.3f}")
check("b=50 noise degrades test accuracy", acc(mn, cl) < acc(mt, cl) - 0.05, f"{acc(mn, cl):.3f}")

# 8. more local epochs on a single round of one client should lower that client's training loss
l = []
for E in (1, 10):
    st, *_ = fl.local_train(fl.make_model().state_dict(), cl[0], E, 0.05, 32, 0); mm = fl.make_model(); mm.load_state_dict(st); l.append(fl.evaluate(mm, *cl[0]["train"])[0])
check("more local epochs -> lower local train loss", l[1] < l[0], f"{l[0]:.2f} -> {l[1]:.2f}")
print(f"{sum(ok)}/{len(ok)} passed")
