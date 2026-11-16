"""python run.py --data DIR [--b 0.0] [--parallel] [--rounds 100] [--C 0.1] [--E 1] -> results/<name>.json"""
import argparse, json, os, ray
import fl

p = argparse.ArgumentParser()
p.add_argument("--data", default="data"); p.add_argument("--synthetic", action="store_true")
p.add_argument("--b", type=float, default=0.0); p.add_argument("--C", type=float, default=0.1)
p.add_argument("--E", type=int, default=10); p.add_argument("--rounds", type=int, default=500)
p.add_argument("--lr", type=float, default=0.03); p.add_argument("--bs", type=int, default=32)
p.add_argument("--seed", type=int, default=0); p.add_argument("--parallel", action="store_true")
p.add_argument("--gpus", type=float, default=0.0, help="GPU fraction per actor (<=1)")
p.add_argument("--name", default=None)
a = p.parse_args()
if a.parallel: ray.init(ignore_reinit_error=True, include_dashboard=False)
raw, test = fl.load_data(a.data, a.seed, a.synthetic)
cl = fl.split_clients(raw, a.b, a.seed)
model, h = fl.fedavg(cl, a.rounds, a.C, a.E, a.lr, a.bs, a.seed, a.parallel, 4 if a.parallel else None, a.gpus)
h["test_acc"] = fl.test_accuracy(model, test); h["args"] = vars(a)
name = a.name or f"C{a.C}_E{a.E}_b{a.b}_{'ray' if a.parallel else 'seq'}_s{a.seed}"
os.makedirs("results", exist_ok=True); json.dump(h, open(f"results/{name}.json", "w"))
print(name, "train_acc %.4f val_acc %.4f test_acc %.4f" % (h["train_acc"][-1], h["val_acc"][-1], h["test_acc"]))
