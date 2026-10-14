"""experiments_multiseed.py
Experiment 15 - Multi-Seed Statistical Evaluation
==================================================

Sweep: 3 scenarios x 2 defenses x N seeds.
Scenarios: clean, byzantine (sign-flip scale 10), labelflip (ratio 0.5).
Defenses : fedavg, optigradtrust_v4 (trust_enabled, combined scoring).

Rules
-----
* set_global_seed(seed) seeds random/numpy/torch/cuda/cudnn BEFORE create_model().
* Malicious client IDs and label-flip decisions use a LOCAL np.random.RandomState
  so both defense arms get identical attack targets under the same seed.
* No config globals for seed/alpha/num_malicious/poison_ratio/defense/attack.
* _cfg.TRUST_SCORING_METHOD is set to combined before each v4 run so trust.py
  reads the right value at update_scores() time.
* Existing Experiments 1-14 are untouched.

CLI
---
    python basecode.py --experiment 15 --quick
    python basecode.py --experiment 15
    python basecode.py --experiment 15 --seeds 42 43 44 45 46 --rounds 50
    python basecode.py --experiment 15 --fresh
    python experiments_multiseed.py --analyze
"""

from __future__ import annotations

import argparse, copy, json, math, random, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as _cudnn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import create_model
from dataset import CIFAR10Dataset
from federated_learning import FLClient, FLServer, FederatedLearner
from attacks import AttackManager, ModelPoisoningAttack, DataPoisoningAttack
from checkpoint import CheckpointManager
import config as _cfg
import trust as _trust

_NUM_CLIENTS     = 10
_NUM_BYZANTINE   = 3
_BYZANTINE_SCALE = 10.0
_POISON_RATIO    = 0.5
_DIRICHLET_ALPHA = 1.0
_BATCH_SIZE      = 64
_NUM_CLASSES     = 10

EXP15_SCENARIOS = ("clean", "byzantine", "labelflip")
EXP15_DEFENSES  = ("fedavg", "optigradtrust_v4")
EXP15_BASE_DIR  = Path("./results/exp15_multiseed")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    _cudnn.deterministic = True
    _cudnn.benchmark     = False


class SeededAttackManager(AttackManager):
    def __init__(self, num_clients, num_byzantine, byzantine_scale,
                 poison_ratio, rng):
        self.num_clients     = num_clients
        self.num_byzantine   = num_byzantine
        self.byzantine_scale = byzantine_scale
        self.poison_ratio    = poison_ratio
        self._rng            = rng
        chosen = rng.choice(num_clients, size=num_byzantine, replace=False)
        self.byzantine_clients = set(int(c) for c in chosen)
        self.model_poisoning = ModelPoisoningAttack(scale=byzantine_scale)
        self.data_poisoning  = _SeededDataPoisoning(poison_ratio=poison_ratio, rng=rng)

    def is_byzantine_client(self, cid): return cid in self.byzantine_clients
    def is_data_poisoning_client(self, cid): return cid in self.byzantine_clients

    def apply_model_poisoning(self, cid, update):
        if self.is_byzantine_client(cid):
            return self.model_poisoning.poison_update(None, update)
        return update

    def apply_data_poisoning(self, images, labels):
        return self.data_poisoning.poison_batch(images, labels)

    def get_byzantine_clients(self):
        return list(self.byzantine_clients)


class _SeededDataPoisoning(DataPoisoningAttack):
    def __init__(self, poison_ratio, rng):
        self.poison_ratio     = poison_ratio
        self.label_flip_type  = "random"
        self.poisoned_indices = set()
        self._rng             = rng

    def poison_batch(self, images, labels, num_classes=_NUM_CLASSES):
        poisoned = labels.clone()
        n_p      = max(1, int(len(labels) * self.poison_ratio))
        idxs     = self._rng.choice(len(labels), size=n_p, replace=False)
        for i in idxs:
            cur = int(poisoned[i].item())
            nl  = int(self._rng.randint(0, num_classes))
            while nl == cur:
                nl = int(self._rng.randint(0, num_classes))
            poisoned[i] = nl
        return images, poisoned


def _to_json_safe(obj):
    if isinstance(obj, dict):   return {str(k): _to_json_safe(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):return float(obj)
    if isinstance(obj, torch.Tensor):  return obj.cpu().tolist()
    return obj


class MultiSeedRunner:
    def __init__(self, scenario, defense, seed, rounds, fresh=False):
        assert scenario in EXP15_SCENARIOS
        assert defense  in EXP15_DEFENSES
        self.scenario = scenario
        self.defense  = defense
        self.seed     = seed
        self.rounds   = rounds
        self.fresh    = fresh
        self.run_name    = f"Exp15_{scenario}_{defense}_seed{seed}"
        self.run_dir     = EXP15_BASE_DIR / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.result_path = self.run_dir / "result.json"
        self.checkpoint_mgr = CheckpointManager(self.run_name)

    def run(self):
        if self.result_path.exists() and not self.fresh:
            print(f"  [Exp15] Skipping {self.run_name} -- result.json exists.")
            with open(self.result_path, encoding="utf-8") as f:
                return json.load(f)
        if self.fresh:
            if self.result_path.exists():
                try: self.result_path.unlink()
                except Exception: pass
            if (self.checkpoint_mgr.ckpt_dir / "progress.json").exists():
                try: (self.checkpoint_mgr.ckpt_dir / "progress.json").unlink()
                except Exception: pass
            if (self.checkpoint_mgr.ckpt_dir / "DONE").exists():
                try: (self.checkpoint_mgr.ckpt_dir / "DONE").unlink()
                except Exception: pass

        print(f"\n{'='*70}\n  Exp15: {self.run_name}")
        print(f"  scenario={self.scenario}  defense={self.defense}  "
              f"seed={self.seed}  rounds={self.rounds}\n{'='*70}")
        try:
            result = self._execute()
            with open(self.result_path, "w", encoding="utf-8") as f:
                json.dump(_to_json_safe(result), f, indent=2)
            print(f"  [Exp15] Saved {self.result_path}")
            return result
        except Exception as exc:
            import traceback
            print(f"  [Exp15] ERROR in {self.run_name}: {exc}")
            traceback.print_exc()
            return None

    def _configure_globals(self):
        pqc_on   = True
        trust_on = (self.defense == "optigradtrust_v4")
        _cfg.PQC_ENABLED          = pqc_on
        _cfg.ENCRYPT_MESSAGE      = pqc_on
        _cfg.SIGN_MESSAGE         = pqc_on
        _cfg.TRUST_SCORE_ENABLED  = trust_on
        _cfg.TRUST_SCORING_METHOD = "combined"
        _trust.TRUST_SCORING_METHOD = "combined"

    def _setup(self):
        set_global_seed(self.seed)
        self._configure_globals()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        global_model = create_model(device=device)
        print("  Loading CIFAR-10 dataset...")
        dataset = CIFAR10Dataset(data_dir="./data",
                                 alpha=_DIRICHLET_ALPHA,
                                 num_clients=_NUM_CLIENTS,
                                 seed=self.seed)
        local_rng = np.random.RandomState(self.seed)
        attack_manager = None
        malicious_ids  = []
        if self.scenario in ("byzantine", "labelflip"):
            attack_manager = SeededAttackManager(
                num_clients=_NUM_CLIENTS, num_byzantine=_NUM_BYZANTINE,
                byzantine_scale=_BYZANTINE_SCALE, poison_ratio=_POISON_RATIO,
                rng=local_rng)
            malicious_ids = attack_manager.get_byzantine_clients()
            print(f"  Malicious clients ({self.scenario}): {sorted(malicious_ids)}")
        print(f"  Creating {_NUM_CLIENTS} clients...")
        clients = []
        for cid in range(_NUM_CLIENTS):
            cm = create_model(device=device)
            cm.load_state_dict(copy.deepcopy(global_model.state_dict()))
            cl = dataset.get_client_dataloader(cid, batch_size=_BATCH_SIZE,
                                               train=True, shuffle=True)
            clients.append(FLClient(client_id=cid, model=cm, dataset=cl,
                                    attack_manager=attack_manager, device=device))
        trust_on = (self.defense == "optigradtrust_v4")
        server = FLServer(model=global_model, defense_name="fedavg",
                          device=device, trust_enabled=trust_on,
                          trust_scoring_method="combined" if trust_on else None)
        if trust_on and server.trust_manager:
            server.trust_manager.scoring_method = "combined"
        for c in clients:
            server.register_client(c.client_id, c.get_public_key())
        learner     = FederatedLearner(clients, server)
        test_loader = dataset.get_global_testloader(batch_size=_BATCH_SIZE)
        return learner, test_loader, dataset, attack_manager, malicious_ids, device

    def _execute(self):
        learner, test_loader, _, _, malicious_ids, _ = self._setup()
        apply_byz = (self.scenario == "byzantine")
        apply_lf  = (self.scenario == "labelflip")
        acc_hist, loss_hist = [], []
        start_round = 0

        # Attempt crash-recovery from checkpoint if available and not fresh
        if not self.fresh and getattr(_cfg, 'CHECKPOINT_ENABLED', True):
            state = self.checkpoint_mgr.load_latest()
            if state:
                last = state['last_completed_round']
                learner.server.model.load_state_dict(state['model_state_dict'])
                global_state = learner.server.model.state_dict()
                for c in learner.clients:
                    c.model.load_state_dict(global_state)
                    c.round_num = last + 1
                saved_metrics = state.get('metrics', {})
                acc_hist = [float(x) for x in saved_metrics.get('test_accuracy', [])]
                loss_hist = [float(x) for x in saved_metrics.get('test_loss', [])]
                start_round = last + 1
                print(f"  [Exp15] Resuming {self.run_name} from round {start_round+1}/{self.rounds}")

        t0 = time.time()
        for rnd in range(start_round, self.rounds):
            stats = learner.perform_round(round_num=rnd, test_loader=test_loader,
                                          apply_data_poisoning=apply_lf,
                                          apply_model_poisoning=apply_byz)
            acc  = stats.get("test_accuracy", 0.0)
            loss = stats.get("test_loss",     0.0)
            acc_hist.append(float(acc))
            loss_hist.append(float(loss))
            cl = stats.get("client_losses", [0.0])
            avg = float(np.mean(cl)) if cl else 0.0
            print(f"  Round {rnd+1:>4}/{self.rounds} | "
                  f"TrainLoss:{avg:.4f} | TestLoss:{loss:.4f} | Acc:{acc:.2f}%")
            if self.defense == "optigradtrust_v4" and learner.server.trust_manager:
                learner.server.trust_manager.print_scores(rnd)

            # Checkpoint saving
            ckpt_interval = getattr(_cfg, 'CHECKPOINT_INTERVAL', 5)
            if getattr(_cfg, 'CHECKPOINT_ENABLED', True) and ((rnd + 1) % ckpt_interval == 0 or rnd == self.rounds - 1):
                self.checkpoint_mgr.save(
                    rnd,
                    learner.server.model.state_dict(),
                    {'test_accuracy': acc_hist, 'test_loss': loss_hist},
                    experiment_config={'scenario': self.scenario, 'defense': self.defense, 'seed': self.seed}
                )

        if getattr(_cfg, 'CHECKPOINT_ENABLED', True) and self.rounds > start_round:
            self.checkpoint_mgr.mark_complete()

        total_time = time.time() - t0
        final  = acc_hist[-1] if acc_hist else 0.0
        best   = max(acc_hist)  if acc_hist else 0.0
        last5  = float(np.mean(acc_hist[-5:])) if len(acc_hist) >= 5 else final
        h_trust = m_trust = None
        if self.defense == "optigradtrust_v4" and learner.server.trust_manager:
            sc   = learner.server.trust_manager.get_scores()
            mset = set(malicious_ids)
            h_sc = [v for k,v in sc.items() if k not in mset]
            m_sc = [v for k,v in sc.items() if k in mset]
            h_trust = float(np.mean(h_sc)) if h_sc else None
            m_trust = float(np.mean(m_sc)) if m_sc else None
        return {
            "config": {
                "scenario": self.scenario, "defense": self.defense,
                "seed": self.seed, "rounds": self.rounds,
                "num_clients": _NUM_CLIENTS, "num_byzantine": _NUM_BYZANTINE,
                "byzantine_scale": _BYZANTINE_SCALE,
                "poison_ratio": _POISON_RATIO,
                "dirichlet_alpha": _DIRICHLET_ALPHA, "pqc_enabled": True,
            },
            "seed": self.seed,
            "malicious_client_ids": sorted(malicious_ids),
            "final_accuracy": final, "last5_mean_accuracy": last5,
            "best_accuracy": best,
            "per_round_accuracy": acc_hist, "per_round_loss": loss_hist,
            "mean_trust_honest": h_trust, "mean_trust_malicious": m_trust,
            "total_time_sec": total_time,
        }


def run_experiment_15(seeds=(42,43,44,45,46), rounds=None, fresh=False):
    if rounds is None:
        rounds = _cfg.NUM_ROUNDS
    EXP15_BASE_DIR.mkdir(parents=True, exist_ok=True)
    total = len(EXP15_SCENARIOS)*len(EXP15_DEFENSES)*len(seeds)
    print(f"\n{'='*70}\n  Experiment 15: Multi-Seed Statistical Evaluation")
    print(f"  Seeds:{seeds}  Rounds:{rounds}  Total:{total}\n{'='*70}")
    results = {}
    for scenario in EXP15_SCENARIOS:
        for defense in EXP15_DEFENSES:
            for seed in seeds:
                r = MultiSeedRunner(scenario=scenario, defense=defense,
                                    seed=seed, rounds=rounds, fresh=fresh)
                results[r.run_name] = r.run()
    done = sum(1 for v in results.values() if v is not None)
    print(f"\n[Exp15] Sweep done. Finished:{done}/{len(results)}")
    analyze_experiment_15()
    return results


def _load_all_results(base_dir=None):
    if base_dir is None:
        base_dir = EXP15_BASE_DIR
    out = []
    for p in Path(base_dir).glob("*/result.json"):
        try:
            with open(p, encoding="utf-8") as f: out.append(json.load(f))
        except Exception as e: print(f"  [Analyze] skip {p}: {e}")
    return out


def _ci95(values):
    from scipy.stats import t as t_dist
    n = len(values)
    if n < 2: return float("nan")
    se = np.std(values, ddof=1) / math.sqrt(n)
    return float(float(t_dist.ppf(0.975, df=n-1)) * se)


def analyze_experiment_15(out_dir=None):
    from scipy.stats import ttest_rel, wilcoxon
    if out_dir is None: out_dir = EXP15_BASE_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_all_results(base_dir=out_dir)
    if not raw:
        print("[Analyze] No result.json files found.")
        return {}
    print(f"\n[Analyze] Found {len(raw)} completed runs.")
    idx = {}
    for r in raw:
        cfg  = r.get("config", {})
        sc   = cfg.get("scenario","?")
        df   = cfg.get("defense","?")
        seed = r.get("seed",-1)
        idx[(sc,df,seed)] = r
    all_seeds = sorted(set(k[2] for k in idx))
    summary_rows = []
    for scenario in EXP15_SCENARIOS:
        for defense in EXP15_DEFENSES:
            fa, l5 = [], []
            for s in all_seeds:
                e = idx.get((scenario,defense,s))
                if e:
                    fa.append(e["final_accuracy"])
                    l5.append(e["last5_mean_accuracy"])
            n = len(fa)
            if n==0: continue
            fa, l5 = np.array(fa), np.array(l5)
            summary_rows.append({
                "scenario": scenario, "defense": defense, "n": n,
                "final_mean": float(np.mean(fa)),
                "final_std":  float(np.std(fa,ddof=1)) if n>1 else 0.0,
                "final_min":  float(np.min(fa)),
                "final_max":  float(np.max(fa)),
                "last5_mean": float(np.mean(l5)),
                "last5_std":  float(np.std(l5,ddof=1)) if n>1 else 0.0,
                "last5_min":  float(np.min(l5)),
                "last5_max":  float(np.max(l5)),
            })
    paired_rows = []
    for scenario in EXP15_SCENARIOS:
        ps = [s for s in all_seeds
              if (scenario,"fedavg",s) in idx
              and (scenario,"optigradtrust_v4",s) in idx]
        if not ps: continue
        v4  = [idx[(scenario,"optigradtrust_v4",s)]["final_accuracy"] for s in ps]
        avg = [idx[(scenario,"fedavg",s)]["final_accuracy"] for s in ps]
        diffs = np.array(v4) - np.array(avg)
        n     = len(diffs)
        md    = float(np.mean(diffs))
        sd    = float(np.std(diffs,ddof=1)) if n>1 else 0.0
        ci    = _ci95(diffs)
        dz    = (md/sd) if sd>1e-12 else float("nan")
        wins  = int(np.sum(diffs>0))
        p_t   = float("nan")
        if n>=2: _,p_t = ttest_rel(v4,avg)
        p_w = float("nan")
        if n>=3 and np.any(diffs!=0):
            try: _,p_w = wilcoxon(diffs)
            except: pass
        paired_rows.append({
            "scenario": scenario, "n": n, "seeds": ps,
            "mean_diff": md, "std_diff": sd,
            "ci95_half": ci if not math.isnan(ci) else None,
            "p_value_t": float(p_t) if not math.isnan(p_t) else None,
            "p_value_w": float(p_w) if not math.isnan(p_w) else None,
            "cohens_dz": float(dz)  if not math.isnan(dz)  else None,
            "v4_wins": wins, "per_seed_diff": diffs.tolist(),
        })
    _write_summary_csv(out_dir/"summary_table.csv",    summary_rows)
    _write_summary_md (out_dir/"summary_table.md",     summary_rows)
    _write_paired_csv (out_dir/"paired_comparison.csv",paired_rows)
    _write_paired_md  (out_dir/"paired_comparison.md", paired_rows)
    _plot_bar_chart(summary_rows,  out_dir/"bar_chart.png")
    _plot_accuracy_curves(idx,     out_dir/"accuracy_curves.png")
    _plot_paired_diff(paired_rows, out_dir/"paired_diff.png")
    analysis = {"summary": summary_rows, "paired_comparison": paired_rows,
                "num_runs_found": len(raw)}
    with open(out_dir/"results.json","w",encoding="utf-8") as f:
        json.dump(_to_json_safe(analysis), f, indent=2)
    print(f"\n[Analyze] Outputs -> {out_dir}/")
    return analysis


def _write_summary_csv(path, rows):
    hdr = ["scenario","defense","n",
           "final_mean","final_std","final_min","final_max",
           "last5_mean","last5_std","last5_min","last5_max"]
    lines = [",".join(hdr)]
    for r in rows: lines.append(",".join(str(r.get(h,"")) for h in hdr))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary_md(path, rows):
    hdr = ("| Scenario | Defense | n | Final (mean+-std) |"
           " Last-5 (mean+-std) | Min / Max |")
    sep = ("|----------|---------|---|-------------------|"
           "--------------------|-----------|")
    lines = ["## Experiment 15 -- Summary Table","",hdr,sep]
    for r in rows:
        fa = f"{r['final_mean']:.2f}+-{r['final_std']:.2f}"
        l5 = f"{r['last5_mean']:.2f}+-{r['last5_std']:.2f}"
        mm = f"{r['final_min']:.2f} / {r['final_max']:.2f}"
        lines.append(f"| {r['scenario']} | {r['defense']} |"
                     f" {r['n']} | {fa} | {l5} | {mm} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_paired_csv(path, rows):
    hdr = ["scenario","n","mean_diff","std_diff","ci95_half",
           "p_value_t","p_value_w","cohens_dz","v4_wins"]
    lines = [",".join(hdr)]
    for r in rows:
        lines.append(",".join(
            str(r.get(h,"")) if r.get(h) is not None else "nan"
            for h in hdr))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_paired_md(path, rows):
    hdr = ("| Scenario | n | Mean diff | Std | 95% CI |"
           " p(t) | p(W) | Cohens_dz | v4 wins |")
    sep = ("|----------|---|-----------|-----|--------|"
           "------|------|-----------|---------|")
    lines = ["## Experiment 15 -- Paired Comparison (v4 minus FedAvg)","",
             "> Positive diff = v4 better. n = paired seeds.","",hdr,sep]
    def _f(v): return f"{v:.4f}" if v is not None else "nan"
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {r['n']}"
            f" | {_f(r['mean_diff'])} | {_f(r['std_diff'])}"
            f" | +-{_f(r['ci95_half'])} | {_f(r['p_value_t'])}"
            f" | {_f(r['p_value_w'])} | {_f(r['cohens_dz'])}"
            f" | {r['v4_wins']}/{r['n']} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_bar_chart(summary_rows, path):
    if not summary_rows: return
    scenarios = list(dict.fromkeys(r["scenario"] for r in summary_rows))
    defenses  = list(dict.fromkeys(r["defense"]  for r in summary_rows))
    x, w      = np.arange(len(scenarios)), 0.35
    colors    = ["#4C72B0","#DD8452"]
    labels    = {"fedavg":"FedAvg","optigradtrust_v4":"OptiGradTrust v4"}
    fig, ax   = plt.subplots(figsize=(9,5))
    for i, df in enumerate(defenses):
        means, stds = [], []
        for sc in scenarios:
            row = next((r for r in summary_rows
                        if r["scenario"]==sc and r["defense"]==df), None)
            means.append(row["final_mean"] if row else 0.0)
            stds.append( row["final_std"]  if row else 0.0)
        ax.bar(x+i*w, means, w, yerr=stds, capsize=5,
               label=labels.get(df,df), color=colors[i],
               alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x+w/2)
    ax.set_xticklabels([s.capitalize() for s in scenarios], fontsize=12)
    ax.set_ylabel("Final Test Accuracy (%)", fontsize=12)
    ax.set_title("Experiment 15 -- Final Accuracy (mean+-std)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.set_ylim(0,105)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [Plot] {path.name}")


def _plot_accuracy_curves(idx, path):
    if not idx: return
    colors = {"fedavg":"#4C72B0","optigradtrust_v4":"#DD8452"}
    labels = {"fedavg":"FedAvg","optigradtrust_v4":"OptiGradTrust v4"}
    scenarios = list(EXP15_SCENARIOS)
    fig, axes = plt.subplots(1,len(scenarios),figsize=(15,4),sharey=False)
    if len(scenarios)==1: axes=[axes]
    for ax, scenario in zip(axes, scenarios):
        for defense in EXP15_DEFENSES:
            runs = [v for (sc,df,_),v in idx.items()
                    if sc==scenario and df==defense]
            if not runs: continue
            mr  = max(len(r["per_round_accuracy"]) for r in runs)
            arr = np.full((len(runs),mr), np.nan)
            for j,r in enumerate(runs):
                a = r["per_round_accuracy"]; arr[j,:len(a)] = a
            mc = np.nanmean(arr,axis=0)
            sc_curve = np.nanstd(arr,axis=0,ddof=1 if len(runs)>1 else 0)
            xs = np.arange(1,mr+1); c = colors[defense]
            ax.plot(xs,mc,color=c,label=labels.get(defense,defense),linewidth=1.8)
            ax.fill_between(xs,mc-sc_curve,mc+sc_curve,color=c,alpha=0.20)
        ax.set_title(f"Scenario: {scenario.capitalize()}", fontsize=11)
        ax.set_xlabel("Round",fontsize=10); ax.set_ylabel("Accuracy (%)",fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.3,linestyle="--")
    fig.suptitle("Experiment 15 -- Accuracy vs Round (mean+-std)",fontsize=13,y=1.01)
    fig.tight_layout(); fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  [Plot] {path.name}")


def _plot_paired_diff(paired_rows, path):
    if not paired_rows: return
    n_sc = len(paired_rows)
    fig, axes = plt.subplots(1,n_sc,figsize=(5*n_sc,4),sharey=False)
    if n_sc==1: axes=[axes]
    for ax, row in zip(axes, paired_rows):
        seeds = row["seeds"]; diffs = row["per_seed_diff"]
        cols  = ["#2ecc71" if d>0 else "#e74c3c" for d in diffs]
        ax.bar(range(len(seeds)),diffs,color=cols,edgecolor="black",
               linewidth=0.7,alpha=0.85)
        ax.axhline(0,color="k",linewidth=0.8,linestyle="--")
        md = row["mean_diff"]
        ax.axhline(md,color="navy",linewidth=1.5,linestyle="-",
                   label=f"mean={md:.2f}%")
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([str(s) for s in seeds],fontsize=9)
        ax.set_xlabel("Seed",fontsize=10)
        ax.set_ylabel("Delta Acc (v4-FedAvg) %",fontsize=10)
        pt = row["p_value_t"]
        ps = f"p(t)={pt:.3f}" if pt is not None else "p(t)=nan"
        ax.set_title(f"{row['scenario'].capitalize()}  n={row['n']}  {ps}",fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.35,linestyle="--")
    fig.suptitle("Experiment 15 -- Per-Seed Paired Diff (v4-FedAvg)",
                 fontsize=13,y=1.02)
    fig.tight_layout(); fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  [Plot] {path.name}")


def _parse_multiseed_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[42,43,44,45,46])
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--fresh", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_multiseed_args()
    if args.analyze:
        analyze_experiment_15()
    else:
        seeds  = (42,43) if args.quick else tuple(args.seeds)
        rounds = 2 if args.quick else args.rounds
        run_experiment_15(seeds=seeds, rounds=rounds, fresh=args.fresh)
