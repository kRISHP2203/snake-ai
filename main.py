"""Command-line entry point.

Examples
  python main.py train --agent q   --episodes 3000
  python main.py train --agent dqn --episodes 1500
  python main.py evaluate --games 100
  python main.py sweep --agent q --param gamma --values 0.5 0.9 0.99
  python main.py plot
  python main.py watch --agent q            # needs pygame (add --ascii for text mode)
"""
import argparse
import os
import time

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from analysis import plots
from game.environment import SnakeEnv
from training.evaluate import evaluate_agents
from training.sweep import sweep
from training.train import train

MODELS = "results/models"
LOGS = "results"


def hyperparams(args):
    hp = {}
    if args.gamma is not None: hp["gamma"] = args.gamma
    if args.eps_decay is not None: hp["eps_decay"] = args.eps_decay
    if args.lr is not None:
        hp["alpha" if args.agent == "q" else "lr"] = args.lr
    if args.batch_size is not None: hp["batch_size"] = args.batch_size
    return hp


def load_trained(kind):
    path = f"{MODELS}/{kind}.npz"
    if not os.path.exists(path):
        return None
    return (QLearningAgent() if kind == "q" else DQNAgent()).load(path)


def cmd_train(args):
    agent, rows = train(args.agent, episodes=args.episodes, seed=args.seed, size=args.size,
                        log_path=f"{LOGS}/training_log_{args.agent}.csv",
                        save_path=f"{MODELS}/{args.agent}",
                        checkpoint_every=args.checkpoint_every, **hyperparams(args))
    print(f"Done. Model saved to {MODELS}/{args.agent}.npz, log to {LOGS}/training_log_{args.agent}.csv")


def cmd_evaluate(args):
    agents = {"random": RandomAgent(), "rule_based": RuleBasedAgent()}
    for kind, label in (("q", "q_learning"), ("dqn", "dqn")):
        trained = load_trained(kind)
        if trained: agents[label] = trained
    df, summary = evaluate_agents(agents, n_games=args.games, size=args.size,
                                  out_csv=f"{LOGS}/evaluation_games.csv")
    print(summary.to_string())
    summary.to_csv(f"{LOGS}/evaluation_summary.csv")


def cmd_sweep(args):
    values = [float(v) if "." in v or "e" in v.lower() else int(v) for v in args.values]
    sweep(args.agent, args.param, values, seeds=range(args.seeds), episodes=args.episodes)


def cmd_plot(args):
    logs = {k: f"{LOGS}/training_log_{k}.csv" for k in ("q", "dqn") if os.path.exists(f"{LOGS}/training_log_{k}.csv")}
    if logs:
        plots.plot_training(logs, f"{LOGS}/graphs/training_score.png")
        plots.plot_training(logs, f"{LOGS}/graphs/training_reward.png", column="reward")
    if os.path.exists(f"{LOGS}/evaluation_games.csv"):
        plots.plot_comparison(f"{LOGS}/evaluation_games.csv", f"{LOGS}/graphs/agent_comparison.png")
    if os.path.exists(f"{LOGS}/experiment_results.csv"):
        import pandas as pd
        for p in pd.read_csv(f"{LOGS}/experiment_results.csv")["param"].unique():
            plots.plot_sweep(f"{LOGS}/experiment_results.csv", p, f"{LOGS}/graphs/sweep_{p}.png")
    print(f"Graphs written to {LOGS}/graphs/")


def cmd_watch(args):
    agent = {"random": RandomAgent, "rule": RuleBasedAgent}.get(args.agent)
    agent = agent() if agent else load_trained(args.agent)
    if agent is None:
        raise SystemExit(f"No trained model for '{args.agent}'. Run: python main.py train --agent {args.agent}")
    env = SnakeEnv(size=args.size)
    if args.ascii:
        for g in range(args.games):
            state, done = env.reset(seed=50_000 + g), False
            while not done:
                state, _, done, _ = env.step(agent.act(state))
                print("\033[2J\033[H" + env.render_ascii()); time.sleep(0.08)
    else:
        from game.render import watch
        watch(env, agent, games=args.games)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train"); t.set_defaults(fn=cmd_train)
    t.add_argument("--agent", choices=["q", "dqn"], default="q")
    t.add_argument("--episodes", type=int, default=2000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--size", type=int, default=10)
    t.add_argument("--checkpoint-every", type=int, default=None)
    t.add_argument("--gamma", type=float); t.add_argument("--lr", type=float)
    t.add_argument("--eps-decay", type=float); t.add_argument("--batch-size", type=int)

    e = sub.add_parser("evaluate"); e.set_defaults(fn=cmd_evaluate)
    e.add_argument("--games", type=int, default=100); e.add_argument("--size", type=int, default=10)

    s = sub.add_parser("sweep"); s.set_defaults(fn=cmd_sweep)
    s.add_argument("--agent", choices=["q", "dqn"], default="q")
    s.add_argument("--param", required=True, help="gamma | alpha | eps_decay (q); lr | gamma | eps_decay | batch_size (dqn)")
    s.add_argument("--values", nargs="+", required=True)
    s.add_argument("--seeds", type=int, default=3); s.add_argument("--episodes", type=int, default=2000)

    sub.add_parser("plot").set_defaults(fn=cmd_plot)

    w = sub.add_parser("watch"); w.set_defaults(fn=cmd_watch)
    w.add_argument("--agent", choices=["q", "dqn", "random", "rule"], default="q")
    w.add_argument("--games", type=int, default=3); w.add_argument("--size", type=int, default=10)
    w.add_argument("--ascii", action="store_true")

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
