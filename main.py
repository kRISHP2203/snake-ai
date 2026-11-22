"""Command-line entry point.

Train and look at one agent
  python main.py train --agent q   --state rich --episodes 3000
  python main.py train --agent ddqn --state rich --episodes 1500
  python main.py evaluate --games 100
  python main.py watch  --agent ddqn_rich          (Pygame window; --ascii for text mode)
  python main.py replay --agent ddqn_rich          (saves an animated GIF, no Pygame needed)

Experiments (results go to results/*.csv, graphs to results/graphs/)
  python main.py multiseed --seeds 5               train every agent over several seeds and compare
  python main.py sweep --agent q --param gamma --values 0.5 0.9 0.99
  python main.py generalise                        evaluate saved models on other grid sizes
  python main.py experiments                       the whole suite above, with the default settings
  python main.py plot                              redraw all graphs from the CSV files

Agent names: random, rule_v1, rule_v2, q_basic, q_rich, ddqn_basic, ddqn_rich
"""
import argparse
import os
import time

from agents.double_dqn_agent import DoubleDQNAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent, RuleBasedV2Agent
from analysis import plots
from game.environment import SnakeEnv
from training import config
from training.evaluate import evaluate_agents
from training.experiments import CONFIGS, run_generalisation, run_multiseed, summarise_multiseed
from training.sweep import sweep
from training.train import train

MODELS, RESULTS = "results/models", "results"
ALIASES = {"rule": "rule_v1", "q": "q_rich", "ddqn": "ddqn_rich"}


def parse_value(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def get_agent(name):
    name = ALIASES.get(name, name)
    if name == "random":
        return RandomAgent()
    if name == "rule_v1":
        return RuleBasedAgent()
    if name == "rule_v2":
        return RuleBasedV2Agent()
    path = f"{MODELS}/{name}.npz"
    if not os.path.exists(path):
        kind, _, state = name.partition("_")
        if kind not in ("q", "ddqn") or state not in ("", "basic", "rich"):
            raise SystemExit(f"Unknown agent '{name}'. Use one of: random, rule_v1, rule_v2, "
                             "q_basic, q_rich, ddqn_basic, ddqn_rich")
        raise SystemExit(f"No trained model '{path}'. Train one first:\n"
                         f"  python main.py train --agent {kind} --state {state or 'rich'}")
    return (QLearningAgent() if name.startswith("q_") else DoubleDQNAgent()).load(path)


def hyperparams(args):
    hp = {}
    if args.gamma is not None: hp["gamma"] = args.gamma
    if args.eps_decay is not None: hp["eps_decay"] = args.eps_decay
    if args.lr is not None: hp["alpha" if args.agent == "q" else "lr"] = args.lr
    if args.batch_size is not None: hp["batch_size"] = args.batch_size
    return hp


def cmd_train(args):
    env_kwargs = {"size": args.size}
    if args.reward_step is not None:
        env_kwargs["reward_step"] = args.reward_step
    name = f"{args.agent}_{args.state}"
    try:
        train(args.agent, episodes=args.episodes, seed=args.seed, state_mode=args.state, env_kwargs=env_kwargs,
              log_path=f"{RESULTS}/training_log_{name}.csv", save_path=f"{MODELS}/{name}",
              checkpoint_every=args.checkpoint_every, **hyperparams(args))
    except TypeError as err:                       # e.g. --batch-size with --agent q
        raise SystemExit(f"Setting not supported by '{args.agent}': {err}")
    print(f"Done. Model: {MODELS}/{name}.npz   Log: {RESULTS}/training_log_{name}.csv")


def cmd_evaluate(args):
    agents = {"random": RandomAgent(), "rule_v1": RuleBasedAgent(), "rule_v2": RuleBasedV2Agent()}
    for kind, state in CONFIGS:
        name = f"{kind}_{state}"
        if os.path.exists(f"{MODELS}/{name}.npz"):
            agents[name] = get_agent(name)
    _, summary = evaluate_agents(agents, n_games=args.games, size=args.size,
                                 out_csv=f"{RESULTS}/evaluation_games.csv")
    print(summary.to_string())
    summary.to_csv(f"{RESULTS}/evaluation_summary.csv")


def cmd_multiseed(args):
    df = run_multiseed(seeds=args.seeds, eval_games=args.games)
    summary = summarise_multiseed(df)
    print("\n" + summary.to_string())
    summary.to_csv(f"{RESULTS}/multiseed_summary.csv")


def cmd_sweep(args):
    values = [parse_value(v) for v in args.values]
    episodes = args.episodes or config.SWEEP_EPISODES[args.agent]
    sweep(args.agent, args.param, values, seeds=range(args.seeds), episodes=episodes, state_mode=args.state)


def cmd_generalise(args):
    run_generalisation(eval_games=args.games)


def cmd_experiments(args):
    quick = args.quick
    out = f"{RESULTS}/quick" if quick else RESULTS          # quick numbers are meaningless: keep them away from the real ones
    os.makedirs(out, exist_ok=True)
    if args.only in ("all", "multiseed"):
        df = run_multiseed(seeds=2 if quick else config.SEEDS, eval_games=30 if quick else config.EVAL_GAMES,
                           episodes={"q": 300, "ddqn": 100} if quick else None,
                           out_dir=out, model_dir=f"{out}/models")
        summary = summarise_multiseed(df)
        print("\n" + summary.to_string())
        summary.to_csv(f"{out}/multiseed_summary.csv")
    if args.only in ("all", "sweeps"):
        path = f"{out}/experiment_results.csv"
        if os.path.exists(path):
            os.remove(path)                                  # start clean so reruns do not duplicate rows
        for kind, param, values in config.SWEEPS:
            episodes = config.SWEEP_EPISODES[kind]
            sweep(kind, param, values, seeds=range(2 if quick else config.SWEEP_SEEDS),
                  episodes=60 if quick and kind == "ddqn" else (200 if quick else episodes),
                  eval_games=20 if quick else config.EVAL_GAMES, state_mode="rich", out_csv=path)
    if args.only in ("all", "generalise"):
        run_generalisation(model_dir=f"{out}/models", eval_games=20 if quick else config.EVAL_GAMES,
                           out_csv=f"{out}/generalisation_results.csv")


def cmd_plot(args):
    g = f"{RESULTS}/graphs"
    curves = plots.load_curves(RESULTS)
    if not curves.empty:
        plots.plot_training_curves(curves, f"{g}/training_score.png", column="score")
        plots.plot_training_curves(curves, f"{g}/training_reward.png", column="reward")
    for csv_name, fn, out in (("multiseed_results.csv", plots.plot_multiseed, "multiseed_comparison.png"),
                              ("evaluation_games.csv", plots.plot_comparison, "agent_comparison_games.png"),
                              ("experiment_results.csv", plots.plot_sweeps_overview, "sweeps_overview.png"),
                              ("generalisation_results.csv", plots.plot_generalisation, "generalisation.png")):
        if os.path.exists(f"{RESULTS}/{csv_name}"):
            fn(f"{RESULTS}/{csv_name}", f"{g}/{out}")
    print(f"Graphs written to {g}/")


def cmd_watch(args):
    agent = get_agent(args.agent)
    env = SnakeEnv(size=args.size, state_mode=agent.state_mode)
    if args.ascii:
        for game in range(args.games):
            state, done = env.reset(seed=1_000_000 + game), False
            while not done:
                state, _, done, _ = env.step(agent.act(state))
                print("\033[2J\033[H" + env.render_ascii())
                time.sleep(0.08)
    else:
        from game.render import watch
        watch(env, agent, games=args.games, fps=args.fps, label=ALIASES.get(args.agent, args.agent))


def cmd_replay(args):
    from game.replay import save_gif
    agent = get_agent(args.agent)
    out = args.out or f"{RESULTS}/replay_{ALIASES.get(args.agent, args.agent)}.gif"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    score, n = save_gif(agent, out, size=args.size, seed=args.seed, fps=args.fps,
                        label=ALIASES.get(args.agent, args.agent))
    print(f"Saved {out}  (final score {score}, {n} frames)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train"); t.set_defaults(fn=cmd_train)
    t.add_argument("--agent", choices=["q", "ddqn"], default="q")
    t.add_argument("--state", choices=["basic", "rich"], default="rich")
    t.add_argument("--episodes", type=int, default=2000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--size", type=int, default=config.SIZE)
    t.add_argument("--checkpoint-every", type=int, default=None)
    t.add_argument("--gamma", type=float); t.add_argument("--lr", type=float)
    t.add_argument("--eps-decay", type=float); t.add_argument("--batch-size", type=int)
    t.add_argument("--reward-step", type=float, help="reward added on every step that does not eat (e.g. -0.01)")

    e = sub.add_parser("evaluate"); e.set_defaults(fn=cmd_evaluate)
    e.add_argument("--games", type=int, default=config.EVAL_GAMES); e.add_argument("--size", type=int, default=config.SIZE)

    m = sub.add_parser("multiseed"); m.set_defaults(fn=cmd_multiseed)
    m.add_argument("--seeds", type=int, default=config.SEEDS); m.add_argument("--games", type=int, default=config.EVAL_GAMES)

    s = sub.add_parser("sweep"); s.set_defaults(fn=cmd_sweep)
    s.add_argument("--agent", choices=["q", "ddqn"], default="q")
    s.add_argument("--param", required=True,
                   help="q: gamma alpha eps_decay | ddqn: lr gamma eps_decay batch_size | both: reward_step")
    s.add_argument("--values", nargs="+", required=True)
    s.add_argument("--state", choices=["basic", "rich"], default="rich")
    s.add_argument("--seeds", type=int, default=config.SWEEP_SEEDS)
    s.add_argument("--episodes", type=int, default=None)

    g = sub.add_parser("generalise"); g.set_defaults(fn=cmd_generalise)
    g.add_argument("--games", type=int, default=config.EVAL_GAMES)

    x = sub.add_parser("experiments"); x.set_defaults(fn=cmd_experiments)
    x.add_argument("--only", choices=["all", "multiseed", "sweeps", "generalise"], default="all")
    x.add_argument("--quick", action="store_true", help="tiny runs to check everything works (numbers are meaningless)")

    sub.add_parser("plot").set_defaults(fn=cmd_plot)

    w = sub.add_parser("watch"); w.set_defaults(fn=cmd_watch)
    w.add_argument("--agent", default="ddqn_rich"); w.add_argument("--games", type=int, default=3)
    w.add_argument("--size", type=int, default=config.SIZE); w.add_argument("--fps", type=int, default=12)
    w.add_argument("--ascii", action="store_true")

    r = sub.add_parser("replay"); r.set_defaults(fn=cmd_replay)
    r.add_argument("--agent", default="ddqn_rich"); r.add_argument("--out", default=None)
    r.add_argument("--size", type=int, default=config.SIZE); r.add_argument("--seed", type=int, default=1_000_000)
    r.add_argument("--fps", type=int, default=10)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
