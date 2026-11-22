"""Record a game as an animated GIF (board + the agent's Q-values). No Pygame needed.

Handy for reports, slides, and for checking an agent without opening a window.
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from game.environment import SnakeEnv

ACTION_NAMES = ("Straight", "Turn right", "Turn left")
SURFACE, GRID_BG, BODY, HEAD, FOOD = "#fcfcfb", "#ecebe7", "#1baf7a", "#0e7852", "#e34948"
INK, INK2, BAR, BAR_PICK = "#0b0b0b", "#52514e", "#c4c3bd", "#2a78d6"


def record_game(agent, size=10, seed=1_000_000, max_steps=2000):
    """Play one game with fixed seed. Returns a list of per-step snapshots."""
    env = SnakeEnv(size=size, state_mode=getattr(agent, "state_mode", "basic"))
    state, done = env.reset(seed=seed), False
    frames, getter = [], getattr(agent, "q_values", None)
    frames.append({"snake": list(env.snake), "food": env.food, "score": 0, "step": 0,
                   "q": None if getter is None else list(getter(state)), "action": None, "cause": None})
    while not done and env.steps < max_steps:
        q = None if getter is None else list(getter(state))
        action = agent.act(state, explore=False)
        state, _, done, info = env.step(action)
        frames.append({"snake": list(env.snake), "food": env.food, "score": env.score, "step": env.steps,
                       "q": q, "action": action, "cause": info["cause"]})
    return frames


def _draw(frame, size, name, fig, ax_board, ax_q):
    ax_board.clear(); ax_q.clear()
    ax_board.set_xlim(0, size); ax_board.set_ylim(size, 0); ax_board.set_aspect("equal")
    ax_board.axis("off")
    ax_board.add_patch(plt.Rectangle((0, 0), size, size, color=GRID_BG, zorder=0))
    for i, (x, y) in enumerate(frame["snake"]):
        ax_board.add_patch(FancyBboxPatch((x + 0.08, y + 0.08), 0.84, 0.84, boxstyle="round,pad=0,rounding_size=0.18",
                                          color=HEAD if i == 0 else BODY, linewidth=0))
    if frame["food"]:
        fx, fy = frame["food"]
        ax_board.add_patch(FancyBboxPatch((fx + 0.22, fy + 0.22), 0.56, 0.56, boxstyle="round,pad=0,rounding_size=0.28",
                                          color=FOOD, linewidth=0))
    ax_board.set_title(f"{name}    score {frame['score']}    step {frame['step']}", loc="left",
                       fontsize=10, color=INK)
    ax_q.set_facecolor(SURFACE)
    for side in ("top", "right", "bottom", "left"):
        ax_q.spines[side].set_visible(False)
    ax_q.set_xticks([]); ax_q.set_yticks([])
    if frame["q"] is None:
        ax_q.text(0.0, 0.7, "Hand-written agent:\nno Q-values to show.", fontsize=9, color=INK2, va="top")
        if frame["action"] is not None:
            ax_q.text(0.0, 0.3, f"Pick: {ACTION_NAMES[frame['action']]}", fontsize=10, color=INK)
    else:
        q = frame["q"]
        low, high = min(q), max(q)
        span = (high - low) or 1.0
        ax_q.set_xlim(0, 1.45); ax_q.set_ylim(-0.6, 3.0)
        ax_q.text(0.0, 2.75, "Value of each move (Q)", fontsize=9, color=INK2, va="center")
        for a, value in enumerate(q):
            y = 2.0 - a
            ax_q.barh(y, 0.1 + 0.9 * (value - low) / span, height=0.4,
                      color=BAR_PICK if a == frame["action"] else BAR, left=0)
            ax_q.text(0.0, y + 0.42, f"{ACTION_NAMES[a]}   {value:.1f}", fontsize=9,
                      color=INK if a == frame["action"] else INK2, va="center")
    if frame["cause"] and frame["cause"] != "won":
        ax_q.text(0.0, -0.45, f"Game over ({frame['cause']})", fontsize=10, color=FOOD)


def save_gif(agent, out_path, size=10, seed=1_000_000, fps=10, max_frames=300, label=None):
    frames = record_game(agent, size=size, seed=seed)
    if len(frames) > max_frames:                       # keep the GIF small: thin out, keep the ending
        stride = -(-len(frames) // max_frames)
        frames = frames[::stride] + [frames[-1]] * 1
    fig, (ax_board, ax_q) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1, 0.8]})
    fig.patch.set_facecolor(SURFACE)
    images = []
    for frame in frames:
        _draw(frame, size, label or getattr(agent, "name", "agent"), fig, ax_board, ax_q)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=70, facecolor=SURFACE)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    plt.close(fig)
    images += [images[-1]] * (fps)                     # hold the last frame for a second
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return frames[-1]["score"], len(frames)
