"""Pygame viewer: the game board plus a live panel showing what the agent "thinks".

Controls:  SPACE pause/resume   UP/DOWN faster/slower   N next game   ESC or window close: quit
For learning agents the panel shows the agent's value (Q) for each of the three moves and
highlights the one it picks. Hand-written agents have no Q-values, so only the pick is shown.
"""
ACTION_NAMES = ("Straight", "Turn right", "Turn left")

BG, GRID_BG = (24, 24, 26), (34, 34, 38)
SNAKE_BODY, SNAKE_HEAD, FOOD = (27, 175, 122), (14, 120, 82), (227, 73, 72)
TEXT, TEXT_DIM, BAR, BAR_PICK = (235, 235, 232), (150, 150, 146), (90, 92, 100), (42, 120, 214)


def _q_values(agent, state):
    getter = getattr(agent, "q_values", None)
    return None if getter is None else list(getter(state))


def watch(env, agent, games=5, fps=12, cell=30, seed=1_000_000, label=None):
    import pygame  # imported lazily so training never needs it

    label = label or agent.name
    pygame.init()
    board, panel = env.size * cell, 250
    screen = pygame.display.set_mode((board + panel, board))
    pygame.display.set_caption(f"Snake AI - {label}")
    font, small = pygame.font.SysFont(None, 26), pygame.font.SysFont(None, 21)
    clock = pygame.time.Clock()
    paused = False

    def draw(game, action, q_vals, finished_cause=None):
        screen.fill(BG)
        pygame.draw.rect(screen, GRID_BG, (0, 0, board, board))
        for i, (x, y) in enumerate(env.snake):
            colour = SNAKE_HEAD if i == 0 else SNAKE_BODY
            pygame.draw.rect(screen, colour, (x * cell + 1, y * cell + 1, cell - 2, cell - 2), border_radius=5)
        if env.food:
            fx, fy = env.food
            pygame.draw.rect(screen, FOOD, (fx * cell + 5, fy * cell + 5, cell - 10, cell - 10), border_radius=8)

        x0 = board + 18
        lines = [(label, font, TEXT), (f"Game {game + 1}/{games}", small, TEXT_DIM),
                 (f"Score  {env.score}", font, TEXT), (f"Steps  {env.steps}", small, TEXT_DIM),
                 (f"Speed  {fps} fps" + ("  (paused)" if paused else ""), small, TEXT_DIM)]
        y = 18
        for text, f, colour in lines:
            screen.blit(f.render(text, True, colour), (x0, y))
            y += 30
        y += 14
        screen.blit(small.render("What the agent sees", True, TEXT_DIM), (x0, y))
        y += 28
        if q_vals is None:
            screen.blit(small.render("hand-written rules, no Q-values", True, TEXT_DIM), (x0, y))
            if action is not None:
                screen.blit(font.render("Pick: " + ACTION_NAMES[action], True, TEXT), (x0, y + 30))
        else:
            low, high = min(q_vals), max(q_vals)
            span = (high - low) or 1.0
            for a, value in enumerate(q_vals):
                length = 12 + int(150 * (value - low) / span)       # bars are relative, not absolute
                colour = BAR_PICK if a == action else BAR
                pygame.draw.rect(screen, colour, (x0, y + 18, length, 12), border_radius=4)
                screen.blit(small.render(f"{ACTION_NAMES[a]}  {value:6.1f}", True, TEXT if a == action else TEXT_DIM),
                            (x0, y))
                y += 44
        if finished_cause:
            screen.blit(font.render(f"Game over: {finished_cause}", True, FOOD), (x0, board - 64))
        screen.blit(small.render("SPACE pause  UP/DOWN speed", True, TEXT_DIM), (x0, board - 36))
        screen.blit(small.render("N next game  ESC quit", True, TEXT_DIM), (x0, board - 20))
        pygame.display.flip()

    for game in range(games):
        state, done, skip, action, info = env.reset(seed=seed + game), False, False, None, {}
        draw(game, None, _q_values(agent, state))
        while not done and not skip:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_UP:
                        fps = min(60, fps + 2)
                    elif event.key == pygame.K_DOWN:
                        fps = max(2, fps - 2)
                    elif event.key == pygame.K_n:
                        skip = True
            if paused or skip:
                clock.tick(30)
                continue
            q_vals = _q_values(agent, state)
            action = agent.act(state, explore=False)
            state, _, done, info = env.step(action)
            draw(game, action, q_vals)
            clock.tick(fps)
        if done:
            draw(game, action, None if _q_values(agent, state) is None else _q_values(agent, state),
                 finished_cause=info.get("cause"))
            pygame.time.wait(1200)
    pygame.quit()
