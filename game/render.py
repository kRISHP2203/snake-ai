"""Optional Pygame viewer (only imported when you use the 'watch' command)."""
import time


def watch(env, agent, games=5, fps=12, cell=30, seed=50_000):
    import pygame  # imported lazily so training never needs it

    pygame.init()
    size = env.size * cell
    screen = pygame.display.set_mode((size, size + 30))
    pygame.display.set_caption(f"Snake AI - {agent.name}")
    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    for g in range(games):
        state = env.reset(seed=seed + g)
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            state, _, done, info = env.step(agent.act(state, explore=False))
            screen.fill((20, 20, 20))
            for i, (x, y) in enumerate(env.snake):
                colour = (80, 220, 100) if i else (40, 160, 70)
                pygame.draw.rect(screen, colour, (x * cell + 1, y * cell + 1, cell - 2, cell - 2))
            if env.food:
                fx, fy = env.food
                pygame.draw.rect(screen, (230, 60, 60), (fx * cell + 4, fy * cell + 4, cell - 8, cell - 8))
            label = font.render(f"{agent.name}  game {g + 1}/{games}  score {env.score}", True, (230, 230, 230))
            screen.blit(label, (6, size + 6))
            pygame.display.flip()
            clock.tick(fps)
        time.sleep(0.6)
    pygame.quit()
