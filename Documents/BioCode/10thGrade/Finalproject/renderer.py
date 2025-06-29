# renderer.py
import pygame
from game.config import GRID_SIZE, PREY_STARVATION_TICKS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FOOD_COLOR = (255, 223, 0)
PREY_COLOR = (100, 200, 255)
STARVING_PREY_COLOR = (255, 60, 60)
PREDATOR_COLOR = (255, 80, 80)
RL_PREY_BORDER = (255, 255, 0)
TEXT_COLOR = (0, 0, 0)
SIDE_PANEL_TEXT_COLOR = (255, 255, 255)  # White text for side panel

# Grid settings
CELL_SIZE = 100
MARGIN = 2
INFO_HEIGHT = 60
GRID_WIDTH = GRID_SIZE * CELL_SIZE + (GRID_SIZE + 1) * MARGIN
# Total window width is 2.25 times the grid width so there is extra space for our action log.
WIDTH = 2.25 * GRID_WIDTH  
HEIGHT = GRID_WIDTH  # keep grid square
WINDOW_HEIGHT = HEIGHT + INFO_HEIGHT

pygame.init()
font = pygame.font.SysFont("Arial", 24)
small_font = pygame.font.SysFont("Arial", 16)
screen = pygame.display.set_mode((int(WIDTH), int(WINDOW_HEIGHT)))
pygame.display.set_caption("Animals DND :D")

def ActionDraw(action_log_str):
    """
    Independently draws the provided text in the side panel.
    You can call this function with any string (newlines allowed) to display custom text.
    """
    side_panel_width = WIDTH - GRID_WIDTH
    panel_x = int(GRID_WIDTH)  # Side panel begins right after the grid
    panel_rect = pygame.Rect(panel_x, 0, int(side_panel_width), int(HEIGHT))
    # Fill the side panel background (black)
    pygame.draw.rect(screen, BLACK, panel_rect)

    # If no text is provided, show a default message.
    if not action_log_str or action_log_str.strip() == "":
        action_log_str = "No actions logged."

    y_offset = 10
    for line in action_log_str.splitlines():
        # Render the line using the side panel text color (white)
        text_surface = small_font.render(line, True, SIDE_PANEL_TEXT_COLOR)
        screen.blit(text_surface, (panel_x + 5, y_offset))
        y_offset += text_surface.get_height() + 4

def draw(world, predator_count, prey_count, turn, action_log_str=""):
    """
    Draw the entire screen:
     - A grid background, food, and agents.
     - An info bar below the grid.
     - The side panel with a custom action log string.
    """
    screen.fill(BLACK)

    # Draw grid background (white cells)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(
                x * (CELL_SIZE + MARGIN) + MARGIN,
                y * (CELL_SIZE + MARGIN) + MARGIN,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, WHITE, rect)

    # Draw food (cell with label "F")
    fx, fy = world.food
    draw_cell(fx, fy, [FOOD_COLOR], ["F"])

    # Group agents by tile
    tile_map = {}
    for agent in world.agents:
        pos = agent.position
        if pos not in tile_map:
            tile_map[pos] = []
        tile_map[pos].append(agent)

    # Draw agents with stacking and labels
    for (x, y), agents in tile_map.items():
        colors = []
        labels = []
        for agent in agents:
            is_original = hasattr(world, "rl_agent") and (agent is world.rl_agent)
            if agent.__class__.__name__ == "Prey":
                color = STARVING_PREY_COLOR if getattr(agent, "steps_since_last_meal", 0) >= PREY_STARVATION_TICKS else PREY_COLOR
                colors.append(color)
                labels.append("OG-P" if is_original else "P")
            else:
                colors.append(PREDATOR_COLOR)
                labels.append("H")

        draw_cell(x, y, colors, labels)

    # Draw info bar below the grid
    info = f"Turn: {turn}   Prey: {prey_count}   Predators: {predator_count}"
    info_surface = font.render(info, True, TEXT_COLOR)
    screen.blit(info_surface, (10, int(HEIGHT) + 15))

    # Draw the side panel with the provided custom action log text.
    ActionDraw(action_log_str)

    pygame.display.flip()

def draw_cell(x, y, colors, labels=None):
    base_x = x * (CELL_SIZE + MARGIN) + MARGIN
    base_y = y * (CELL_SIZE + MARGIN) + MARGIN
    n = len(colors)

    if n == 1:
        pygame.draw.rect(screen, colors[0], (base_x, base_y, CELL_SIZE, CELL_SIZE))
        if labels:
            label = small_font.render(labels[0], True, TEXT_COLOR)
            screen.blit(label, (base_x + 5, base_y + 5))
    else:
        sub_size = CELL_SIZE // 2
        for idx, color in enumerate(colors[:4]):
            sub_x = idx % 2
            sub_y = idx // 2
            rect = pygame.Rect(
                base_x + sub_x * sub_size,
                base_y + sub_y * sub_size,
                sub_size,
                sub_size
            )
            pygame.draw.rect(screen, color, rect)
            if labels and idx < len(labels):
                label = small_font.render(labels[idx], True, TEXT_COLOR)
                screen.blit(label, (rect.x + 4, rect.y + 2))
    