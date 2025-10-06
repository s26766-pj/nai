import pygame
import math
from colors import (
    COLOR_RED,
    COLOR_BLUE,
    COLOR_RED_FILL,
    COLOR_BLUE_FILL,
    COLOR_BLACK,
    COLOR_WHITE,
    COLOR_BG,
    COLOR_ITEM_BG,
    COLOR_ITEM_BG_HOVER,
    COLOR_START_ENABLED,
    COLOR_START_DISABLED,
    COLOR_START_BORDER,
)

PLAYERS = [
    "human1",
    "human2",
    "Minimax",
    "MCTS",
    "Hybrid",
]


def generate_hex_points(cx, cy, size):
    points = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = math.pi / 180 * angle_deg
        x = cx + size * math.cos(angle_rad)
        y = cy + size * math.sin(angle_rad)
        points.append((x, y))
    return points


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    px1, py1 = polygon[0]
    for i in range(1, n + 1):
        px2, py2 = polygon[i % n]
        if min(py1, py2) < y <= max(py1, py2) and x <= max(px1, px2):
            if py1 != py2:
                xinters = (y - py1) * (px2 - px1) / (py2 - py1 + 1e-9) + px1
            else:
                xinters = px1
            if px1 == px2 or x <= xinters:
                inside = not inside
        px1, py1 = px2, py2
    return inside


def draw_hex(surface, center, size, fill_color, outline_color=COLOR_BLACK, outline_width=3):
    points = generate_hex_points(center[0], center[1], size)
    pygame.draw.polygon(surface, fill_color, points, 0)
    pygame.draw.polygon(surface, outline_color, points, outline_width)
    return points


def run_menu() -> tuple[str, str]:
    pygame.init()
    screen = pygame.display.set_mode((1000, 650))
    pygame.display.set_caption("Hex - Choose Players")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 22, bold=True)
    small_font = pygame.font.SysFont("Arial", 16)

    # Layout
    center_x = 500
    # Hex settings and layout for player items (two rows)
    hex_radius = 48
    gap_x = 40
    gap_y = 40

    # Top row: human1 and human2 side-by-side
    humans_y = 300
    human_left_x = center_x - (hex_radius * 2 + gap_x) // 2
    human_right_x = center_x + (hex_radius * 2 + gap_x) // 2

    # Bottom row: 3 AI hexes centered
    ai_y = humans_y + hex_radius + gap_y
    ai_spacing = hex_radius * 2 + gap_x
    ai_xs = [center_x - ai_spacing, center_x, center_x + ai_spacing]

    layout = [
        ("human1", (human_left_x, humans_y)),
        ("human2", (human_right_x, humans_y)),
        ("Minimax", (ai_xs[0], ai_y)),
        ("MCTS", (ai_xs[1], ai_y)),
        ("Hybrid", (ai_xs[2], ai_y)),
    ]

    # Player items as hexagons with precomputed polygons
    items = []
    for name, (cx, cy) in layout:
        poly = generate_hex_points(cx, cy, hex_radius)
        minx = min(p[0] for p in poly)
        miny = min(p[1] for p in poly)
        maxx = max(p[0] for p in poly)
        maxy = max(p[1] for p in poly)
        rect = pygame.Rect(minx, miny, maxx - minx, maxy - miny)
        items.append({"name": name, "center": (cx, cy), "radius": hex_radius, "poly": poly, "rect": rect})

    # Drop targets (hexes)
    red_center = (200, 250)
    blue_center = (800, 250)
    hex_size = 70
    red_hex = generate_hex_points(*red_center, hex_size)
    blue_hex = generate_hex_points(*blue_center, hex_size)

    selected_red: str | None = None
    selected_blue: str | None = None

    dragging = False
    drag_name: str | None = None
    drag_pos = (0, 0)  # current mouse position while dragging
    drag_radius = hex_radius

    # Start button (hex) layout - same size as player hexes, above humans
    start_font = pygame.font.SysFont("Arial", 28, bold=True)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Try to start drag first (so button click won't block dragging)
                for it in items:
                    if point_in_polygon((mx, my), it["poly"]):
                        dragging = True
                        drag_name = it["name"]
                        drag_radius = it["radius"]
                        drag_pos = (mx, my)
                        break

                # If not dragging, allow clearing selection by clicking hexes
                if not dragging:
                    if point_in_polygon((mx, my), red_hex):
                        selected_red = None
                    elif point_in_polygon((mx, my), blue_hex):
                        selected_blue = None

                    # If both selected and click START hex, proceed
                    if selected_red is not None and selected_blue is not None:
                        # compute start hex polygon (above human hexes)
                        start_center = (center_x, humans_y - (hex_radius + gap_y))
                        start_hex = generate_hex_points(*start_center, hex_radius)
                        if point_in_polygon((mx, my), start_hex):
                            running = False

            # (obsolete start_rect click removed; hex START is handled above)

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and dragging:
                mx, my = event.pos
                # Drop into targets
                if drag_name is not None:
                    if point_in_polygon((mx, my), red_hex):
                        if selected_blue == drag_name:
                            selected_blue = None
                        selected_red = drag_name
                    elif point_in_polygon((mx, my), blue_hex):
                        if selected_red == drag_name:
                            selected_red = None
                        selected_blue = drag_name
                dragging = False
                drag_name = None
                drag_pos = (0, 0)

            if event.type == pygame.MOUSEMOTION and dragging:
                mx, my = event.pos
                drag_pos = (mx, my)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    # Only allow continue when both sides selected
                    if selected_red is not None and selected_blue is not None:
                        running = False

        # Draw
        screen.fill(COLOR_BG)

        # Draw hex targets
        draw_hex(screen, red_center, hex_size, COLOR_RED_FILL, COLOR_RED, 6)
        draw_hex(screen, blue_center, hex_size, COLOR_BLUE_FILL, COLOR_BLUE, 6)

        # Labels under hexes
        red_label = font.render("RED", True, COLOR_RED)
        blue_label = font.render("BLUE", True, COLOR_BLUE)
        screen.blit(red_label, (red_center[0] - red_label.get_width() // 2, red_center[1] + hex_size + 12))
        screen.blit(blue_label, (blue_center[0] - blue_label.get_width() // 2, blue_center[1] + hex_size + 12))

        # Current assignments
        assign_font = pygame.font.SysFont("Arial", 20, bold=True)
        red_assign_text = selected_red if selected_red is not None else "(drag here)"
        blue_assign_text = selected_blue if selected_blue is not None else "(drag here)"
        red_assign = assign_font.render(red_assign_text, True, COLOR_BLACK)
        blue_assign = assign_font.render(blue_assign_text, True, COLOR_BLACK)
        screen.blit(red_assign, (red_center[0] - red_assign.get_width() // 2, red_center[1] - hex_size - 28))
        screen.blit(blue_assign, (blue_center[0] - blue_assign.get_width() // 2, blue_center[1] - hex_size - 28))

        # Player list (center column as hexagons)
        mx, my = pygame.mouse.get_pos()
        for it in items:
            is_hover = point_in_polygon((mx, my), it["poly"]) and not dragging
            fill = COLOR_ITEM_BG_HOVER if is_hover else COLOR_ITEM_BG
            draw_hex(screen, it["center"], it["radius"], fill, COLOR_BLACK, 3)
            text = font.render(it["name"], True, COLOR_BLACK)
            screen.blit(text, (it["center"][0] - text.get_width() // 2, it["center"][1] - text.get_height() // 2))

        # Dragging item (on top) as a hexagon following the cursor
        if dragging and drag_name is not None:
            draw_hex(screen, drag_pos, drag_radius, COLOR_ITEM_BG_HOVER, COLOR_BLACK, 3)
            text = font.render(drag_name, True, COLOR_BLACK)
            screen.blit(text, (drag_pos[0] - text.get_width() // 2, drag_pos[1] - text.get_height() // 2))

        # Draw START hex button (same size as player hex) above human players
        if selected_red is not None and selected_blue is not None:
            start_center = (center_x, humans_y - (hex_radius + gap_y))
            draw_hex(screen, start_center, hex_radius, COLOR_START_ENABLED, COLOR_START_BORDER, 4)
            label = start_font.render("START", True, COLOR_WHITE)
            screen.blit(label, (start_center[0] - label.get_width() // 2, start_center[1] - label.get_height() // 2))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return selected_red, selected_blue


if __name__ == "__main__":
    # Quick standalone test
    red, blue = run_menu()
    print("Selected:", red, blue)

