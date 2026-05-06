"""Generate DarktableMCP logo options."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SIZE = 512
BG    = (13, 13, 13)        # near-black
RED   = (224, 27, 36)       # Darktable red
RED2  = (180, 20, 28)       # darker red
WHITE = (255, 255, 255)
GREY  = (180, 180, 180)
DIM   = (60, 60, 60)
ORANGE= (230, 100, 30)

OUT = r"D:\Dev\DarktableMCP\logos"
import os; os.makedirs(OUT, exist_ok=True)


def new(bg=BG):
    img = Image.new("RGBA", (SIZE, SIZE), bg + (255,))
    return img, ImageDraw.Draw(img)


def save(img, name):
    path = f"{OUT}\\{name}.png"
    img.save(path)
    print(f"Saved {path}")
    return path


# ── helpers ────────────────────────────────────────────────────────────────

def circle_points(cx, cy, r, n=64):
    return [(cx + r * math.cos(2*math.pi*i/n),
             cy + r * math.sin(2*math.pi*i/n)) for i in range(n)]


def draw_ring(draw, cx, cy, r_out, r_in, color, n=256):
    for i in range(n):
        a0 = 2*math.pi*i/n
        a1 = 2*math.pi*(i+1)/n
        pts = [
            (cx + r_out*math.cos(a0), cy + r_out*math.sin(a0)),
            (cx + r_out*math.cos(a1), cy + r_out*math.sin(a1)),
            (cx + r_in *math.cos(a1), cy + r_in *math.sin(a1)),
            (cx + r_in *math.cos(a0), cy + r_in *math.sin(a0)),
        ]
        draw.polygon(pts, fill=color)


def draw_aperture_blade(draw, cx, cy, r, angle_deg, color):
    """One aperture blade."""
    a = math.radians(angle_deg)
    spread = math.radians(28)
    tip_r  = r * 0.18
    pts = []
    for da in np.linspace(-spread, spread, 20):
        pts.append((cx + r * math.cos(a + da), cy + r * math.sin(a + da)))
    for da in np.linspace(spread, -spread, 20):
        pts.append((cx + tip_r * math.cos(a + math.pi + da),
                    cy + tip_r * math.sin(a + math.pi + da)))
    draw.polygon(pts, fill=color)


# ══════════════════════════════════════════════════════════════════════════
# LOGO 1 – Aperture iris with circuit traces
# ══════════════════════════════════════════════════════════════════════════
def logo1():
    img, draw = new()
    cx = cy = SIZE // 2

    # outer glow ring
    for r in range(220, 200, -1):
        alpha = int(60 * (r - 200) / 20)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=RED + (alpha,), width=1)

    # outer dark ring
    draw_ring(draw, cx, cy, 215, 195, (30, 30, 30))
    draw_ring(draw, cx, cy, 195, 192, RED)

    # 9 aperture blades in layered dark/red
    for i in range(9):
        angle = i * 40
        draw_aperture_blade(draw, cx, cy, 185, angle, (20, 20, 20))
        draw_aperture_blade(draw, cx, cy, 170, angle + 20, (35, 8, 8))

    # inner circle (lens glass look)
    for r in range(90, 0, -1):
        t = r / 90
        col = (int(10 + t*20), int(10 + t*20), int(20 + t*40))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=col)

    # circuit traces radiating outward
    rng = np.random.default_rng(42)
    for _ in range(12):
        a = rng.uniform(0, 2*math.pi)
        r0 = 95
        r1 = 185 + rng.integers(0, 20)
        # main trace
        x0, y0 = cx + r0*math.cos(a), cy + r0*math.sin(a)
        # one 90° bend
        mid_r = rng.uniform(130, 160)
        a_bend = a + rng.uniform(-0.3, 0.3)
        xm, ym = cx + mid_r*math.cos(a_bend), cy + mid_r*math.sin(a_bend)
        x1, y1 = cx + r1*math.cos(a_bend + rng.uniform(-0.1,0.1)), cy + r1*math.sin(a_bend + rng.uniform(-0.1,0.1))
        col = RED + (120,) if rng.random() > 0.5 else (80, 80, 80, 120)
        draw.line([(x0,y0),(xm,ym),(x1,y1)], fill=col, width=1)
        # node dot
        draw.ellipse([x1-3, y1-3, x1+3, y1+3], fill=RED+(180,))

    # centre aperture hole
    draw_ring(draw, cx, cy, 55, 50, RED)
    draw.ellipse([cx-49, cy-49, cx+49, cy+49], fill=BG)

    # "MCP" text in centre
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 28)
        font_sm = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    except:
        font = font_sm = ImageFont.load_default()
    draw.text((cx, cy-8), "MCP", font=font, fill=WHITE, anchor="mm")
    draw.text((cx, cy+18), "darktable", font=font_sm, fill=GREY, anchor="mm")

    # inner ring tick marks
    for i in range(36):
        a = math.radians(i * 10)
        r_o = 192
        r_i = 196 if i % 3 == 0 else 194
        draw.line([(cx+r_o*math.cos(a), cy+r_o*math.sin(a)),
                   (cx+r_i*math.cos(a), cy+r_i*math.sin(a))],
                  fill=RED if i % 3 == 0 else DIM, width=1)

    return img

# ══════════════════════════════════════════════════════════════════════════
# LOGO 2 – Hexagonal node network on dark circle
# ══════════════════════════════════════════════════════════════════════════
def logo2():
    img, draw = new()
    cx = cy = SIZE // 2

    # background disc with radial gradient
    arr = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    y_idx, x_idx = np.ogrid[:SIZE, :SIZE]
    dist = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    mask = dist < 220
    t = np.clip(1 - dist / 220, 0, 1)
    arr[mask, 0] = (22 * t[mask]).astype(np.uint8)
    arr[mask, 1] = (22 * t[mask]).astype(np.uint8)
    arr[mask, 2] = (35 * t[mask]).astype(np.uint8)
    arr[mask, 3] = 255
    base = Image.fromarray(arr, "RGBA")
    img = Image.alpha_composite(img, base)
    draw = ImageDraw.Draw(img)

    # outer red ring
    draw_ring(draw, cx, cy, 222, 218, RED)
    draw_ring(draw, cx, cy, 218, 216, (80, 10, 10))

    # AI node network — concentric hex rings
    node_positions = []
    radii = [0, 70, 140]
    counts = [1, 6, 12]
    for r, n in zip(radii, counts):
        for i in range(n):
            a = 2*math.pi*i/n + (math.pi/6 if n==6 else math.pi/12)
            nx, ny = cx + r*math.cos(a), cy + r*math.sin(a)
            node_positions.append((nx, ny))

    # draw connections
    rng = np.random.default_rng(7)
    all_pos = np.array([(p[0], p[1]) for p in node_positions])
    for i, (x1, y1) in enumerate(node_positions):
        dists = np.sqrt((all_pos[:,0]-x1)**2 + (all_pos[:,1]-y1)**2)
        close = np.where((dists < 120) & (dists > 1))[0]
        for j in close:
            x2, y2 = node_positions[j]
            alpha = rng.integers(60, 130)
            col = RED + (alpha,) if rng.random() > 0.4 else (100, 100, 100, alpha)
            draw.line([(x1,y1),(x2,y2)], fill=col, width=1)

    # draw nodes
    for i, (nx, ny) in enumerate(node_positions):
        r_node = 10 if i == 0 else (6 if i < 7 else 4)
        col = RED if i == 0 else (WHITE if i < 7 else GREY)
        draw.ellipse([nx-r_node, ny-r_node, nx+r_node, ny+r_node], fill=col)
        if i == 0:
            draw.ellipse([nx-4, ny-4, nx+4, ny+4], fill=BG)

    # outer nodes on ring
    for i in range(18):
        a = math.radians(i * 20)
        r = 200
        nx, ny = cx + r*math.cos(a), cy + r*math.sin(a)
        draw.ellipse([nx-3, ny-3, nx+3, ny+3], fill=DIM)
        node_positions.append((nx, ny))

    return img

# ══════════════════════════════════════════════════════════════════════════
# LOGO 3 – Lens rings + spark / shutter burst
# ══════════════════════════════════════════════════════════════════════════
def logo3():
    img, draw = new()
    cx = cy = SIZE // 2

    # concentric lens rings
    ring_specs = [(210, 4, DIM), (195, 2, RED), (178, 8, (25,25,25)),
                  (165, 2, DIM), (140, 3, RED2), (100, 2, DIM)]
    for r, w, col in ring_specs:
        for dr in range(w):
            draw.ellipse([cx-(r-dr), cy-(r-dr), cx+(r-dr), cy+(r-dr)],
                         outline=col, width=1)

    # shutter burst lines
    for i in range(16):
        a = math.radians(i * 22.5)
        r0 = 108
        r1 = 185 + (15 if i % 2 == 0 else 0)
        w  = 2 if i % 2 == 0 else 1
        col = RED if i % 4 == 0 else DIM
        draw.line([(cx + r0*math.cos(a), cy + r0*math.sin(a)),
                   (cx + r1*math.cos(a), cy + r1*math.sin(a))],
                  fill=col, width=w)

    # inner lens glass gradient
    arr = np.array(img)
    y_idx, x_idx = np.ogrid[:SIZE, :SIZE]
    dist = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    inner = dist < 98
    t = 1 - dist[inner] / 98
    arr[inner, 0] = np.clip(arr[inner, 0] + (t * 30).astype(int), 0, 255)
    arr[inner, 2] = np.clip(arr[inner, 2] + (t * 60).astype(int), 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # AI "spark" — three bezier-ish curves meeting at centre
    for i in range(6):
        a = math.radians(i * 60)
        x1 = cx + 95 * math.cos(a)
        y1 = cy + 95 * math.sin(a)
        draw.line([(cx, cy), (x1, y1)], fill=RED + (160,), width=2)
        draw.ellipse([x1-5, y1-5, x1+5, y1+5], fill=RED)

    # centre dot
    draw.ellipse([cx-8, cy-8, cx+8, cy+8], fill=WHITE)
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=BG)

    # text below centre
    try:
        font_lg = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 34)
        font_sm = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    except:
        font_lg = font_sm = ImageFont.load_default()

    # pill background for text
    draw.rounded_rectangle([cx-90, cy+108, cx+90, cy+145], radius=8,
                            fill=(30, 30, 30))
    draw.text((cx, cy+120), "DarktableMCP", font=font_sm, fill=WHITE, anchor="mm")
    draw.rounded_rectangle([cx-90, cy+108, cx+90, cy+111], radius=0, fill=RED)

    return img

# ══════════════════════════════════════════════════════════════════════════
# LOGO 4 – Minimal flat mark (good for small icons / favicon)
# ══════════════════════════════════════════════════════════════════════════
def logo4():
    img, draw = new()
    cx = cy = SIZE // 2

    # filled red circle base
    draw.ellipse([cx-200, cy-200, cx+200, cy+200], fill=RED)

    # dark cutout ring
    draw.ellipse([cx-160, cy-160, cx+160, cy+160], fill=BG)

    # 8-blade aperture in white
    for i in range(8):
        angle = i * 45
        draw_aperture_blade(draw, cx, cy, 152, angle, WHITE)
        draw_aperture_blade(draw, cx, cy, 145, angle + 22.5, (200, 200, 200))

    # dark centre
    draw.ellipse([cx-55, cy-55, cx+55, cy+55], fill=BG)

    # small red AI-node constellation
    nodes = [(0,0), (28,-20), (-28,-20), (0,32), (36,18), (-36,18)]
    for i, (dx, dy) in enumerate(nodes):
        nx, ny = cx+dx, cy+dy
        for j, (dx2, dy2) in enumerate(nodes):
            if j > i:
                draw.line([(nx,ny),(cx+dx2,cy+dy2)],
                          fill=RED+(180,), width=1)
    for dx, dy in nodes:
        r = 6 if (dx, dy) == (0, 0) else 3
        draw.ellipse([cx+dx-r, cy+dy-r, cx+dx+r, cy+dy+r], fill=WHITE)

    # outer tick marks
    for i in range(24):
        a = math.radians(i * 15)
        r_o = 198
        r_i = 194 if i % 3 != 0 else 190
        draw.line([(cx+r_o*math.cos(a), cy+r_o*math.sin(a)),
                   (cx+r_i*math.cos(a), cy+r_i*math.sin(a))],
                  fill=WHITE if i % 3 == 0 else (200,200,200), width=1)

    return img


# ── run ────────────────────────────────────────────────────────────────────
paths = []
for i, fn in enumerate([logo1, logo2, logo3, logo4], 1):
    img = fn()
    # composite onto solid black for final PNG (remove alpha)
    bg = Image.new("RGB", (SIZE, SIZE), BG)
    if img.mode == "RGBA":
        bg.paste(img, mask=img.split()[3])
    else:
        bg.paste(img)
    paths.append(save(bg, f"logo_option_{i}"))

print("Done.", paths)
