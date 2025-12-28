import math
import random
import numpy as np
import pygame
from opensimplex import OpenSimplex
import threading

# Paramètres
TEX_W = 512
TEX_H = 256
SCREEN_W = 900
SCREEN_H = 700
BG_COLOR = (10, 10, 30)

def lerp(a, b, t):
    return a + (b - a) * t

def blend_colors(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))

class PlanetGenerator:
    def __init__(self, width=TEX_W, height=TEX_H, seed=None):
        self.w = width
        self.h = height
        self.seed = seed if seed is not None else random.randrange(0, 1000000)
        self.noise = OpenSimplex(seed=self.seed)

    def octave_noise(self, x, y, z, octaves=5, persistence=0.5, lacunarity=2.0):
        amp = 1.0
        freq = 1.0
        total = 0.0
        max_amp = 0.0
        for _ in range(octaves):
            # 'opensimplex' exposes noise3 (3D noise) in this environment
            total += amp * self.noise.noise3(x * freq, y * freq, z * freq)
            max_amp += amp
            amp *= persistence
            freq *= lacunarity
        return total / max_amp

    def generate_elevation(self):
        w, h = self.w, self.h
        elev = np.zeros((h, w), dtype=np.float32)
        for j in range(h):
            lat = (j / (h - 1)) * math.pi - math.pi / 2  # -pi/2..pi/2
            sin_lat = math.sin(lat)
            cos_lat = math.cos(lat)
            for i in range(w):
                lon = (i / (w - 1)) * 2 * math.pi - math.pi  # -pi..pi
                x = math.cos(lat) * math.cos(lon)
                y = math.sin(lat)
                z = math.cos(lat) * math.sin(lon)
                # base continental features (large scale)
                n1 = self.octave_noise(x * 1.5, y * 1.5, z * 1.5, octaves=5)
                # detail
                n2 = self.octave_noise(x * 6.0, y * 6.0, z * 6.0, octaves=3)
                val = 0.6 * n1 + 0.4 * n2
                # bias towards oceans a bit
                val = val - 0.15
                elev[j, i] = val
        # normalize to -1..1
        minv = elev.min()
        maxv = elev.max()
        elev = (elev - minv) / (maxv - minv) * 2.0 - 1.0
        return elev

    def generate_texture(self):
        elev = self.generate_elevation()
        h, w = elev.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # palette
        deep_water = (7, 25, 95)
        shallow_water = (50, 100, 200)
        sand = (194, 178, 128)
        grass = (90, 160, 80)
        forest = (34, 110, 50)
        rock = (120, 120, 120)
        snow = (240, 240, 240)

        for j in range(h):
            for i in range(w):
                e = elev[j, i]
                if e < -0.25:
                    t = np.clip((e + 1.0) / 0.75, 0.0, 1.0)
                    c = blend_colors(deep_water, shallow_water, t)
                elif e < -0.02:
                    t = np.clip((e + 0.25) / 0.23, 0.0, 1.0)
                    c = blend_colors(shallow_water, sand, t)
                elif e < 0.12:
                    t = np.clip((e + 0.02) / 0.14, 0.0, 1.0)
                    c = blend_colors(sand, grass, t)
                elif e < 0.45:
                    t = np.clip((e - 0.12) / 0.33, 0.0, 1.0)
                    c = blend_colors(grass, forest, t)
                elif e < 0.7:
                    t = np.clip((e - 0.45) / 0.25, 0.0, 1.0)
                    c = blend_colors(forest, rock, t)
                else:
                    t = np.clip((e - 0.7) / 0.3, 0.0, 1.0)
                    c = blend_colors(rock, snow, t)
                img[j, i] = c
        return img

class PlanetRenderer:
    def __init__(self, texture, screen, radius=260):
        self.texture = texture  # numpy array h,w,3
        self.screen = screen
        self.cx = SCREEN_W // 2
        self.cy = SCREEN_H // 2
        self.radius = radius
        self.scale = 1.0
        # precompute sphere vectors for pixels inside circle
        self._precompute_pixel_vectors()
        self.yaw = 0.0
        self.pitch = 0.0

    def _precompute_pixel_vectors(self):
        r = self.radius
        cx = self.cx
        cy = self.cy
        # créer des grilles 2D de coordonnées écran pour avoir dx,dy de forme (H,W)
        xs, ys = np.meshgrid(np.arange(SCREEN_W), np.arange(SCREEN_H))
        dx = (xs - cx) / r
        dy = (ys - cy) / r
        mask = dx * dx + dy * dy <= 1.0
        idxs = np.where(mask)
        px = dx[mask]
        py = dy[mask]
        # on-screen coordinates -> sphere surface: z = sqrt(1 - x^2 - y^2)
        pz = np.sqrt(np.clip(1.0 - px * px - py * py, 0.0, 1.0))
        # store in array shape (N,3)
        self.mask = mask
        self.pixel_coords = np.stack((px, -py, pz), axis=1)  # y inverted to match screen
        self.index_y = idxs[0]
        self.index_x = idxs[1]
        self.N = self.pixel_coords.shape[0]

    def rotate_vectors(self, yaw, pitch):
        # build rotation matrix (apply pitch then yaw)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        # yaw around y-axis
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        # pitch around x-axis
        Rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        R = Ry @ Rp
        # apply
        rotated = self.pixel_coords @ R.T
        return rotated

    def render(self, yaw, pitch, light_dir=(0.5, 1.0, 0.2)):
        if self.texture is None:
            return
        tex_h, tex_w, _ = self.texture.shape
        rotated = self.rotate_vectors(yaw, pitch)
        x_r = rotated[:, 0]
        y_r = rotated[:, 1]
        z_r = rotated[:, 2]
        # spherical coords
        lon = np.arctan2(z_r, x_r)  # -pi..pi
        lat = np.arcsin(y_r)       # -pi/2..pi/2
        u = ((lon + math.pi) / (2 * math.pi)) * tex_w
        v = ((lat + math.pi/2) / math.pi) * tex_h
        # wrap u
        u = np.mod(u, tex_w).astype(np.int32)
        v = np.clip(v.astype(np.int32), 0, tex_h - 1)
        # sample texture
        colors = self.texture[v, u]
        # lighting: simple Lambertian with light_dir
        ld = np.array(light_dir)
        ld = ld / np.linalg.norm(ld)
        normals = rotated  # sphere normals
        intensity = np.clip(normals @ ld, 0.0, 1.0)
        intensity = intensity[:, None]
        shaded = np.clip(colors.astype(np.float32) * (0.4 + 0.6 * intensity), 0, 255).astype(np.uint8)
        # build full image and blit to screen
        surf_arr = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        surf_arr[self.index_y, self.index_x] = shaded
        surf = pygame.surfarray.make_surface(surf_arr.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption('Simulation de Planète - prototype')
    clock = pygame.time.Clock()

    # génération asynchrone de la texture pour éviter de bloquer l'UI
    generator = PlanetGenerator()
    texture_holder = {"texture": None}
    texture_ready = threading.Event()

    def worker():
        texture_holder["texture"] = generator.generate_texture()
        texture_ready.set()

    threading.Thread(target=worker, daemon=True).start()
    renderer = None

    dragging = False
    last_mouse = (0, 0)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:  # wheel up
                    if renderer is not None:
                        renderer.radius = min(400, renderer.radius + 10)
                        renderer._precompute_pixel_vectors()
                elif event.button == 5:  # wheel down
                    if renderer is not None:
                        renderer.radius = max(80, renderer.radius - 10)
                        renderer._precompute_pixel_vectors()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging and renderer is not None:
                    mx, my = event.pos
                    lx, ly = last_mouse
                    dx = mx - lx
                    dy = my - ly
                    # adjust sensitivity
                    renderer.yaw += dx * 0.01
                    renderer.pitch += dy * 0.01
                    renderer.pitch = np.clip(renderer.pitch, -math.pi/2 + 0.01, math.pi/2 - 0.01)
                    last_mouse = (mx, my)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # relancer génération asynchrone
                    generator = PlanetGenerator(seed=None)
                    texture_ready.clear()
                    threading.Thread(target=worker, daemon=True).start()
                    if renderer is not None:
                        renderer.texture = None

        screen.fill(BG_COLOR)
        # si texture prête, créer le renderer (la première fois) ou mettre à jour la texture
        if texture_ready.is_set():
            if renderer is None:
                renderer = PlanetRenderer(texture_holder["texture"], screen)
            else:
                renderer.texture = texture_holder["texture"]

        if renderer is not None:
            renderer.render(renderer.yaw, renderer.pitch)
        else:
            # afficher message de génération
            font = pygame.font.SysFont('arial', 24)
            surf = font.render('Génération de la planète...', True, (220,220,220))
            screen.blit(surf, (SCREEN_W//2 - surf.get_width()//2, SCREEN_H//2 - surf.get_height()//2))

        # overlay instructions
        font = pygame.font.SysFont('arial', 14)
        texts = ["Cliquer-glisser : rotation", "Molette : zoom", "R : régénérer"]
        for i, t in enumerate(texts):
            surf = font.render(t, True, (220, 220, 220))
            screen.blit(surf, (10, 10 + i * 18))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == '__main__':
    main()
