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

class SpaceBackground:
    def __init__(self, width, height, seed=42):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.noise = OpenSimplex(seed=seed)
        self.generate()

    def generate(self):
        self.surface.fill((5, 5, 15))  # Deep space color

        # Generate nebula/galaxies using noise
        for y in range(0, self.height, 2):
            for x in range(0, self.width, 2):
                n = self.noise.noise2(x * 0.005, y * 0.005)
                if n > 0.3:
                    # Blue/Purple nebula
                    r = int(20 * (n - 0.3))
                    g = int(10 * (n - 0.3))
                    b = int(50 * (n - 0.3))
                    pygame.draw.rect(self.surface, (min(r, 255), min(g, 255), min(b, 255)), (x, y, 2, 2))
                elif n < -0.4:
                    # Reddish nebula
                    r = int(40 * abs(n + 0.4))
                    g = int(5 * abs(n + 0.4))
                    b = int(10 * abs(n + 0.4))
                    pygame.draw.rect(self.surface, (min(r, 255), min(g, 255), min(b, 255)), (x, y, 2, 2))

        # Generate stars
        for _ in range(300):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            size = random.choice([1, 1, 1, 2])
            brightness = random.randint(150, 255)
            color = (brightness, brightness, brightness)
            if size == 1:
                self.surface.set_at((x, y), color)
            else:
                pygame.draw.circle(self.surface, color, (x, y), size - 0.5)

    def draw(self, screen, yaw=0, pitch=0):
        # Subtle parallax effect
        ox = int(yaw * 20) % self.width
        oy = int(pitch * 20) % self.height
        
        # Draw with wrapping for parallax
        screen.blit(self.surface, (-ox, -oy))
        screen.blit(self.surface, (self.width - ox, -oy))
        screen.blit(self.surface, (-ox, self.height - oy))
        screen.blit(self.surface, (self.width - ox, self.height - oy))

class PlanetGenerator:
    def __init__(self, width=TEX_W, height=TEX_H, seed=None):
        self.w = width
        self.h = height
        self.seed = seed if seed is not None else random.randrange(0, 1000000)
        self.noise = OpenSimplex(seed=self.seed)
        self.rng = np.random.RandomState(self.seed)

    def octave_noise(self, x, y, z, octaves=5, persistence=0.5, lacunarity=2.0):
        amp = 1.0
        freq = 1.0
        total = 0.0
        max_amp = 0.0
        for _ in range(octaves):
            # 'opensimplex' expose noise3 (3D noise)
            total += amp * self.noise.noise3(x * freq, y * freq, z * freq)
            max_amp += amp
            amp *= persistence
            freq *= lacunarity
        return total / max_amp

    def generate_elevation(self):
        w, h = self.w, self.h
        elev = np.zeros((h, w), dtype=np.float32)
        # génération primaire avec couche de "ridge" pour montagnes plus marquées
        for j in range(h):
            lat = (j / (h - 1)) * math.pi - math.pi / 2  # -pi/2..pi/2
            for i in range(w):
                lon = (i / (w - 1)) * 2 * math.pi - math.pi  # -pi..pi
                x = math.cos(lat) * math.cos(lon)
                y = math.sin(lat)
                z = math.cos(lat) * math.sin(lon)

                # grandes masses continentales
                n1 = self.octave_noise(x * 1.5, y * 1.5, z * 1.5, octaves=5)
                # détails
                n2 = self.octave_noise(x * 6.0, y * 6.0, z * 6.0, octaves=3)
                # bruit haute fréquence pour créer des crêtes de montagne (ridge)
                nm = self.octave_noise(x * 12.0, y * 12.0, z * 12.0, octaves=2)
                ridge = abs(nm)

                val = 0.55 * n1 + 0.35 * n2 + 0.22 * ridge
                # bias vers océans
                val = val - 0.2
                elev[j, i] = val

        # normaliser à -1..1
        minv = elev.min()
        maxv = elev.max()
        elev = (elev - minv) / (maxv - minv) * 2.0 - 1.0

        # calculer pente approximative (gradient) pour placer forêts / villages
        gy, gx = np.gradient(elev)
        slope = np.sqrt(gx * gx + gy * gy)

        # carte d'humidité (motif pour forêts) — bruit à fréquence moyenne
        moisture = np.zeros_like(elev)
        for j in range(h):
            lat = (j / (h - 1)) * math.pi - math.pi / 2
            for i in range(w):
                lon = (i / (w - 1)) * 2 * math.pi - math.pi
                x = math.cos(lat) * math.cos(lon)
                y = math.sin(lat)
                z = math.cos(lat) * math.sin(lon)
                m = self.octave_noise(x * 3.0, y * 3.0, z * 3.0, octaves=3)
                moisture[j, i] = m

        # normaliser moisture 0..1
        moisture = (moisture - moisture.min()) / (moisture.max() - moisture.min() + 1e-9)

        return elev, slope, moisture

    def generate_cloud_map(self, width=None, height=None, samples=8, base=0.02, thickness=0.05, scale=3.0, smooth=1):
        """
        Pré-intégration de la densité de nuages: pour chaque texel (lon,lat)
        on échantillonne le bruit 3D le long d'une colonne de hauteur (base..base+thickness)
        et on moyenne pour obtenir une densité. On applique ensuite un lissage simple.
        Cette méthode est exécutée à la génération (worker thread), donc elle peut être
        un peu coûteuse.
        """
        w = self.w if width is None else width
        h = self.h if height is None else height
        cloud = np.zeros((h, w), dtype=np.float32)
        # échantillonnage discret le long de la colonne
        for j in range(h):
            lat = (j / (h - 1)) * math.pi - math.pi / 2
            cos_lat = math.cos(lat)
            sin_lat = math.sin(lat)
            for i in range(w):
                lon = (i / (w - 1)) * 2 * math.pi - math.pi
                sumd = 0.0
                for k in range(samples):
                    t = k / (samples - 1) if samples > 1 else 0.0
                    r = 1.0 + base + t * thickness
                    x = r * cos_lat * math.cos(lon)
                    y = r * sin_lat
                    z = r * cos_lat * math.sin(lon)
                    n = self.octave_noise(x * scale, y * scale, z * scale, octaves=3)
                    # normalize noise -1..1 -> 0..1
                    sumd += (n * 0.5 + 0.5)
                cloud[j, i] = sumd / samples

        # normalize 0..1
        cloud = (cloud - cloud.min()) / (cloud.max() - cloud.min() + 1e-9)
        # boost contrast to form distinct cloud patches
        cloud = np.clip((cloud - 0.4) * 1.6, 0.0, 1.0)

        # simple box blur (1 pass)
        if smooth and smooth > 0:
            padded = np.pad(cloud, ((1, 1), (1, 1)), mode='edge')
            tmp = np.empty_like(cloud)
            for y in range(h):
                for x in range(w):
                    tmp[y, x] = np.mean(padded[y:y+3, x:x+3])
            cloud = tmp

        return cloud

    def generate_texture(self):
        elev, slope, moisture = self.generate_elevation()
        h, w = elev.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # palette
        deep_water = (7, 25, 95)
        shallow_water = (50, 100, 200)
        sand = (194, 178, 128)
        grass = (90, 160, 80)
        forest = (30, 100, 45)
        rock = (120, 120, 120)
        snow = (240, 240, 240)
        town_color = (200, 180, 160)

        # colorisation de base
        for j in range(h):
            for i in range(w):
                e = elev[j, i]
                m = moisture[j, i]
                s = slope[j, i]
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
                    base = blend_colors(grass, forest, np.clip((m - 0.35) / 0.65, 0.0, 1.0))
                    t = np.clip((e - 0.12) / 0.33, 0.0, 1.0)
                    c = blend_colors(base, rock, t if s < 0.08 else min(1.0, t + 0.2))
                elif e < 0.7:
                    t = np.clip((e - 0.45) / 0.25, 0.0, 1.0)
                    c = blend_colors(forest, rock, t)
                else:
                    t = np.clip((e - 0.7) / 0.3, 0.0, 1.0)
                    c = blend_colors(rock, snow, t)
                img[j, i] = c

        # petites variations de forêt
        for j in range(h):
            for i in range(w):
                e = elev[j, i]
                m = moisture[j, i]
                s = slope[j, i]
                if 0.02 <= e <= 0.45 and m > 0.4 and s < 0.08:
                    dens = np.clip((m - 0.4) / 0.6, 0.0, 1.0) * (1.0 - min(1.0, s * 10.0))
                    if self.rng.rand() < dens * 0.25:
                        base = tuple(int(v) for v in img[j, i])
                        img[j, i] = blend_colors(base, forest, 0.7)

        # placement de villages
        towns_count = 18
        placed = 0
        attempts = 0
        max_attempts = towns_count * 40
        town_mask = np.zeros((h, w), dtype=np.uint8)
        while placed < towns_count and attempts < max_attempts:
            attempts += 1
            ci = self.rng.randint(0, w)
            cj = self.rng.randint(0, h)
            e = elev[cj, ci]
            s = slope[cj, ci]
            if not (-0.02 < e < 0.4):
                continue
            if s > 0.05:
                continue
            r = 8
            y0 = max(0, cj - r)
            y1 = min(h, cj + r + 1)
            x0 = max(0, ci - r)
            x1 = min(w, ci + r + 1)
            win = elev[y0:y1, x0:x1]
            if not np.any(win < -0.02):
                continue
            radius = self.rng.randint(2, 5)
            yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = xx*xx + yy*yy <= radius*radius
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    y = cj + dy
                    x = ci + dx
                    if 0 <= x < w and 0 <= y < h and mask[dy+radius, dx+radius]:
                        base = tuple(int(v) for v in img[y, x])
                        img[y, x] = blend_colors(base, town_color, 0.85)
                        town_mask[y, x] = 1
            placed += 1

        # créer positions 3D et normales à partir de la heightmap
        height_scale = 0.06
        pos = np.zeros((h, w, 3), dtype=np.float32)
        for j in range(h):
            lat = (j / (h - 1)) * math.pi - math.pi / 2
            for i in range(w):
                lon = (i / (w - 1)) * 2 * math.pi - math.pi
                r = 1.0 + elev[j, i] * height_scale
                x = r * math.cos(lat) * math.cos(lon)
                y = r * math.sin(lat)
                z = r * math.cos(lat) * math.sin(lon)
                pos[j, i, 0] = x
                pos[j, i, 1] = y
                pos[j, i, 2] = z

        dlat = np.gradient(pos, axis=0)
        dlon = np.gradient(pos, axis=1)
        normals = np.cross(dlon, dlat)
        norms = np.linalg.norm(normals, axis=2, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms
        normals = normals.astype(np.float32)

        # ambient occlusion simple via blur of elevation
        padded = np.pad(elev, ((1,1),(1,1)), mode='edge')
        blurred = np.empty_like(elev)
        for y in range(h):
            for x in range(w):
                blurred[y, x] = np.mean(padded[y:y+3, x:x+3])
        ao = 1.0 - np.clip((blurred - elev) * 4.0, 0.0, 1.0)
        ao = (0.6 + 0.4 * ao).astype(np.float32)

        # generate cloud density map
        cloud = self.generate_cloud_map(width=w, height=h, samples=8, base=0.02, thickness=0.05, scale=3.0, smooth=1)

        tex = {
            'color': img,
            'normal': normals,
            'ao': ao,
            'elev': elev,
            'town_mask': town_mask,
            'cloud_density': cloud
        }
        return tex

class PlanetRenderer:
    def __init__(self, texture_dict, screen, radius=260):
        # texture_dict: {'color','normal','ao',...}
        self.texture_dict = texture_dict
        self.texture_color = texture_dict['color'] if texture_dict is not None else None
        self.normal_map = texture_dict['normal'] if texture_dict is not None else None
        self.ao_map = texture_dict['ao'] if texture_dict is not None else None
        self.cloud_map = texture_dict.get('cloud_density', None) if texture_dict is not None else None
        self.screen = screen
        self.cx = SCREEN_W // 2
        self.cy = SCREEN_H // 2
        self.radius = radius
        self.scale = 1.0
        self._precompute_pixel_vectors()
        self.yaw = 0.0
        self.pitch = 0.0
        # cloud rendering params
        self.cloud_shadow_k = 2.0
        self.cloud_ambient = 0.25
        self.cloud_albedo = 1.0
        # cloud animation (uv offset in texel units) and wind (fraction of texture per second)
        # cloud_wind is fraction of full texture per second (u_fraction, v_fraction)
        self.cloud_offset = np.array([0.0, 0.0], dtype=np.float32)
        self.cloud_wind = np.array([0.02, 0.0], dtype=np.float32)  # slow eastward wind by default
        # shadow sampling distance (fraction of texture width/height to offset when sampling shadow)
        self.cloud_shadow_uv = 0.06

    def _precompute_pixel_vectors(self):
        r = self.radius
        cx = self.cx
        cy = self.cy
        xs, ys = np.meshgrid(np.arange(SCREEN_W), np.arange(SCREEN_H))
        dx = (xs - cx) / r
        dy = (ys - cy) / r
        mask = dx * dx + dy * dy <= 1.0
        idxs = np.where(mask)
        px = dx[mask]
        py = dy[mask]
        pz = np.sqrt(np.clip(1.0 - px * px - py * py, 0.0, 1.0))
        self.mask = mask
        self.pixel_coords = np.stack((px, -py, pz), axis=1)
        self.index_y = idxs[0]
        self.index_x = idxs[1]
        self.N = self.pixel_coords.shape[0]

    def rotate_vectors(self, yaw, pitch):
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        R = Ry @ Rp
        rotated = self.pixel_coords @ R.T
        return rotated, R

    def render(self, yaw, pitch, light_dir=(0.5, 1.0, 0.2)):
        if self.texture_color is None:
            return
        tex_h, tex_w, _ = self.texture_color.shape
        rotated, R = self.rotate_vectors(yaw, pitch)
        x_r = rotated[:, 0]
        y_r = rotated[:, 1]
        z_r = rotated[:, 2]
        lon = np.arctan2(z_r, x_r)
        lat = np.arcsin(y_r)
        # compute float UV for bilinear sampling
        u_f = ((lon + math.pi) / (2 * math.pi)) * tex_w
        v_f = ((lat + math.pi/2) / math.pi) * tex_h
        # integer indices for nearest color lookup (wrap lon)
        u = np.mod(u_f, tex_w).astype(np.int32)
        v = np.clip(v_f.astype(np.int32), 0, tex_h - 1)
        colors = self.texture_color[v, u].astype(np.float32)

        # sample normal map and rotate
        normals_tex = self.normal_map[v, u].astype(np.float32)
        normals_rot = normals_tex @ R.T

        # lighting
        ld = np.array(light_dir, dtype=np.float32)
        ld = ld / np.linalg.norm(ld)
        intensity = np.clip(np.sum(normals_rot * ld[None, :], axis=1), 0.0, 1.0)
        intensity = intensity[:, None]
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        half = (ld + view_dir)
        half = half / np.linalg.norm(half)
        spec_val = np.clip(np.sum(normals_rot * half[None, :], axis=1), 0.0, 1.0)
        spec = (spec_val ** 32)[:, None].astype(np.float32)

        ao = self.ao_map[v, u].astype(np.float32)[:, None]

        # base shaded surface (keep float for later blending)
        shaded = colors * (0.2 + 0.8 * intensity)
        shaded = np.clip(shaded + 200.0 * spec, 0, 255)
        shaded = shaded * ao

        # cloud compositing (animated bilinear sampling + directional shadow approximation)
        if self.cloud_map is not None:
            cloud = self.cloud_map
            cf = u_f.copy()
            vf = v_f.copy()
            # update cloud offset using wall time (seconds)
            now = pygame.time.get_ticks() / 1000.0
            if not hasattr(self, '_last_time'):
                self._last_time = now
            dt = max(0.0, now - self._last_time)
            self._last_time = now
            # wind given in fraction of texture per second -> convert to texels
            self.cloud_offset[0] = (self.cloud_offset[0] + self.cloud_wind[0] * tex_w * dt) % tex_w
            self.cloud_offset[1] = self.cloud_offset[1] + self.cloud_wind[1] * tex_h * dt

            # apply advection offset to sampling coordinates
            cf_m = (cf + self.cloud_offset[0]) % tex_w
            vf_m = np.clip(vf + self.cloud_offset[1], 0.0, tex_h - 1.0)

            # bilinear sample at moved coordinates to get local cloud density
            u0 = np.floor(cf_m).astype(np.int32) % tex_w
            v0 = np.floor(vf_m).astype(np.int32)
            v0 = np.clip(v0, 0, tex_h - 1)
            u1 = (u0 + 1) % tex_w
            v1 = np.clip(v0 + 1, 0, tex_h - 1)
            fu = (cf_m - np.floor(cf_m)).astype(np.float32)
            fv = (vf_m - np.floor(vf_m)).astype(np.float32)
            d00 = cloud[v0, u0]
            d10 = cloud[v0, u1]
            d01 = cloud[v1, u0]
            d11 = cloud[v1, u1]
            density = (d00 * (1 - fu) * (1 - fv) + d10 * fu * (1 - fv) + d01 * (1 - fu) * fv + d11 * fu * fv)
            density = np.clip(density, 0.0, 1.0)

            # approximate directional shadow: sample cloud density at an offset along the light direction
            # use light direction components to compute a small UV shift (fraction of texture)
            shadow_uv = self.cloud_shadow_uv
            # offset in texels: negative because shadow falls opposite the sun direction
            u_shadow_offset = -ld[0] * shadow_uv * tex_w
            v_shadow_offset = -ld[1] * shadow_uv * tex_h
            cf_s = (cf_m + u_shadow_offset) % tex_w
            vf_s = np.clip(vf_m + v_shadow_offset, 0.0, tex_h - 1.0)

            u0s = np.floor(cf_s).astype(np.int32) % tex_w
            v0s = np.floor(vf_s).astype(np.int32)
            v0s = np.clip(v0s, 0, tex_h - 1)
            u1s = (u0s + 1) % tex_w
            v1s = np.clip(v0s + 1, 0, tex_h - 1)
            fus = (cf_s - np.floor(cf_s)).astype(np.float32)
            fvs = (vf_s - np.floor(vf_s)).astype(np.float32)
            sd00 = cloud[v0s, u0s]
            sd10 = cloud[v0s, u1s]
            sd01 = cloud[v1s, u0s]
            sd11 = cloud[v1s, u1s]
            density_shadow = (sd00 * (1 - fus) * (1 - fvs) + sd10 * fus * (1 - fvs) + sd01 * (1 - fus) * fvs + sd11 * fus * fvs)
            density_shadow = np.clip(density_shadow, 0.0, 1.0)

            # cloud lighting based on spherical direction (for cloud brightening)
            lit = np.clip(np.sum(rotated * ld[None, :], axis=1), 0.0, 1.0)
            cloud_rgb = (255.0 * (self.cloud_ambient + (1.0 - self.cloud_ambient) * lit))

            # apply shadow to surface using the shadowed density (directional approx)
            shadow = np.exp(-self.cloud_shadow_k * density_shadow).astype(np.float32)
            shaded = shaded * shadow[:, None]

            # alpha and compose clouds over shaded surface (use local density for alpha)
            alpha = np.clip(density * 1.4, 0.0, 1.0).astype(np.float32)
            cloud_rgb3 = np.stack([cloud_rgb, cloud_rgb, cloud_rgb], axis=1)
            shaded = cloud_rgb3 * alpha[:, None] + shaded * (1.0 - alpha)[:, None]

        # finalize and write to surface buffer
        final_colors = np.clip(shaded, 0, 255).astype(np.uint8)
        surf_arr = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        surf_arr[self.index_y, self.index_x] = final_colors
        surf = pygame.surfarray.make_surface(surf_arr.swapaxes(0, 1))
        surf.set_colorkey((0, 0, 0))
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
    
    bg = SpaceBackground(SCREEN_W, SCREEN_H)

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
                        renderer.texture_dict = None
                        renderer.texture_color = None
                        renderer.normal_map = None
                        renderer.ao_map = None
                        renderer.cloud_map = None

        # screen.fill(BG_COLOR)
        if renderer is not None:
            bg.draw(screen, renderer.yaw, renderer.pitch)
        else:
            bg.draw(screen, 0, 0)
            
        # si texture prête, créer le renderer (la première fois) ou mettre à jour la texture
        if texture_ready.is_set():
            if renderer is None:
                renderer = PlanetRenderer(texture_holder["texture"], screen)
            else:
                renderer.texture_dict = texture_holder["texture"]
                # mettre à jour sous-cartes pour le renderer
                if texture_holder["texture"] is not None:
                    renderer.texture_color = texture_holder["texture"]["color"]
                    renderer.normal_map = texture_holder["texture"]["normal"]
                    renderer.ao_map = texture_holder["texture"]["ao"]
                    renderer.cloud_map = texture_holder["texture"].get("cloud_density", None)

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