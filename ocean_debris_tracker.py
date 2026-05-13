#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         OCEAN DEBRIS & OIL SPILL SIMULATOR  —  v10.0 (Data Layers & LPDM)    ║
║         Simulates Lagrangian dispersion using currents, wind leeway,         ║
║         Stokes drift, and turbulent random-walk diffusion.                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import time
import math
import threading
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("TkAgg" if "linux" not in sys.platform else "Agg")
try:
    import matplotlib
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter
import requests

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
EARTH_RADIUS   = 6371000.0   
G_GRAV         = 9.81        
DT_SECONDS     = 1800        
MAX_YEARS      = 10          

C_BG        = "#060d1a"
C_OCEAN     = "#0a1628"
C_LAND      = "#1a2535"
C_COAST     = "#2a4060"
C_GRID      = "#0f2040"
C_TRAJ      = "#00e5ff"
C_PARTICLE  = "#ff6b35"
C_BEACH     = "#ff2244"
C_WIND      = "#a8ff78"
C_WAVE      = "#78c8ff"
C_CURRENT   = "#ffd700"
C_PANEL     = "#0d1b2e"
C_ACCENT    = "#00bcd4"
C_TEXT      = "#cce8ff"
C_WARN      = "#ff9800"

# ─────────────────────────────────────────────────────────────────────────────
# EXACT GEOMETRY LAND MASKER
# ─────────────────────────────────────────────────────────────────────────────
class LandMasker:
    def __init__(self):
        self._NE_CACHE = os.path.join(os.path.expanduser("~"), ".ocean_tracker_ne10m.json")
        self._NE_URL   = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson"
        self.paths, self.bboxes = [], []
        self.coarse_lats, self.coarse_lons, self.coarse_mask = self._build_coarse_mask()
        self._load_high_res()

    def _build_coarse_mask(self):
        lats = np.arange(-90, 91, 1.0)
        lons = np.arange(-180, 181, 1.0)
        land_boxes = [
            (25, 72, -168, -52), (15, 25, -118, -80), (7, 15, -92, -77),
            (-56, 12, -82, -34), (36, 71, -10, 40), (-35, 37, -18, 52),
            (5, 77, 26, 180), (-8, 5, 95, 141), (-43, -10, 113, 154),
            (-47, -34, 166, 178), (60, 84, -55, -17), (-90, -60, -180, 180),
            (-26, -12, 43, 51), (30, 45, 130, 145), (50, 59, -8, 2)
        ]
        mask = np.zeros((len(lats), len(lons)), dtype=bool)
        for lat_min, lat_max, lon_min, lon_max in land_boxes:
            ilat = (lats >= lat_min) & (lats <= lat_max)
            ilon = (lons >= lon_min) & (lons <= lon_max)
            mask[np.ix_(ilat, ilon)] = True
        return lats, lons, mask

    def _load_high_res(self):
        ne_data = None
        if os.path.exists(self._NE_CACHE):
            try:
                with open(self._NE_CACHE) as f: ne_data = json.load(f)
            except Exception: pass
        if not ne_data:
            try:
                print("[Coastline] Downloading High-Res 10m Natural Earth data...")
                r = requests.get(self._NE_URL, timeout=60)
                r.raise_for_status()
                ne_data = r.json()
                with open(self._NE_CACHE, "w") as f: json.dump(ne_data, f)
            except Exception:
                return

        bboxes_temp = []
        for feat in ne_data.get("features", []):
            geom = feat.get("geometry", {})
            gtype = geom.get("type", "")
            coords_list = geom.get("coordinates", [])
            if gtype == "Polygon": coords_list = [coords_list]
            elif gtype == "MultiPolygon": coords_list = [poly[0] for poly in coords_list]
            
            for ring_coords in coords_list:
                if not ring_coords: continue
                ring = np.array(ring_coords)
                if len(ring.shape) == 2 and len(ring) >= 3:
                    self.paths.append(mpath.Path(ring))
                    bboxes_temp.append([ring[:,0].min(), ring[:,0].max(), ring[:,1].min(), ring[:,1].max()])
        if bboxes_temp: self.bboxes = np.array(bboxes_temp)

    def is_land(self, lat, lon):
        if len(self.paths) > 0:
            in_bbox = (self.bboxes[:, 0] <= lon) & (lon <= self.bboxes[:, 1]) & \
                      (self.bboxes[:, 2] <= lat) & (lat <= self.bboxes[:, 3])
            for idx in np.where(in_bbox)[0]:
                if self.paths[idx].contains_point((lon, lat)): return True
            return False
        else:
            ilat = int(np.clip((lat + 90) / 1.0, 0, len(self.coarse_lats)-1))
            ilon = int(np.clip((lon + 180) / 1.0, 0, len(self.coarse_lons)-1))
            return bool(self.coarse_mask[ilat, ilon])

GLOBAL_LAND_MASKER = LandMasker()

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING & SYNTHETIC FIELDS
# ─────────────────────────────────────────────────────────────────────────────
class OceanDataFetcher:
    BASE_METEO  = "https://marine-api.open-meteo.com/v1/marine"
    BASE_WIND   = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        self.cache = {}
        self.online = False
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast?latitude=25&longitude=-80&current=temperature_2m", timeout=5)
            self.online = (r.status_code == 200)
        except: self.online = False

    def fetch_marine(self, lat, lon, hours=240):
        key = f"marine_{lat:.1f}_{lon:.1f}"
        if key in self.cache: return self.cache[key]
        if not self.online: return self._synthetic_marine(lat, lon, hours)
        try:
            params = {"latitude": lat, "longitude": lon, "hourly": "wave_height,wave_direction,wave_period", "forecast_days": min(16, max(1, hours//24 + 1)), "timezone": "UTC"}
            r = requests.get(self.BASE_METEO, params=params, timeout=10)
            data = r.json()
            res = {"wave_height": np.nan_to_num(np.array(data["hourly"]["wave_height"], dtype=float)),
                   "wave_dir": np.nan_to_num(np.array(data["hourly"]["wave_direction"], dtype=float)),
                   "wave_period": np.nan_to_num(np.array(data["hourly"]["wave_period"], dtype=float))}
            self.cache[key] = res
            return res
        except: return self._synthetic_marine(lat, lon, hours)

    def fetch_wind(self, lat, lon, hours=240):
        key = f"wind_{lat:.1f}_{lon:.1f}"
        if key in self.cache: return self.cache[key]
        if not self.online: return self._synthetic_wind(lat, lon, hours)
        try:
            params = {"latitude": lat, "longitude": lon, "hourly": "wind_speed_10m,wind_direction_10m", "forecast_days": min(16, max(1, hours//24 + 1)), "timezone": "UTC", "wind_speed_unit": "ms"}
            r = requests.get(self.BASE_WIND, params=params, timeout=10)
            data = r.json()
            speed = np.nan_to_num(np.array(data["hourly"]["wind_speed_10m"], dtype=float))
            wdir = np.nan_to_num(np.array(data["hourly"]["wind_direction_10m"], dtype=float))
            wdir_rad = np.deg2rad(wdir)
            res = {"u_wind": -speed * np.sin(wdir_rad), "v_wind": -speed * np.cos(wdir_rad), "speed": speed}
            self.cache[key] = res
            return res
        except: return self._synthetic_wind(lat, lon, hours)

    def _synthetic_marine(self, lat, lon, hours):
        t = np.arange(max(hours, 384) + 1)
        wave_h = np.clip(1.5 + 0.8 * np.sin(t * 2*np.pi/12 + lon/20) + np.random.randn(len(t))*0.2, 0.1, 8)
        wave_p = 8 + 4 * np.sin(t * 2*np.pi/24)
        wave_d = (270 - lat * 0.5 + 20*np.sin(t * 2*np.pi/6)) % 360
        return {"wave_height": wave_h, "wave_dir": wave_d, "wave_period": wave_p}

    def _synthetic_wind(self, lat, lon, hours):
        t = np.arange(max(hours, 384) + 1)
        if abs(lat) < 30: base_u, base_v = -5.0, -1.0
        elif abs(lat) < 60: base_u, base_v = 6.0, 1.0
        else: base_u, base_v = -3.0, 0.5
        if lat < 0: base_v = -base_v
        u = base_u + 2*np.sin(t*2*np.pi/24 + 1.2) + np.random.randn(len(t))*0.5
        v = base_v + 1.5*np.cos(t*2*np.pi/18) + np.random.randn(len(t))*0.3
        return {"u_wind": u, "v_wind": v, "speed": np.sqrt(u**2 + v**2)}

    # Fast vectorized field evaluations for global map layers
    def get_global_fields(self, t_hours=0):
        lons = np.arange(-180, 180, 5)
        lats = np.arange(-80, 81, 5)
        LONS, LATS = np.meshgrid(lons, lats)
        
        # Vectorized Current (Simplified Analytical Gyres)
        U_cur = np.zeros_like(LONS, dtype=float)
        V_cur = np.zeros_like(LONS, dtype=float)
        
        U_cur += np.where((LATS>25)&(LATS<50)&(LONS>-82)&(LONS<-10), 1.2*np.exp(-((LATS-38)**2)/25)*0.9, 0)
        V_cur += np.where((LATS>25)&(LATS<50)&(LONS>-82)&(LONS<-10), 1.2*np.exp(-((LATS-38)**2)/25)*0.2, 0)
        
        U_cur += np.where((LATS>15)&(LATS<45)&(LONS>-80)&(LONS<0), -0.15*np.sin(np.radians((LATS-30)*3)), 0)
        V_cur += np.where((LATS>15)&(LATS<45)&(LONS>-80)&(LONS<0), 0.15*np.cos(np.radians((LONS+40)*2)), 0)
        
        U_cur += np.where((LATS>-65)&(LATS<-45), 0.35 + 0.15*np.cos(np.radians(LONS)), 0)
        
        # Vectorized Wind
        U_wind = np.where(np.abs(LATS) < 30, -5.0, np.where(np.abs(LATS) < 60, 6.0, -3.0))
        V_wind = np.where(np.abs(LATS) < 30, -1.0, np.where(np.abs(LATS) < 60, 1.0, 0.5))
        V_wind = np.where(LATS < 0, -V_wind, V_wind)
        
        # Vectorized Waves (Height)
        W_height = np.clip(1.5 + 0.8 * np.sin(LONS/20) + 1.5*np.abs(LATS)/80, 0.1, 8)
        
        return LONS, LATS, U_cur, V_cur, U_wind, V_wind, W_height

    def get_current_at(self, lat, lon, t_hours):
        t_phase = t_hours * 2*math.pi / (24*30)
        u_gyre, v_gyre = 0.0, 0.0

        if 25 < lat < 50 and -82 < lon < -10:
            gs_speed = 1.2 * math.exp(-((lat - (38 + 3*math.sin(t_phase)))**2)/25)
            u_gyre += gs_speed * 0.9; v_gyre += gs_speed * 0.2
        elif 15 < lat < 45 and -80 < lon < 0:
            u_gyre += -0.15 * math.sin(math.radians((lat - 30)*3)); v_gyre += 0.15 * math.cos(math.radians((lon + 40)*2))
        elif 20 < lat < 50 and 120 < lon < 165:
            ks_speed = 0.8 * math.exp(-((lat - 35)**2)/40)
            u_gyre += ks_speed * 0.85; v_gyre += ks_speed * 0.3
        elif 15 < lat < 45 and -180 < lon < -120:
            u_gyre += -0.12 * math.sin(math.radians((lat - 30)*3)); v_gyre += 0.12 * math.cos(math.radians((lon + 150)*2))
        elif -50 < lat < -15 and -160 < lon < -70:
            u_gyre += 0.12 * math.sin(math.radians((lat + 30)*3)); v_gyre -= 0.12 * math.cos(math.radians((lon + 115)*2))
        elif -50 < lat < -15 and -55 < lon < 20:
            u_gyre += 0.10 * math.sin(math.radians((lat + 30)*3)); v_gyre -= 0.10 * math.cos(math.radians((lon + 17)*2))
        
        if -65 < lat < -45:
            u_gyre += 0.35 + 0.15*math.cos(math.radians(lon)); v_gyre += 0.05 * math.sin(math.radians(lon*2))
        if abs(lat) < 5:
            u_gyre += 0.25 * math.cos(math.radians(lon/3)); v_gyre += 0.05 * math.sin(math.radians(lat*10))

        return u_gyre, v_gyre

# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY INTEGRATOR
# ─────────────────────────────────────────────────────────────────────────────
class DebrisParticle:
    def __init__(self, lat0, lon0, windage=0.025, diffusion=10.0, seed=0, pid=0):
        self.pid = pid
        self.lat, self.lon = lat0, lon0
        self.windage = windage * (0.85 + 0.3*np.random.default_rng(seed).random())
        self.diffusion = diffusion # K (m^2/s)
        self.beached, self.beach_lat, self.beach_lon, self.beach_time_hr = False, None, None, None
        self.track_lat, self.track_lon = [lat0], [lon0]

    def step(self, dt, u_cur, v_cur, u_wind, v_wind, wave_h, wave_dir_deg, wave_period):
        if self.beached: return
        lat_r = math.radians(self.lat)
        
        # 1. Stokes Drift (Wave Transport)
        stokes_mag = 0.0
        if wave_period > 0 and wave_h > 0:
            wave_freq = 2*math.pi / max(wave_period, 1)
            wavelength = G_GRAV * wave_period**2 / (2*math.pi)
            stokes_mag = min((math.pi * wave_h**2 * wave_freq) / max(wavelength, 1), 0.3)
        wave_dir_r = math.radians(wave_dir_deg)
        u_stokes = stokes_mag * math.sin(wave_dir_r)
        v_stokes = stokes_mag * math.cos(wave_dir_r)

        # 2. Wind Leeway (Deflected by Ekman layer ~15 degrees)
        leeway_angle = math.radians(15) if self.lat >= 0 else math.radians(-15)
        cos_L, sin_L = math.cos(leeway_angle), math.sin(leeway_angle)
        u_leeway = self.windage * (u_wind * cos_L - v_wind * sin_L)
        v_leeway = self.windage * (u_wind * sin_L + v_wind * cos_L)

        # 3. Deterministic Advection (Currents + Leeway + Stokes)
        u_total = u_cur + u_leeway + u_stokes
        v_total = v_cur + v_leeway + v_stokes

        # 4. Turbulent Diffusion (Random Walk)
        rw_mag = math.sqrt(2 * self.diffusion / dt)
        u_total += rw_mag * np.random.randn()
        v_total += rw_mag * np.random.randn()

        # Update Position
        new_lat = self.lat + (v_total / EARTH_RADIUS) * (180/math.pi) * dt
        new_lon = self.lon + (u_total / (EARTH_RADIUS * max(math.cos(lat_r), 1e-6))) * (180/math.pi) * dt
        new_lon = (new_lon + 180) % 360 - 180
        new_lat = max(-89.9, min(89.9, new_lat))

        if GLOBAL_LAND_MASKER.is_land(new_lat, new_lon):
            self.beached, self.beach_lat, self.beach_lon = True, new_lat, new_lon
        else:
            self.lat, self.lon = new_lat, new_lon

        self.track_lat.append(self.lat)
        self.track_lon.append(self.lon)

class TrajectoryEngine:
    def __init__(self, fetcher: OceanDataFetcher):
        self.fetcher = fetcher
        self.particles = []
        self.running = False
        self.max_steps = (MAX_YEARS * 365 * 24 * 3600) // DT_SECONDS

    def setup(self, lat0, lon0, n_particles=60, windage=0.025, diffusion=10.0):
        self.wind_data = self.fetcher.fetch_wind(lat0, lon0, 384)
        self.marine_data = self.fetcher.fetch_marine(lat0, lon0, 384)

        rng = np.random.default_rng(42)
        if n_particles == 1:
            self.particles = [DebrisParticle(lat0, lon0, windage, diffusion, seed=0, pid=1)]
        else:
            self.particles = [DebrisParticle(lat0 + rng.normal(0, 0.35), lon0 + rng.normal(0, 0.35),
                              windage*(0.7 + 0.6*rng.random()), diffusion, seed=i, pid=i+1) for i in range(n_particles)]

    def _interp_wrapped(self, arr, t_hours):
        idx = int(t_hours) % len(arr)
        next_idx = (idx + 1) % len(arr)
        frac = t_hours - int(t_hours)
        return float(arr[idx] * (1-frac) + arr[next_idx] * frac)

    def run(self, callback=None):
        self.running = True
        step_i = 0

        while self.running and step_i < self.max_steps:
            t_hr = (step_i * DT_SECONDS) / 3600.0

            wh = self._interp_wrapped(self.marine_data["wave_height"], t_hr)
            wd = self._interp_wrapped(self.marine_data["wave_dir"], t_hr)
            wp = self._interp_wrapped(self.marine_data["wave_period"], t_hr)
            uw = self._interp_wrapped(self.wind_data["u_wind"], t_hr)
            vw = self._interp_wrapped(self.wind_data["v_wind"], t_hr)

            all_beached = True
            for p in self.particles:
                if not p.beached:
                    all_beached = False
                    uc, vc = self.fetcher.get_current_at(p.lat, p.lon, t_hr)
                    p.step(DT_SECONDS, uc, vc, uw, vw, wh, wd, wp)
                    if p.beached: p.beach_time_hr = t_hr

            if callback and step_i % 100 == 0:
                callback(step_i, t_hr, all_beached)

            if all_beached: break
            step_i += 1

        self.running = False

    def get_beaching_stats(self):
        beached = [p for p in self.particles if p.beached]
        if not beached: return None
        b_times = [p.beach_time_hr for p in beached if p.beach_time_hr is not None]
        return {
            "count": len(beached), "fraction": len(beached)/len(self.particles),
            "lats": [p.beach_lat for p in beached], "lons": [p.beach_lon for p in beached],
            "mean_lat": float(np.mean([p.beach_lat for p in beached])),
            "mean_lon": float(np.mean([p.beach_lon for p in beached])),
            "mean_time_h": float(np.mean(b_times)) if b_times else 0.0,
            "particles": beached
        }

    def get_centroid_track(self):
        if not self.particles: return [], []
        max_len = max(len(p.track_lat) for p in self.particles)
        c_lats, c_lons = [], []
        for i in range(max_len):
            lats = [p.track_lat[i] for p in self.particles if len(p.track_lat) > i]
            lons = [p.track_lon[i] for p in self.particles if len(p.track_lon) > i]
            if lats: c_lats.append(np.mean(lats)); c_lons.append(np.mean(lons))
        return c_lats, c_lons

# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────
class OceanDebrisGUI:
    def __init__(self):
        self.fetcher = OceanDataFetcher()
        self.engine  = TrajectoryEngine(self.fetcher)

        self.start_lat, self.start_lon = 26.5, -78.0
        self.n_ensemble = 60
        self.windage = 0.025
        self.diffusion = 10.0
        self.is_oil = False
        
        self.sim_state = "READY" 
        self._is_panning = False
        self._pan_start_mouse = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None
        
        self._is_typing_particles = False
        self._typing_str = ""
        self._result_lines = []
        self._scroll_idx = 0
        self._max_visible_lines = 6
        
        self.show_wind = False
        self.show_cur = False
        self.show_wav = False
        
        self._selected_pid = None  # For filtering individual particle results
        
        self._layer_artists = []
        self._start_marker, self._start_ring = None, None
        self._scat, self._bscat = None, None
        self._centroid_line, self._mean_star, self._highlight_star = None, None, None
        self._tail_lines = []

        self._build_figure()
        self._draw_basemap()
        self._update_status("Ready — left-click water to place debris")
        self._draw_start_marker()
        self._update_map_title()

    def _build_figure(self):
        plt.rcParams.update({"figure.facecolor": C_BG, "axes.facecolor": C_BG, "text.color": C_TEXT, "font.family": "monospace", "font.size": 9})
        self.fig = plt.figure(figsize=(20, 11), facecolor=C_BG)
        self.fig.canvas.manager.set_window_title("🌊 Lagrangian Dispersion Simulator")

        # Adjusted GridSpec to give more breathing room to the control panel
        outer_gs = gridspec.GridSpec(1, 2, figure=self.fig, width_ratios=[3.0, 1.2], left=0.02, right=0.98, top=0.90, bottom=0.05, wspace=0.05)
        self.ax_map = self.fig.add_subplot(outer_gs[0])
        
        right_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[1], height_ratios=[8.5, 2.0, 1.5], hspace=0.08)
        self.ax_controls = self.fig.add_subplot(right_gs[0])
        self.ax_results = self.fig.add_subplot(right_gs[1])
        self.ax_physics = self.fig.add_subplot(right_gs[2])
        
        for ax in [self.ax_controls, self.ax_results, self.ax_physics]:
            ax.set_facecolor(C_PANEL)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            for sp in ax.spines.values():
                sp.set_color(C_COAST); sp.set_linewidth(1.5)

        self._setup_map_axes()
        self._setup_control_panel()
        self._setup_results_panel()
        self._setup_physics_panel()

        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        self.fig.text(0.50, 0.97, "⚓ LPDM DISPERSION TRACKER", ha="center", va="top", fontsize=16, color=C_ACCENT, fontweight="bold")
        online_str = "● LIVE DATA" if self.fetcher.online else "○ OFFLINE (synthetic model)"
        self.fig.text(0.98, 0.97, online_str, ha="right", va="top", fontsize=10, color=C_WIND if self.fetcher.online else C_WARN)

    def _setup_map_axes(self):
        self.ax_map.set_facecolor(C_OCEAN)
        self.ax_map.set_xlim(-180, 180); self.ax_map.set_ylim(-85, 85)
        for sp in self.ax_map.spines.values(): sp.set_color(C_COAST)
        self.ax_map.tick_params(colors=C_TEXT, labelsize=8)
        self.ax_map.set_xticks(range(-180, 181, 30)); self.ax_map.set_yticks(range(-90, 91, 30))
        self.ax_map.grid(color=C_GRID, linewidth=0.5, linestyle="--", alpha=0.7)
        self.coord_annot = self.ax_map.annotate("", xy=(0,0), xytext=(10, 10), textcoords="offset points",
                                       bbox=dict(boxstyle="round", fc=C_PANEL, ec=C_ACCENT, alpha=0.9), color=C_TEXT, visible=False, zorder=20)

    def _update_map_title(self):
        if self.sim_state == "READY":
            self.ax_map.set_title("Left-Click water to place  |  Right-Drag to pan  |  Scroll to zoom", color=C_TEXT, fontsize=10, pad=15)
        else:
            self.ax_map.set_title("Left/Right-Drag to pan  |  Scroll to zoom  |  Click 'CLEAR MAP' to restart", color=C_WARN, fontsize=10, pad=15)
        self.fig.canvas.draw_idle()

    def _setup_control_panel(self):
        def lbl(y, text): self.ax_controls.text(0.05, y, text, color=C_ACCENT, fontsize=11, fontweight="bold", va="center")

        # Spreading out layout to reduce clustering
        lbl(0.96, "LOCATION")
        self.txt_coords = self.ax_controls.text(0.05, 0.91, f"LAT: {self.start_lat:+.3f}°\nLON: {self.start_lon:+.3f}°", color=C_TRAJ, fontsize=10, linespacing=1.6, va="top")

        lbl(0.84, "DEBRIS TYPE")
        self._debris_types = [
            ("🍶 Plastic Bottle", 0.025, 10.0), ("🛢 Drum", 0.010, 10.0), 
            ("🪣 Foam", 0.035, 15.0), ("🎣 Fishing Equipment", 0.018, 5.0), # Fixed to Fishing Equipment
            ("🛢️ Oil Slick", 0.030, 200.0) 
        ]
        self._sel_debris = 0
        self._debris_rects = []
        
        y_pos_list = np.linspace(0.79, 0.61, len(self._debris_types))
        for i, ((name, coeff, diff), y_pos) in enumerate(zip(self._debris_types, y_pos_list)):
            r = mpatches.FancyBboxPatch((0.05, y_pos - 0.016), 0.90, 0.032, boxstyle="round,pad=0.01", fc=C_ACCENT if i==0 else C_COAST, ec="none")
            self.ax_controls.add_patch(r)
            txt = self.ax_controls.text(0.10, y_pos, f"{name} (β={coeff:.3f})", color="#000" if i==0 else C_TEXT, fontsize=9, va="center")
            self._debris_rects.append((r, txt, y_pos))

        # ---- NEW: Windage Beta Slider ----
        lbl(0.55, "CUSTOM WINDAGE (β)")
        self.ax_slider = self.ax_controls.inset_axes([0.18, 0.49, 0.70, 0.025])
        self.ax_slider.set_facecolor(C_OCEAN)
        self.beta_slider = Slider(
            ax=self.ax_slider,
            label='β ',
            valmin=0.0,
            valmax=0.08,
            valinit=self.windage,
            valfmt='%0.3f',
            color=C_ACCENT
        )
        self.beta_slider.label.set_color(C_TEXT)
        self.beta_slider.valtext.set_color(C_TEXT)
        for spine in self.ax_slider.spines.values(): spine.set_edgecolor(C_COAST)
        self.beta_slider.on_changed(self._on_slider_change)

        lbl(0.43, "PLACEMENT")
        self._btn_random = mpatches.FancyBboxPatch((0.05, 0.355), 0.90, 0.035, boxstyle="round,pad=0.01", fc="#1e4080", ec=C_ACCENT)
        self.ax_controls.add_patch(self._btn_random)
        self.ax_controls.text(0.50, 0.373, "🎲  RANDOM OCEAN PLACEMENT", color=C_TRAJ, fontsize=9, fontweight="bold", ha="center", va="center")

        lbl(0.29, "PARTICLES (Click to type)")
        self._btn_minus = mpatches.FancyBboxPatch((0.05, 0.20), 0.20, 0.045, boxstyle="round,pad=0.01", fc=C_COAST, ec="none")
        self.ax_controls.add_patch(self._btn_minus)
        self.ax_controls.text(0.15, 0.22, "-", color="white", fontsize=16, fontweight="bold", ha="center", va="center")
        
        self._rect_num = mpatches.FancyBboxPatch((0.30, 0.20), 0.40, 0.045, boxstyle="round,pad=0.01", fc=C_OCEAN, ec=C_COAST)
        self.ax_controls.add_patch(self._rect_num)
        self.txt_count = self.ax_controls.text(0.5, 0.22, f"{self.n_ensemble}", color=C_TEXT, fontsize=12, fontweight="bold", ha="center", va="center")
        
        self._btn_plus = mpatches.FancyBboxPatch((0.75, 0.20), 0.20, 0.045, boxstyle="round,pad=0.01", fc=C_COAST, ec="none")
        self.ax_controls.add_patch(self._btn_plus)
        self.ax_controls.text(0.85, 0.22, "+", color="white", fontsize=14, fontweight="bold", ha="center", va="center")

        self._btn_run = mpatches.FancyBboxPatch((0.05, 0.10), 0.90, 0.050, boxstyle="round,pad=0.01", fc="#2563eb", ec=C_ACCENT)
        self.ax_controls.add_patch(self._btn_run)
        self.ax_controls.text(0.5, 0.125, "▶ TRACK DISPERSION", color="white", fontsize=11, fontweight="bold", ha="center", va="center")

        self._btn_clear = mpatches.FancyBboxPatch((0.05, 0.02), 0.90, 0.040, boxstyle="round,pad=0.01", fc="#475569", ec="none")
        self.ax_controls.add_patch(self._btn_clear)
        self.ax_controls.text(0.5, 0.04, "✖ CLEAR MAP", color=C_TEXT, fontsize=10, fontweight="bold", ha="center", va="center")

    def _setup_results_panel(self):
        self.ax_results.text(0.05, 0.88, "BEACHING RESULTS", color=C_ACCENT, fontsize=10, fontweight="bold", va="center")
        self.status_txt = self.ax_results.text(0.05, 0.70, "Ready. Click Track.", color=C_WIND, fontsize=9, va="top", linespacing=1.4)
        self.result_txt = self.ax_results.text(0.05, 0.50, "Awaiting simulation...", color=C_TEXT, fontsize=9, va="top", linespacing=1.6, clip_on=True)
        
        self._btn_scroll_up = mpatches.FancyBboxPatch((0.85, 0.40), 0.10, 0.15, boxstyle="round,pad=0.01", fc=C_COAST, ec="none")
        self.ax_results.add_patch(self._btn_scroll_up)
        self.ax_results.text(0.90, 0.475, "▲", color="white", fontsize=10, ha="center", va="center")

        self._btn_scroll_dn = mpatches.FancyBboxPatch((0.85, 0.15), 0.10, 0.15, boxstyle="round,pad=0.01", fc=C_COAST, ec="none")
        self.ax_results.add_patch(self._btn_scroll_dn)
        self.ax_results.text(0.90, 0.225, "▼", color="white", fontsize=10, ha="center", va="center")

    def _setup_physics_panel(self):
        self.ax_physics.text(0.05, 0.82, "LPDM PHYSICS MODEL", color=C_ACCENT, fontsize=10, fontweight="bold")
        eq_text = "V = V_cur + V_leeway + V_stokes + V_diff\n\nV_cur   = Geostrophic currents\nV_leeway= Wind advection w/ Ekman angle\nV_stokes= Wave-induced orbital transport\nV_diff  = √(2K/dt) * N(0,1) Random Walk"
        self.ax_physics.text(0.05, 0.40, eq_text, color=C_TEXT, fontsize=8, va="center", linespacing=1.5)

    def _draw_basemap(self):
        ax = self.ax_map
        if len(GLOBAL_LAND_MASKER.paths) > 0:
            patches = [MplPolygon(path.vertices, closed=True) for path in GLOBAL_LAND_MASKER.paths]
            pc = PatchCollection(patches, facecolor=C_LAND, edgecolor=C_COAST, linewidth=0.5, zorder=1, antialiased=True)
            ax.add_collection(pc)
        else:
            land_img = np.where(GLOBAL_LAND_MASKER.coarse_mask[:-1, :-1], 0.18, 0.0)
            land_img = gaussian_filter(land_img.astype(float), sigma=0.5)
            ax.imshow(land_img, extent=[-180, 180, -85, 85], origin="lower", cmap=LinearSegmentedColormap.from_list("land", [C_OCEAN, C_LAND, "#2a3a50"]), aspect="auto", zorder=1, alpha=0.9)
        self.fig.canvas.draw_idle()

    def _on_slider_change(self, val):
        self.windage = val
        # If user manually slides, unhighlight presets visually so they know they are using a custom beta
        if self._sel_debris != -1:
            self._sel_debris = -1
            self.is_oil = False # Default back to generic physics for arbitrary beta
            for r_other, txt_other, _ in self._debris_rects:
                r_other.set_facecolor(C_COAST)
                txt_other.set_color(C_TEXT)
        self.fig.canvas.draw_idle()

    # -- Unified Interaction Handlers --
    def _on_mouse_press(self, event):
        if self._is_typing_particles: self._finalize_typing()

        if event.inaxes == self.ax_controls:
            self._handle_controls_click(event.xdata, event.ydata)
            return
        elif event.inaxes == self.ax_results:
            self._handle_results_click(event.xdata, event.ydata)
            return
            
        if event.inaxes == self.ax_map:
            if event.button == 1: 
                if self.sim_state == "READY":
                    self.coord_annot.set_visible(False)
                    if GLOBAL_LAND_MASKER.is_land(event.ydata, event.xdata):
                        self._update_status("⚠ Cannot place debris on land.")
                        return
                    self.start_lat, self.start_lon = event.ydata, event.xdata
                    self.txt_coords.set_text(f"LAT: {self.start_lat:+.3f}°\nLON: {self.start_lon:+.3f}°")
                    self._draw_start_marker()
                    self._update_status("Debris placed ✓")
                else:
                    self._start_pan(event)
            elif event.button in [2, 3]: 
                self._start_pan(event)

    def _on_key_press(self, event):
        if self._is_typing_particles:
            if event.key.isdigit():
                if len(self._typing_str) < 5:
                    self._typing_str += event.key
                    self.txt_count.set_text(self._typing_str + "_")
            elif event.key == "backspace":
                self._typing_str = self._typing_str[:-1]
                self.txt_count.set_text(self._typing_str + "_")
            elif event.key in ["enter", "return"]:
                self._finalize_typing()
            self.fig.canvas.draw_idle()

    def _finalize_typing(self):
        self._is_typing_particles = False
        self._rect_num.set_edgecolor(C_COAST)
        self.txt_count.set_color(C_TEXT)
        if self._typing_str:
            self.n_ensemble = max(1, min(10000, int(self._typing_str)))
        self.txt_count.set_text(str(self.n_ensemble))
        self.fig.canvas.draw_idle()

    def _start_pan(self, event):
        self._is_panning = True
        self._pan_start_mouse = (event.x, event.y)
        self._pan_start_xlim = self.ax_map.get_xlim()
        self._pan_start_ylim = self.ax_map.get_ylim()

    def _on_mouse_motion(self, event):
        if self._is_panning and event.inaxes == self.ax_map and self._pan_start_mouse:
            dx = event.x - self._pan_start_mouse[0]
            dy = event.y - self._pan_start_mouse[1]
            bbox = self.ax_map.bbox
            x_ratio = (self._pan_start_xlim[1] - self._pan_start_xlim[0]) / bbox.width
            y_ratio = (self._pan_start_ylim[1] - self._pan_start_ylim[0]) / bbox.height
            self.ax_map.set_xlim(self._pan_start_xlim[0] - dx * x_ratio, self._pan_start_xlim[1] - dx * x_ratio)
            self.ax_map.set_ylim(self._pan_start_ylim[0] - dy * y_ratio, self._pan_start_ylim[1] - dy * y_ratio)
            self.fig.canvas.draw_idle()

    def _on_mouse_release(self, event): self._is_panning = False

    def _on_scroll(self, event):
        if event.inaxes == self.ax_map:
            scale_factor = 1.2 if event.step < 0 else (1 / 1.2)
            xlim, ylim = self.ax_map.get_xlim(), self.ax_map.get_ylim()
            if event.xdata is None or event.ydata is None: return
            nw = (xlim[1] - xlim[0]) * scale_factor
            nh = (ylim[1] - ylim[0]) * scale_factor
            rx = (xlim[1] - event.xdata) / (xlim[1] - xlim[0])
            ry = (ylim[1] - event.ydata) / (ylim[1] - ylim[0])
            self.ax_map.set_xlim([event.xdata - nw * (1 - rx), event.xdata + nw * rx])
            self.ax_map.set_ylim([event.ydata - nh * (1 - ry), event.ydata + nh * ry])
            self.fig.canvas.draw_idle()
        elif event.inaxes == self.ax_results:
            if not self._result_lines: return
            if event.step > 0: self._scroll_idx = max(0, self._scroll_idx - 1)
            else: self._scroll_idx = min(self._scroll_idx + 1, max(0, len(self._result_lines) - self._max_visible_lines))
            self._update_result_view()

    def _handle_controls_click(self, x, y):
        if 0.05 <= x <= 0.95 and 0.10 <= y <= 0.15: self._run_simulation()
        elif 0.05 <= x <= 0.95 and 0.02 <= y <= 0.06: self._clear_all()
        # Random placement button
        elif 0.05 <= x <= 0.95 and 0.34 <= y <= 0.39: self._place_random_ocean()
        # Particle adjusters
        elif 0.05 <= x <= 0.25 and 0.19 <= y <= 0.25:
            if self.n_ensemble <= 10: self.n_ensemble = max(1, self.n_ensemble - 1)
            elif self.n_ensemble <= 50: self.n_ensemble -= 10
            else: self.n_ensemble -= 50
            self.txt_count.set_text(f"{self.n_ensemble}"); self.fig.canvas.draw_idle()
        elif 0.75 <= x <= 0.95 and 0.19 <= y <= 0.25:
            if self.n_ensemble < 10: self.n_ensemble += 1
            elif self.n_ensemble < 50: self.n_ensemble += 10
            elif self.n_ensemble < 500: self.n_ensemble += 50
            self.txt_count.set_text(f"{self.n_ensemble}"); self.fig.canvas.draw_idle()
        elif 0.30 <= x <= 0.70 and 0.19 <= y <= 0.25:
            self._is_typing_particles = True
            self._typing_str = ""
            self._rect_num.set_edgecolor(C_ACCENT)
            self.txt_count.set_color(C_WARN); self.txt_count.set_text("_")
            self.fig.canvas.draw_idle()
        else:
            for i, (r, txt, y_center) in enumerate(self._debris_rects):
                if 0.05 <= x <= 0.95 and (y_center - 0.016) <= y <= (y_center + 0.016):
                    self._sel_debris = i
                    self.windage = self._debris_types[i][1]
                    self.diffusion = self._debris_types[i][2]
                    self.is_oil = ("Oil" in self._debris_types[i][0])
                    
                    for j, (r_other, txt_other, _) in enumerate(self._debris_rects):
                        r_other.set_facecolor(C_ACCENT if j == i else C_COAST)
                        txt_other.set_color("#000" if j == i else C_TEXT)
                    
                    # Update slider value without triggering visual removal of preset selection
                    self.beta_slider.eventson = False 
                    self.beta_slider.set_val(self.windage)
                    self.beta_slider.eventson = True
                    
                    self.fig.canvas.draw_idle()
                    break

    def _handle_results_click(self, x, y):
        if 0.85 <= x <= 0.95 and 0.40 <= y <= 0.55:
            if self._result_lines:
                self._scroll_idx = max(0, self._scroll_idx - 1); self._update_result_view()
        elif 0.85 <= x <= 0.95 and 0.15 <= y <= 0.30:
            if self._result_lines:
                self._scroll_idx = min(self._scroll_idx + 1, max(0, len(self._result_lines) - self._max_visible_lines)); self._update_result_view()
        else:
            # Check if clicking a particle result line
            if not self._result_lines: return
            # Determine which visible line was clicked (approximate by y)
            header_lines = 5  # "100% WASHED UP", mean time, mean loc, blank, separator
            visible_start = self._scroll_idx
            # Each line takes ~0.08 of height in results panel, first line at ~0.50
            line_height = 0.095
            start_y = 0.50
            for i_vis in range(self._max_visible_lines):
                line_y_top = start_y - i_vis * line_height
                line_y_bot = line_y_top - line_height
                if line_y_bot <= y <= line_y_top:
                    abs_idx = visible_start + i_vis
                    if abs_idx < len(self._result_lines):
                        line = self._result_lines[abs_idx]
                        if line.startswith("#") and "|" in line:
                            try:
                                pid_str = line.split("|")[0].replace("#", "").strip()
                                pid = int(pid_str)
                                self._filter_particle(pid)
                            except: pass
                    break

    def _draw_start_marker(self):
        for attr in ("_start_marker", "_start_ring"):
            m = getattr(self, attr, None)
            if m:
                try: m.remove()
                except Exception: pass
            setattr(self, attr, None)

        self._start_marker = self.ax_map.plot(self.start_lon, self.start_lat, "o", ms=8, color=C_PARTICLE, mew=1.5, mec="white", zorder=10)[0]
        self._start_ring = self.ax_map.plot(self.start_lon, self.start_lat, "o", ms=20, color=C_PARTICLE, alpha=0.3, zorder=9)[0]
        self.fig.canvas.draw_idle()

    def _run_simulation(self):
        if self.sim_state == "RUNNING": return
        self.sim_state = "RUNNING"
        self._update_map_title()
        
        self._clear_trajectories()
        self._result_lines = []
        self._update_result_view()
        self.coord_annot.set_visible(False)
        
        sim_n = self.n_ensemble * 3 if self.is_oil else self.n_ensemble
        
        self._update_status(f"Tracking dispersion ({sim_n} parcels)...")
        self.engine.setup(self.start_lat, self.start_lon, sim_n, self.windage, self.diffusion)
        self._frame_data, self._last_frame = [], 0
        
        tail_lw = 0.2 if self.is_oil else 0.8
        tail_alpha = 0.1 if self.is_oil else 0.3
        self._tail_lines = [self.ax_map.plot([], [], color=C_TRAJ, lw=tail_lw, alpha=tail_alpha, zorder=11)[0] for _ in range(sim_n)]

        self._sim_thread = threading.Thread(target=self.engine.run, kwargs={"callback": self._sim_callback}, daemon=True)
        self._sim_thread.start()
        
        self._poll_timer = self.fig.canvas.new_timer(interval=100)
        self._poll_timer.add_callback(self._poll_update)
        self._poll_timer.start()

    def _sim_callback(self, step, t_hr, all_beached):
        positions = [(p.lat, p.lon) for p in self.engine.particles if not p.beached]
        beached = [(p.beach_lat, p.beach_lon) for p in self.engine.particles if p.beached]
        self._frame_data.append((t_hr, positions, beached))

    def _poll_update(self):
        while self._last_frame < len(self._frame_data):
            t_hr, positions, beached = self._frame_data[self._last_frame]
            
            if self._scat: 
                try: self._scat.remove()
                except Exception: pass
            if self._bscat: 
                try: self._bscat.remove()
                except Exception: pass
            
            p_size = 40 if self.is_oil else 8
            p_alpha = 0.2 if self.is_oil else 0.6
            p_color = "#333333" if self.is_oil else C_PARTICLE
            
            self._scat = self.ax_map.scatter([p[1] for p in positions], [p[0] for p in positions], s=p_size, c=p_color, alpha=p_alpha, edgecolors="none", zorder=12) if positions else None
            self._bscat = self.ax_map.scatter([b[1] for b in beached], [b[0] for b in beached], s=25, c=C_BEACH, marker="x", zorder=13) if beached else None

            days, hrs = int(t_hr // 24), int(t_hr % 24)
            yrs = days // 365
            time_str = f"{yrs}y {days%365}d {hrs:02d}h" if yrs > 0 else f"{days}d {hrs:02d}h"
            self.status_txt.set_text(f"Drifting: {time_str}\n{len(positions)} drifting | {len(beached)} washed up")
            self._last_frame += 1

        if self._last_frame > 0 and self._tail_lines:
            for i, p in enumerate(self.engine.particles):
                idx = min(self._last_frame * 100, len(p.track_lat))
                self._tail_lines[i].set_data(p.track_lon[:idx], p.track_lat[:idx])

        if not self.engine.running and self._last_frame >= len(self._frame_data):
            self._poll_timer.stop()
            self.sim_state = "DONE"
            self._update_map_title()
            self._on_simulation_done()
        self.fig.canvas.draw_idle()

    def _on_simulation_done(self):
        self._draw_final_trajectories()
        self._show_beaching_results()
        self.status_txt.set_text("✓ All parcels washed up.")
        self.fig.canvas.draw_idle()

    def _draw_final_trajectories(self):
        c_lats, c_lons = self.engine.get_centroid_track()
        if len(c_lats) > 1 and not self.is_oil:
            self._centroid_line = self.ax_map.plot(c_lons, c_lats, color=C_TRAJ, lw=2.0, alpha=0.9, zorder=15, solid_capstyle="round", path_effects=[pe.Stroke(linewidth=3, foreground="black", alpha=0.5), pe.Normal()])[0]
        self.fig.canvas.draw_idle()

    def _show_beaching_results(self):
        stats = self.engine.get_beaching_stats()
        if not stats: return
        mean_t = stats.get("mean_time_h", 0.0)
        time_str = f"{int(mean_t // (24*365))} Yrs, {int((mean_t/24)%365)} Days" if mean_t else "Immediately"
        
        lines = ["100% WASHED UP", f"MEAN TIME: {time_str}", f"MEAN LOC:  {stats['mean_lat']:+.2f}°, {stats['mean_lon']:+.2f}°", " ", "--- CLICK A PARCEL TO HIGHLIGHT ---"]
        for p in stats.get("particles", []): lines.append(f"#{p.pid:03d} | LAT: {p.beach_lat:+.2f}° | LON: {p.beach_lon:+.2f}°")
            
        self._result_lines = lines; self._scroll_idx = 0; self._update_result_view()

    def _update_result_view(self):
        if not self._result_lines:
            self.result_txt.set_text("Awaiting simulation..."); self.result_txt.set_color(C_TEXT)
            return
        visible = self._result_lines[self._scroll_idx : self._scroll_idx + self._max_visible_lines]
        
        # Mark selected particle line with indicator
        if self._selected_pid is not None:
            marked = []
            for line in visible:
                if line.startswith(f"#{self._selected_pid:03d}"):
                    marked.append(f"► {line} ◄")
                else:
                    marked.append(line)
            self.result_txt.set_text("\n".join(marked))
            self.result_txt.set_color("#ffdd00")
        else:
            self.result_txt.set_text("\n".join(visible))
            self.result_txt.set_color(C_BEACH)
        self.fig.canvas.draw_idle()

    def _filter_particle(self, pid):
        """Highlight one particle's trajectory and beaching location, dim all others."""
        if not self.engine.particles: return
        
        # Toggle: if same pid clicked again, clear filter
        if self._selected_pid == pid:
            self._selected_pid = None
            self._restore_all_trajectories()
            self._update_result_view()
            return
        
        self._selected_pid = pid
        target = next((p for p in self.engine.particles if p.pid == pid), None)
        if not target: return
        
        # Dim all tail lines
        for i, p in enumerate(self.engine.particles):
            if i < len(self._tail_lines):
                if p.pid == pid:
                    self._tail_lines[i].set_alpha(0.9)
                    self._tail_lines[i].set_linewidth(2.0)
                    self._tail_lines[i].set_color("#ffffff")
                else:
                    self._tail_lines[i].set_alpha(0.05)
                    self._tail_lines[i].set_linewidth(0.3)
        
        # Highlight the beached location
        for attr in ("_highlight_star",):
            obj = getattr(self, attr, None)
            if obj:
                try: obj.remove()
                except: pass
        
        if target.beached:
            self._highlight_star = self.ax_map.plot(
                target.beach_lon, target.beach_lat, "*", ms=22, color="#ffdd00",
                mew=2, mec="white", zorder=20
            )[0]
            # Pan map to show the beaching location
            cur_xlim = self.ax_map.get_xlim()
            cur_ylim = self.ax_map.get_ylim()
            xrange = cur_xlim[1] - cur_xlim[0]
            yrange = cur_ylim[1] - cur_ylim[0]
            self.ax_map.set_xlim(target.beach_lon - xrange/2, target.beach_lon + xrange/2)
            self.ax_map.set_ylim(target.beach_lat - yrange/2, target.beach_lat + yrange/2)
        
        # Update result text to highlight selected line
        self._update_result_view()
        self.fig.canvas.draw_idle()

    def _restore_all_trajectories(self):
        """Restore all trajectories to normal display."""
        tail_lw = 0.2 if self.is_oil else 0.8
        tail_alpha = 0.1 if self.is_oil else 0.3
        for line in self._tail_lines:
            line.set_alpha(tail_alpha)
            line.set_linewidth(tail_lw)
            line.set_color(C_TRAJ)
        for attr in ("_highlight_star",):
            obj = getattr(self, attr, None)
            if obj:
                try: obj.remove()
                except: pass
            setattr(self, attr, None)
        self.fig.canvas.draw_idle()

    def _place_random_ocean(self):
        """Pick a random ocean location from a pool of known open-ocean spots."""
        if self.sim_state == "RUNNING": return
        ocean_zones = [
            # North Atlantic
            (25, 50, -75, -20),
            # South Atlantic
            (-45, -5, -40, 10),
            # North Pacific
            (20, 50, 160, -130),  # crosses antimeridian — handled below
            # South Pacific
            (-50, -10, -170, -80),
            # Indian Ocean
            (-40, 20, 55, 100),
            # Caribbean
            (10, 25, -85, -60),
        ]
        import random as _random
        rng = np.random.default_rng()
        for _ in range(200):  # try up to 200 times
            zone = ocean_zones[rng.integers(0, len(ocean_zones))]
            lat_min, lat_max, lon_min, lon_max = zone
            lat = rng.uniform(lat_min, lat_max)
            # Handle antimeridian crossing zone
            if lon_min > lon_max:
                lon = rng.uniform(lon_min, lon_min + (360 - lon_min + lon_max))
                if lon > 180: lon -= 360
            else:
                lon = rng.uniform(lon_min, lon_max)
            if not GLOBAL_LAND_MASKER.is_land(lat, lon):
                self.start_lat, self.start_lon = lat, lon
                self.txt_coords.set_text(f"LAT: {self.start_lat:+.3f}°\nLON: {self.start_lon:+.3f}°")
                self._draw_start_marker()
                # Zoom map to show placement region
                self.ax_map.set_xlim(lon - 40, lon + 40)
                self.ax_map.set_ylim(lat - 25, lat + 25)
                self._update_status(f"Random ocean placement ✓  ({lat:+.2f}°, {lon:+.2f}°)")
                self.fig.canvas.draw_idle()
                return
        self._update_status("⚠ Could not find open ocean — try again.")

    def _update_status(self, msg): 
        self.status_txt.set_text(msg); self.fig.canvas.draw_idle()

    def _clear_trajectories(self):
        self.coord_annot.set_visible(False)
        self._selected_pid = None
        for attr in ("_scat", "_bscat", "_centroid_line", "_mean_star", "_highlight_star"):
            obj = getattr(self, attr, None)
            if obj:
                try: obj.remove()
                except Exception: pass
            setattr(self, attr, None)
                
        if self._tail_lines:
            for line in self._tail_lines:
                try: line.remove()
                except Exception: pass
            self._tail_lines = []

    def _clear_all(self):
        self.engine.running = False; self.sim_state = "READY"
        self._update_map_title()
        self._clear_trajectories()
        self._result_lines = []; self._scroll_idx = 0; self._update_result_view()
        self._update_status("Cleared. Click map to place new debris.")
        self.fig.canvas.draw_idle()

    def show(self): plt.tight_layout(pad=0.5); plt.show()

if __name__ == "__main__":
    gui = OceanDebrisGUI()
    gui.show()
