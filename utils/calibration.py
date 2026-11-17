import hashlib
import os
import pickle
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from .video_utils import get_video_properties


def ensure_ultralytics_config_dir() -> None:
    """Ensure Ultralytics has a writable settings directory even inside sandboxed environments."""
    if os.environ.get("YOLO_CONFIG_DIR"):
        return
    appdata = os.environ.get("APPDATA", "")
    default_dir = os.path.join(appdata, "Ultralytics") if appdata else ""
    try:
        if default_dir:
            os.stat(os.path.join(default_dir, "settings.json"))
            return
    except OSError:
        pass

    local_cfg = os.path.abspath(os.path.join("stubs", ".ultralytics_cfg"))
    os.makedirs(local_cfg, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = local_cfg


def get_video_stub_dir(video_path: str, base_stub_dir: str = "stubs") -> str:
    """Return a unique, collision-free stub directory keyed by video stem and file/geometry signature."""
    props = get_video_properties(video_path)
    file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
    sig_raw = f"{file_size}_{props['width']}x{props['height']}_{props['frame_count']}"
    short_hash = hashlib.sha1(sig_raw.encode("utf-8")).hexdigest()[:6]
    stem = os.path.splitext(os.path.basename(video_path))[0]
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem).strip("_")
    stub_dir = os.path.join(base_stub_dir, f"{safe_stem}_{short_hash}")
    os.makedirs(stub_dir, exist_ok=True)
    return stub_dir


class FrameScaleContext:
    """
    Relative spatial and temporal scale context for a broadcast video.
    Replaces all fixed pixel and frame-count thresholds with dimensionless multiples
    of frame geometry, video FPS, and local perspective player height H_p(y).
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        median_player_h: float | None = None,
        slope_a: float = 0.0,
        intercept_b: float | None = None,
    ) -> None:
        self.frame_w = int(width)
        self.frame_h = int(height)
        self.frame_diag = float(np.hypot(self.frame_w, self.frame_h))
        self.fps = float(fps) if fps and fps > 1.0 else 25.0
        self.ui_scale = float(self.frame_h) / 1080.0

        default_hp = 0.052 * self.frame_h
        self.median_player_h = float(median_player_h) if median_player_h and median_player_h > 0 else default_hp
        self.slope_a = float(slope_a)
        self.intercept_b = float(intercept_b) if intercept_b is not None else self.median_player_h

    def calibrate_from_tracks(self, tracks: dict) -> "FrameScaleContext":
        """Fit global median player height H_p and perspective height curve H_p(y) = a*y + b via Huber regression."""
        ys: list[float] = []
        hs: list[float] = []

        min_valid_h = 0.016 * self.frame_h
        max_valid_h = 0.42 * self.frame_h

        for frame_dict in tracks.get("players", []):
            for pdata in frame_dict.values():
                x1, y1, x2, y2 = pdata["bbox"]
                bw = max(1e-3, float(x2 - x1))
                bh = float(y2 - y1)
                aspect = bh / bw
                if min_valid_h <= bh <= max_valid_h and 1.05 <= aspect <= 4.8:
                    ys.append(float(y2))
                    hs.append(bh)

        if len(hs) < 10:
            return self

        ys_arr = np.asarray(ys, dtype=np.float64)
        hs_arr = np.asarray(hs, dtype=np.float64)
        self.median_player_h = float(np.median(hs_arr))

        # Subsample up to 2500 points for fast Huber fit
        if len(hs_arr) > 2500:
            idx = np.linspace(0, len(hs_arr) - 1, 2500, dtype=np.int32)
            ys_fit, hs_fit = ys_arr[idx], hs_arr[idx]
        else:
            ys_fit, hs_fit = ys_arr, hs_arr

        # Broadcast perspective constraint: slope_a >= 0 (closer to bottom of frame -> larger player bbox)
        def residuals(params: np.ndarray) -> np.ndarray:
            return (params[0] * ys_fit + params[1]) - hs_fit

        res = least_squares(
            residuals,
            x0=np.array([0.04 * self.median_player_h / max(1.0, self.frame_h), 0.8 * self.median_player_h]),
            bounds=([0.0, 0.05 * self.median_player_h], [0.35, 2.5 * self.median_player_h]),
            loss="soft_l1",
            f_scale=max(2.0, 0.15 * self.median_player_h),
        )
        self.slope_a = float(res.x[0])
        self.intercept_b = float(res.x[1])
        return self

    def player_h_at(self, y: float) -> float:
        """Estimated standing player bounding-box height (in pixels) at vertical coordinate y."""
        est = self.slope_a * float(y) + self.intercept_b
        return float(np.clip(est, 0.42 * self.median_player_h, 2.35 * self.median_player_h))

    def ball_size_bounds_at(self, y: float) -> tuple[float, float]:
        """Return (min_px, max_px) valid ball bbox side length at vertical coordinate y."""
        hp = self.player_h_at(y)
        min_s = max(3.0, 0.062 * hp)
        max_s = min(0.11 * self.frame_h, max(min_s * 3.0, 0.62 * hp))
        return float(min_s), float(max_s)

    def frames_for_seconds(self, seconds: float, minimum: int = 1) -> int:
        """Convert physical duration in seconds to integer frame count at video FPS."""
        return max(minimum, int(round(seconds * self.fps)))

    def odd_kernel(self, diag_fraction: float, minimum: int = 3) -> int:
        """Compute an odd morphological kernel size scaled proportionally to frame diagonal."""
        k = max(minimum, int(round(diag_fraction * self.frame_diag)))
        return k if (k % 2 == 1) else (k + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "fps": self.fps,
            "median_player_h": self.median_player_h,
            "slope_a": self.slope_a,
            "intercept_b": self.intercept_b,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameScaleContext":
        return cls(
            width=data["frame_w"],
            height=data["frame_h"],
            fps=data["fps"],
            median_player_h=data.get("median_player_h"),
            slope_a=data.get("slope_a", 0.0),
            intercept_b=data.get("intercept_b"),
        )


class PitchColorModel:
    """
    Zero-hardcoded-visuals pitch turf color model learned dynamically per video in CIE Lab space.
    Models illumination-invariant chromaticity (a*, b*) via Mahalanobis distance so both sunlit
    and stadium-shadowed grass pass cleanly while yellow/green/white/red kits are never confused with turf.
    """

    def __init__(
        self,
        mean_ab: np.ndarray,
        inv_cov_ab: np.ndarray,
        mahal_sq_pitch: float,
        mahal_sq_strict: float,
        l_min: float,
        l_max: float,
        mean_bgr: tuple[int, int, int],
    ) -> None:
        self.mean_ab = np.asarray(mean_ab, dtype=np.float64)
        self.inv_cov_ab = np.asarray(inv_cov_ab, dtype=np.float64)
        self.mahal_sq_pitch = float(mahal_sq_pitch)
        self.mahal_sq_strict = float(mahal_sq_strict)
        self.l_min = float(l_min)
        self.l_max = float(l_max)
        self.mean_bgr = tuple(int(x) for x in mean_bgr)

    @classmethod
    def fit_from_video(
        cls,
        video_path: str,
        num_samples: int = 12,
        read_from_stub: bool = True,
        stub_path: str | None = None,
    ) -> "PitchColorModel":
        """Learn the video's turf distribution unsupervised from evenly spaced broadcast frames."""
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and data.get("version") == 1:
                return cls(**data["params"])

        props = get_video_properties(video_path)
        n_frames = max(1, props["frame_count"])
        sample_indices = np.linspace(0, max(0, n_frames - 1), min(num_samples, n_frames), dtype=np.int32)

        cap = cv2.VideoCapture(video_path)
        ab_samples: list[np.ndarray] = []
        l_samples: list[np.ndarray] = []
        bgr_samples: list[np.ndarray] = []

        work_w = 320
        work_h = max(120, int(round(work_w * props["height"] / max(1, props["width"]))))

        try:
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                small = cv2.resize(frame, (work_w, work_h), interpolation=cv2.INTER_AREA)
                # Central-lower broadcast region where the playing field dominates screen area
                y0, y1 = int(0.24 * work_h), int(0.88 * work_h)
                x0, x1 = int(0.08 * work_w), int(0.92 * work_w)
                roi_bgr = small[y0:y1, x0:x1]
                roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

                flat_lab = roi_lab.reshape(-1, 3).astype(np.float32)
                flat_bgr = roi_bgr.reshape(-1, 3).astype(np.float32)
                # Exclude pure black bars (e.g. privacy redaction boxes) and saturated white specular glare
                valid = (flat_lab[:, 0] > 12.0) & (flat_lab[:, 0] < 248.0)
                if np.sum(valid) > 100:
                    ab_samples.append(flat_lab[valid, 1:3])
                    l_samples.append(flat_lab[valid, 0])
                    bgr_samples.append(flat_bgr[valid])
        finally:
            cap.release()

        if not ab_samples:
            # Neutral fallback covariance if video could not be decoded
            model = cls(
                mean_ab=np.array([112.0, 152.0]),
                inv_cov_ab=np.diag([1.0 / 64.0, 1.0 / 100.0]),
                mahal_sq_pitch=14.0,
                mahal_sq_strict=8.5,
                l_min=20.0,
                l_max=235.0,
                mean_bgr=(45, 115, 65),
            )
            return model

        all_ab = np.vstack(ab_samples)
        all_l = np.concatenate(l_samples)
        all_bgr = np.vstack(bgr_samples)

        # Subsample up to 24,000 pixels for fast KMeans(k=3) in 2D (a*, b*) chromaticity space
        if len(all_ab) > 24000:
            sel = np.linspace(0, len(all_ab) - 1, 24000, dtype=np.int32)
            sub_ab, sub_l, sub_bgr = all_ab[sel], all_l[sel], all_bgr[sel]
        else:
            sub_ab, sub_l, sub_bgr = all_ab, all_l, all_bgr

        with threadpool_limits(limits=1):
            km = KMeans(n_clusters=3, n_init=10, random_state=42)
            lbls = km.fit_predict(sub_ab.astype(np.float64))

        # Largest population chromaticity mode in central-lower broadcast ROI is the playing turf
        counts = np.bincount(lbls, minlength=3)
        turf_cluster = int(np.argmax(counts))
        turf_mask = lbls == turf_cluster

        turf_ab = sub_ab[turf_mask].astype(np.float64)
        turf_l = sub_l[turf_mask].astype(np.float64)
        turf_bgr = sub_bgr[turf_mask].astype(np.float64)

        # Robust 2-pass estimation (trim top 8% Mahalanobis outliers from initial cluster)
        mu0 = np.median(turf_ab, axis=0)
        diff0 = turf_ab - mu0
        cov0 = np.cov(diff0, rowvar=False) + np.eye(2) * 3.5
        inv_cov0 = np.linalg.inv(cov0)
        d2_0 = np.einsum("ni,ij,nj->n", diff0, inv_cov0, diff0)

        core = d2_0 <= np.percentile(d2_0, 92.0)
        core_ab = turf_ab[core] if np.sum(core) >= 50 else turf_ab
        core_l = turf_l[core] if np.sum(core) >= 50 else turf_l
        core_bgr = turf_bgr[core] if np.sum(core) >= 50 else turf_bgr

        mean_ab = np.mean(core_ab, axis=0)
        diff = core_ab - mean_ab
        # Regularize covariance slightly so sunlit/shadow hue shifts remain well-conditioned
        cov_ab = np.cov(diff, rowvar=False) + np.eye(2) * 4.5
        inv_cov_ab = np.linalg.inv(cov_ab)
        d2_core = np.einsum("ni,ij,nj->n", diff, inv_cov_ab, diff)

        mahal_sq_pitch = float(np.clip(np.percentile(d2_core, 99.3) * 1.35, 10.5, 22.0))
        mahal_sq_strict = float(np.clip(np.percentile(d2_core, 96.5), 5.5, 12.5))

        l_min = float(max(12.0, np.percentile(core_l, 1.0) - 18.0))
        l_max = float(min(245.0, np.percentile(core_l, 99.0) + 14.0))
        med_bgr = np.median(core_bgr, axis=0)

        params = {
            "mean_ab": mean_ab,
            "inv_cov_ab": inv_cov_ab,
            "mahal_sq_pitch": mahal_sq_pitch,
            "mahal_sq_strict": mahal_sq_strict,
            "l_min": l_min,
            "l_max": l_max,
            "mean_bgr": (int(round(med_bgr[0])), int(round(med_bgr[1])), int(round(med_bgr[2]))),
        }
        if stub_path:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump({"version": 1, "params": params}, f)

        return cls(**params)

    def mahalanobis_sq_map(self, bgr_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mahal_sq_2d, L_channel_2d) for an arbitrary BGR image or crop."""
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]
        da = lab[:, :, 1] - float(self.mean_ab[0])
        db = lab[:, :, 2] - float(self.mean_ab[1])
        i00 = float(self.inv_cov_ab[0, 0])
        i01 = float(self.inv_cov_ab[0, 1] + self.inv_cov_ab[1, 0])
        i11 = float(self.inv_cov_ab[1, 1])
        d2 = i00 * (da * da) + i01 * (da * db) + i11 * (db * db)
        return d2, l_chan

    def is_turf_mask(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Boolean mask of turf background pixels inside a player torso crop (kit-safe!)."""
        d2, l_chan = self.mahalanobis_sq_map(bgr_crop)
        return (d2 <= self.mahal_sq_strict) & (l_chan >= self.l_min) & (l_chan <= self.l_max - 4.0)

    def compute_playable_pitch_mask(
        self,
        frame: np.ndarray,
        work_w: int = 480,
        erode_border_frac: float = 0.014,
    ) -> np.ndarray:
        """
        Compute a binary mask (at aspect-preserving work_w resolution) of the convex playable pitch,
        preserving actual screen borders while eroding internal stadium/signage edges proportionally.
        """
        fh, fw = frame.shape[:2]
        work_h = max(90, int(round(work_w * fh / max(1, fw))))
        diag_s = float(np.hypot(work_w, work_h))

        small = cv2.resize(frame, (work_w, work_h), interpolation=cv2.INTER_AREA)
        d2, l_chan = self.mahalanobis_sq_map(small)
        raw_turf = ((d2 <= self.mahal_sq_pitch) & (l_chan >= self.l_min) & (l_chan <= self.l_max)).astype(np.uint8) * 255

        k_open = max(3, int(round(0.016 * diag_s)) | 1)
        k_close = max(5, int(round(0.032 * diag_s)) | 1)
        turf = cv2.morphologyEx(raw_turf, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
        turf = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))

        contours, _ = cv2.findContours(turf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.full((work_h, work_w), 255, dtype=np.uint8)

        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        if erode_border_frac <= 0.0:
            mask = np.zeros((work_h, work_w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            return mask

        erode_r = max(1, int(round(erode_border_frac * diag_s)))
        pad = erode_r + 4
        padded = np.zeros((work_h + 2 * pad, work_w + 2 * pad), dtype=np.uint8)
        cv2.fillConvexPoly(padded, hull + np.array([[[pad, pad]]], dtype=np.int32), 255)

        inner = padded[pad : pad + work_h, pad : pad + work_w]
        if np.any(inner[0, :] > 0):
            padded[:pad, pad : pad + work_w] = inner[0:1, :]
        if np.any(inner[-1, :] > 0):
            padded[pad + work_h :, pad : pad + work_w] = inner[-1:, :]
        if np.any(inner[:, 0] > 0):
            padded[pad : pad + work_h, :pad] = inner[:, 0:1]
        if np.any(inner[:, -1] > 0):
            padded[pad : pad + work_h, pad + work_w :] = inner[:, -1:]

        k_erode = 2 * erode_r + 1
        eroded = cv2.erode(padded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode)))
        return eroded[pad : pad + work_h, pad : pad + work_w]

    @staticmethod
    def is_point_in_mask(mask_small: np.ndarray, x_full: float, y_full: float, orig_w: int, orig_h: int) -> bool:
        """Check if full-resolution coordinate (x_full, y_full) lies inside downsampled mask_small."""
        mh, mw = mask_small.shape[:2]
        sx = int(np.clip(round(x_full * mw / max(1, orig_w)), 0, mw - 1))
        sy = int(np.clip(round(y_full * mh / max(1, orig_h)), 0, mh - 1))
        return bool(mask_small[sy, sx] > 0)

    @staticmethod
    def mask_y_bounds_full(mask_small: np.ndarray, orig_h: int) -> tuple[int, int]:
        """Return full-resolution [y_min, y_max] vertical span of nonzero pitch mask rows."""
        mh = mask_small.shape[0]
        rows = np.where(np.any(mask_small > 0, axis=1))[0]
        if len(rows) == 0:
            return 0, orig_h
        y0 = max(0, int(np.floor(rows[0] * orig_h / mh)))
        y1 = min(orig_h, int(np.ceil((rows[-1] + 1) * orig_h / mh)))
        return y0, y1
