"""A pixel-integrated effective image model for bright planar apertures.

Effective intensity PSFs are a surrogate for the actinic optical system.
The fitted dimensions acquire a physical interpretation only after calibration.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def array_backend(name):
    if name not in {"cpu", "cuda", "auto"}:
        raise ValueError("backend must be 'cpu', 'cuda' or 'auto'")
    if name != "cpu":
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() > 0:
                return cp
        except (ImportError, OSError, RuntimeError):
            if name == "cuda":
                raise RuntimeError(
                    "CUDA fitting requires CuPy and a working CUDA device"
                ) from None
        if name == "cuda":
            raise RuntimeError("No CUDA device is available")
    return np


def to_host(xp, a):
    return a if xp is np else xp.asnumpy(a)


class ApertureLibrary:
    """Interpolate over object dimensions and continuous detector positions."""

    def __init__(self, shape, angle, width, height, sigma, xp):
        self.shape, self.xp = shape, xp
        # Native pixel box integration is performed before sampling. An odd
        # oversampling factor places each kernel center on the fine grid.
        self.scale = 5
        self.w = self._grid(width)
        self.h = self._grid(height) if shape == "ellipse" else np.array([height])
        self.sigma = sigma
        max_width = max(self.w[-1], self.h[-1])
        self.radius = int(np.ceil(0.72 * max_width + 4.5 * max(sigma) + 2))
        filters = ndimage
        if xp is not np:
            from cupyx.scipy import ndimage as filters
        dtype = xp.float64 if xp is np else xp.float32
        fine = (
            xp.arange(
                -self.radius * self.scale, self.radius * self.scale + 1, dtype=dtype
            )
            / self.scale
        )
        y, x = xp.meshgrid(fine, fine, indexing="ij")
        u = np.cos(angle) * x + np.sin(angle) * y
        v = -np.sin(angle) * x + np.cos(angle) * y
        widths = xp.asarray(np.repeat(self.w, len(self.h)), dtype=dtype)
        heights = (
            xp.asarray(np.tile(self.h, len(self.w)), dtype=dtype)
            if shape == "ellipse"
            else widths
        )
        data = xp.empty((len(widths), len(fine), len(fine)), dtype=xp.float32)
        # Filter many aperture sizes together on CUDA, keeping the temporary
        # footprint bounded. The leading dimension indexes independent images.
        batch = 1 if xp is np else max(1, 2**19 // len(fine) ** 2)
        for start in range(0, len(widths), batch):
            group = slice(start, start + batch)
            wi, hi = widths[group, None, None], heights[group, None, None]
            if shape == "square":
                inside = xp.maximum(abs(u) - wi / 2, abs(v) - hi / 2)
            else:
                radius = xp.hypot(2 * u / wi, 2 * v / hi)
                grad = xp.hypot(4 * u / wi**2, 4 * v / hi**2) / xp.maximum(radius, 1e-8)
                inside = (radius - 1) / xp.maximum(grad, 1 / xp.maximum(wi, hi))
            mask = xp.clip(0.5 - inside * self.scale, 0, 1).astype(xp.float32)
            blurred = filters.gaussian_filter(
                mask, (0, sigma[1] * self.scale, sigma[0] * self.scale), mode="constant"
            )
            data[group] = filters.uniform_filter(
                blurred, (1, self.scale, self.scale), mode="constant"
            )
        self.data = data.reshape(len(self.w), len(self.h), len(fine), len(fine))

    @staticmethod
    def _grid(value):
        lo, hi = max(0.6, value * 0.48), max(value * 1.72, 2.5)
        return np.linspace(lo, hi, 9)

    def _size_indices(self, values, grid):
        xp = self.xp
        step = grid[1] - grid[0]
        frac = xp.clip((values - grid[0]) / step, 0, len(grid) - 1 - 1e-5)[:, None]
        idx = frac.astype("int32")
        return idx, frac - idx, step

    def evaluate(self, x, y, width, height, derivatives=True):
        """Return native sampled template and optionally dx,dy,dwidth,dheight."""
        xp = self.xp
        if xp is not np:
            from .cuda_aperture import aperture_sampler

            return aperture_sampler(derivatives)(
                x,
                y,
                width[:, None],
                height[:, None],
                self.data,
                self.data.shape[-1],
                len(self.w),
                len(self.h),
                self.radius,
                self.scale,
                self.w[0],
                self.h[0],
                self.w[1] - self.w[0],
                self.h[1] - self.h[0] if len(self.h) > 1 else 1,
            )
        cx = xp.clip((x + self.radius) * self.scale, 0, self.data.shape[-1] - 1)
        cy = xp.clip((y + self.radius) * self.scale, 0, self.data.shape[-2] - 1)
        ix = xp.minimum(cx.astype("int32"), self.data.shape[-1] - 2)
        iy = xp.minimum(cy.astype("int32"), self.data.shape[-2] - 2)
        fx, fy = cx - ix, cy - iy
        wi, fw, dw = self._size_indices(width, self.w)
        if self.shape == "ellipse":
            hi, fh, dh = self._size_indices(height, self.h)
            h_nodes = [(hi, 1 - fh, -1 / dh), (hi + 1, fh, 1 / dh)]
        else:
            h_nodes = [(0, 1.0, 0.0)]
        image = xp.zeros_like(x)
        if derivatives:
            dx, dy, dwidth, dheight = (xp.zeros_like(x) for _ in range(4))
        for i, weight_w, diff_w in [(wi, 1 - fw, -1 / dw), (wi + 1, fw, 1 / dw)]:
            for j, weight_h, diff_h in h_nodes:
                a00 = self.data[i, j, iy, ix]
                a01 = self.data[i, j, iy, ix + 1]
                a10 = self.data[i, j, iy + 1, ix]
                a11 = self.data[i, j, iy + 1, ix + 1]
                val = (1 - fy) * ((1 - fx) * a00 + fx * a01) + fy * (
                    (1 - fx) * a10 + fx * a11
                )
                weight = weight_w * weight_h
                image += weight * val
                if derivatives:
                    dx -= (
                        self.scale
                        * weight
                        * ((1 - fy) * (a01 - a00) + fy * (a11 - a10))
                    )
                    dy -= (
                        self.scale
                        * weight
                        * ((1 - fx) * (a10 - a00) + fx * (a11 - a01))
                    )
                    dwidth += diff_w * weight_h * val
                    dheight += diff_h * weight_w * val
        if derivatives:
            return image, dx, dy, dwidth, dheight
        return image


def solve_stamps(
    library,
    obs,
    xoff,
    yoff,
    initial_w,
    initial_h,
    neighbors,
    steps,
    *,
    profile_only=False,
):
    """Fit native samples. Neighbor images stay at independent seed locations."""
    xp = library.xp
    obs, x, y = [xp.asarray(a, dtype=xp.float32) for a in (obs, xoff, yoff)]
    w, h = [xp.asarray(a, dtype=xp.float32).copy() for a in (initial_w, initial_h)]
    cx, cy = xp.zeros_like(w), xp.zeros_like(h)
    tail = xp.zeros_like(obs)
    for dx, dy, nw, nh, relative_contrast in neighbors:
        dx, dy, nw, nh, relative_contrast = [
            xp.asarray(a, dtype=xp.float32) for a in (dx, dy, nw, nh, relative_contrast)
        ]
        tail += relative_contrast[:, None] * library.evaluate(
            x - dx[:, None], y - dy[:, None], nw, nh, derivatives=False
        )
    # Scaling each patch is an affine reparameterization; model amplitude and
    # intercept still float. Geometry never depends on an absolute DN threshold.
    lo = xp.min(obs, axis=1)
    scale = xp.maximum(xp.max(obs, axis=1) - lo, 1e-8)
    z = (obs - lo[:, None]) / scale[:, None]
    amp, base = xp.ones_like(w), xp.zeros_like(w)
    ellipse = library.shape == "ellipse"
    eye = xp.eye(6 if ellipse else 5, dtype=xp.float32)
    w = xp.clip(w, library.w[0] + 0.001, library.w[-1] - 0.001)
    if ellipse:
        h = xp.clip(h, library.h[0] + 0.001, library.h[-1] - 0.001)
    for iteration in range(steps + 1):
        final_profile = profile_only and iteration == steps
        if final_profile:
            t = library.evaluate(
                x - cx[:, None], y - cy[:, None], w, h, derivatives=False
            )
        else:
            t, tx, ty, tw, th = library.evaluate(x - cx[:, None], y - cy[:, None], w, h)
        t += tail
        if iteration == 0:
            tmean, zmean = t.mean(axis=1), z.mean(axis=1)
            dt = t - tmean[:, None]
            amp = xp.clip(
                (dt * z).sum(axis=1) / xp.maximum((dt * dt).sum(axis=1), 1e-7), 0.05, 12
            )
            base = zmean - amp * tmean
        residual = z - base[:, None] - amp[:, None] * t
        if final_profile:
            rmse = xp.sqrt((residual**2).mean(axis=1))
            return to_host(xp, xp.stack((xp.median(rmse), xp.median(w))))
        columns = [
            xp.ones_like(t),
            t,
            amp[:, None] * tx,
            amp[:, None] * ty,
            amp[:, None] * tw,
        ]
        if ellipse:
            columns.append(amp[:, None] * th)
        jac = xp.stack(columns, axis=-1)
        noise = xp.maximum(0.015, 1.4826 * xp.median(abs(residual), axis=1))
        weights = xp.minimum(1, 2 * noise[:, None] / xp.maximum(abs(residual), 1e-7))
        transpose = (weights[:, :, None] * jac).transpose(0, 2, 1)
        gram = transpose @ jac
        diagonal = xp.maximum(xp.diagonal(gram, axis1=1, axis2=2), 1e-7)
        norm = xp.sqrt(diagonal)
        correlation = gram / (norm[:, :, None] * norm[:, None, :])
        if iteration == steps:
            eig = xp.linalg.eigvalsh(correlation)
            condition = eig[:, -1] / xp.maximum(eig[:, 0], 1e-9)
            rmse = xp.sqrt((residual**2).mean(axis=1))
            break
        rhs = (transpose @ residual[:, :, None]) / norm[:, :, None]
        delta = xp.linalg.solve(correlation + 3e-4 * eye[None], rhs)[:, :, 0] / norm
        base += xp.clip(delta[:, 0], -0.5, 0.5)
        amp = xp.clip(amp + xp.clip(delta[:, 1], -1, 1), 0.02, 12)
        cx = xp.clip(cx + xp.clip(delta[:, 2], -0.8, 0.8), -2, 2)
        cy = xp.clip(cy + xp.clip(delta[:, 3], -0.8, 0.8), -2, 2)
        w = xp.clip(w + xp.clip(delta[:, 4], -1.8, 1.8), library.w[0], library.w[-1])
        if ellipse:
            h = xp.clip(
                h + xp.clip(delta[:, 5], -1.8, 1.8), library.h[0], library.h[-1]
            )
        else:
            h = w.copy()
    boundary = (w < library.w[0] + 0.02) | (w > library.w[-1] - 0.02)
    if ellipse:
        boundary |= (h < library.h[0] + 0.02) | (h > library.h[-1] - 0.02)
    result = (cx, cy, w, h, rmse, condition, boundary, amp * scale, lo + base * scale)
    return tuple(to_host(xp, field) for field in result)
