"""Tiny software renderer used to make the PNG previews of the trays.

Orthographic z-buffer rasteriser in plain numpy, so previews work without
OpenGL. Not fast, not pretty, but dependable: flat shading, a depth cue so
recesses read as darker, and dark lines along creases and silhouettes.
"""

from __future__ import annotations

import math

import numpy as np


def _rotation(view):
    """view = 'top' or (azimuth_deg, elevation_deg). Camera: x right, y up, z to viewer."""
    if view == "top":
        return np.eye(3)
    az, el = view
    a = math.radians(az)
    t = math.radians(-(90.0 - el))
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    return rx @ rz


def render(objs, view="top", ppm=10.0, ss=2, margin=3.0, bg=(1.0, 1.0, 1.0)):
    """Render objs = [(verts (n,3), tris (m,3), (r,g,b)), ...] and return an RGB image array."""
    rot = _rotation(view)
    tri_list, col_list = [], []
    for verts, tris, rgb in objs:
        cam = np.asarray(verts, dtype=float) @ rot.T
        tri_list.append(cam[np.asarray(tris, dtype=np.int64)])
        col_list.append(np.tile(np.asarray(rgb, dtype=float), (len(tris), 1)))
    T = np.concatenate(tri_list)
    C = np.concatenate(col_list)

    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(n, axis=1)
    keep = (ln > 1e-12) & (n[:, 2] > 1e-9)  # front-facing only
    T, C, n = T[keep], C[keep], n[keep] / ln[keep, None]

    light = np.array([-0.35, 0.50, 0.80])
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.clip(n @ light, 0.0, 1.0)
    C = C * shade[:, None]

    s = ppm * ss
    xmin, xmax = T[..., 0].min() - margin, T[..., 0].max() + margin
    ymin, ymax = T[..., 1].min() - margin, T[..., 1].max() + margin
    W = int(math.ceil((xmax - xmin) * s))
    Hh = int(math.ceil((ymax - ymin) * s))
    zbuf = np.full((Hh, W), -np.inf)
    img = np.empty((Hh, W, 3))
    img[:] = bg

    X = ((T[..., 0] - xmin) * s).tolist()
    Y = ((ymax - T[..., 1]) * s).tolist()
    Z = T[..., 2].tolist()
    cols = C.tolist()
    for i in range(len(X)):
        x0, x1, x2 = X[i]
        y0, y1, y2 = Y[i]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        ax = max(int(math.floor(min(x0, x1, x2))), 0)
        bx = min(int(math.ceil(max(x0, x1, x2))), W - 1)
        ay = max(int(math.floor(min(y0, y1, y2))), 0)
        by = min(int(math.ceil(max(y0, y1, y2))), Hh - 1)
        if bx < ax or by < ay:
            continue
        px = np.arange(ax, bx + 1) + 0.5
        py = (np.arange(ay, by + 1) + 0.5)[:, None]
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
        w2 = 1.0 - w0 - w1
        z0, z1, z2 = Z[i]
        z = w0 * z0 + w1 * z1 + w2 * z2
        sub = zbuf[ay:by + 1, ax:bx + 1]
        m = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7) & (z > sub)
        if m.any():
            sub[m] = z[m]
            img[ay:by + 1, ax:bx + 1][m] = cols[i]

    valid = np.isfinite(zbuf)
    if valid.any():
        zmin, zmax = zbuf[valid].min(), zbuf[valid].max()
        cue = np.where(valid, 0.70 + 0.30 * (zbuf - zmin) / max(zmax - zmin, 1e-9), 1.0)
        img *= cue[..., None]
        zz = np.where(valid, zbuf, zmin - 8.0)
        # second difference: fires on depth steps and creases, not on flat slopes
        d2x = np.abs(zz[:, 2:] - 2 * zz[:, 1:-1] + zz[:, :-2])
        d2y = np.abs(zz[2:, :] - 2 * zz[1:-1, :] + zz[:-2, :])
        edge = np.zeros_like(zz)
        edge[:, 1:-1] = d2x
        edge[1:-1, :] = np.maximum(edge[1:-1, :], d2y)
        edge = np.clip((edge - 0.12) / 0.25, 0.0, 1.0)
        img *= (1.0 - 0.6 * edge)[..., None]

    Hh2, W2 = Hh // ss * ss, W // ss * ss
    img = img[:Hh2, :W2].reshape(Hh2 // ss, ss, W2 // ss, ss, 3).mean(axis=(1, 3))
    return np.clip(img, 0.0, 1.0)


def hstack(images, gap=24, bg=1.0):
    """Place images side by side, top-aligned."""
    h = max(im.shape[0] for im in images)
    out = []
    for k, im in enumerate(images):
        pad = np.full((h, im.shape[1], 3), bg)
        pad[: im.shape[0]] = im
        out.append(pad)
        if k < len(images) - 1:
            out.append(np.full((h, gap, 3), bg))
    return np.concatenate(out, axis=1)


def vstack(images, gap=24, bg=1.0):
    """Stack images vertically, left-aligned."""
    w = max(im.shape[1] for im in images)
    out = []
    for k, im in enumerate(images):
        pad = np.full((im.shape[0], w, 3), bg)
        pad[:, : im.shape[1]] = im
        out.append(pad)
        if k < len(images) - 1:
            out.append(np.full((gap, w, 3), bg))
    return np.concatenate(out, axis=0)


def save(path, img):
    import matplotlib.image as mpimg

    mpimg.imsave(str(path), img)
