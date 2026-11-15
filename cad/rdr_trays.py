#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["manifold3d>=3.0", "numpy", "matplotlib"]
# ///
"""Red Dust Rebellion player trays: parametric generator.

Run from the project root:

    uv run cad/rdr_trays.py              # models + previews
    uv run cad/rdr_trays.py --no-previews

Writes models/<faction>.3mf (tray, lid, lid dot and tray inlay on one plate),
models/lid-ams.3mf (lid with the dot as a separate part, for printing the dot
in with the AMS), models/stl/*.stl and previews/*.png.

Layout of the box recess (144 x 120 x 40 mm): two columns of trays side by
side, three high. Each tray is 71.5 x 119 x 13.4 mm including its sliding
lid, so six slots: five factions plus one plain spares tray. A low rim round
the top of each tray locates the tray stacked on it.

Everything a print might need tuning for (clearances, pocket sizes, lid fit)
is a constant in the PARAMETERS block below.

Coordinates: the tray's long side is X, its short side is Y, Z is up. The
lid slides off towards -X. Layouts are written in interior coordinates
(u, v), measured from the inside corner at the lid-entry end; numbers read
left to right along +u with the player sitting on the v = 0 side.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
from manifold3d import CrossSection, FillRule, JoinType, Manifold, OpType

# =============================================================================
# PARAMETERS (all mm)
# =============================================================================

# --- envelope: two trays side by side fill 144 x 120, three high fill 40.
# Heights throughout are multiples of the 0.2 mm layer so nothing gets rounded when sliced.
LX, LY, H = 119.0, 71.5, 13.4
FLOOR = 1.0
WALL_SIDE = 2.0        # long walls; thick because they carry the lid rails
WALL_FAR = 2.0         # closed end the lid tucks under
WALL_NEAR = 1.6        # low end the lid slides over
R_OUT = 1.5            # outside corner radius

# --- stacking: a rim round the top of each tray locates the tray above it
RIM_H = 0.6            # rim height above the tray top
RIM_LAND = 0.4         # flat width on top of the rim
STACK_CLR = 0.25       # horizontal gap between a rim and the tray sitting in it
CHAMFER = RIM_LAND + RIM_H + STACK_CLR   # 45 degree chamfer round the base that drops into the rim

# --- sliding lid (dovetail, 45 degrees, prints without support)
LID_T = 1.2            # lid thickness; lid top is flush with the tray top
WEB = 0.8              # wall left outside the lid edge
LID_CLR = 0.25         # horizontal gap per side between lid and rails
LID_BLUNT = 0.2        # width trimmed off the lid's feather edge
BUMP_H = 0.4           # detent bump that holds the lid shut
GROOVE_L, GROOVE_W, GROOVE_D = 28.0, 4.0, 0.8   # finger groove across each end of the lid
GROOVE_IN = 6.0        # groove centre, measured in from the end of the lid (clear of the click notches)

# --- wooden pieces as measured
HEX_F2F, HEX_H = 9.96, 10.0
DISC_D, DISC_T = 13.86, 4.04
CUBE = 8.05
CYL_D, CYL_H = 14.20, 10.33

# --- fits
P_HEX = 10.25          # centre-to-centre pitch of nested hexes
NEST_CLR = 0.15        # extra room around the whole nest
ETCH_W, ETCH_D = 0.8, 0.4   # honeycomb lines etched in the nest floor
DISC_POCKET = 14.4
DISC_PITCH = 15.2
CUBE_POCKET = 9.2      # generous: the wooden cubes vary noticeably in size
CUBE_PITCH = 10.0
WELL_W = 15.2          # storage wells for cylinders and spare discs
HEX_SPARE = 10.6       # single spare-hex pocket, flat to flat

# --- heights of the different floors (Z from the table)
Z_NEST_DECK = 5.0      # ledge around a hex nest
Z_CUBE_FLOOR = 3.0     # cubes stand on a raised floor so they sit proud
Z_CUBE_DECK = 5.6
Z_PLAT = 9.6           # plateau that carries the bases track
Z_DISC_FLOOR = 7.6     # discs lie here, 2 mm into the plateau
Z_BIN_WALL = 12.0      # full-height divider (just under the lid)
MARK_D = 0.6           # depth of recessed numbers and symbols
CARD_W, CARD_H = 9.0, 10.6   # card outline drawn round a track number

# --- faction inlays: plain shapes printed in the faction's colour
PILL_L, PILL_W, PILL_T = 36.0, 14.0, 1.6   # tile set into the tray
PILL_CLR = 0.1         # per side between the tile and its recess; on its own this is a loose drop-in fit
PILL_RIB = 0.1         # per side; how much wider than the recess the tile's grip ribs are (raise if still loose)
PILL_RIB_AT = 7.0      # the ribs sit this far either side of the middle of each long edge
PILL_RIB_Z = (0.4, 1.2)   # ribs start above the tile's lead-in chamfer and stop short of its visible face
POKE_D = 5.0           # hole under the tile to push it back out
DOT_D, DOT_T = 14.0, 0.6                   # round inlay set into the lid
DOT_CLR = 0.1          # per side

# --- derived
ZL = H - LID_T                         # underside of the lid
X0, Y0 = WALL_NEAR, WALL_SIDE          # interior origin
U = LX - WALL_NEAR - WALL_FAR          # interior length  (115.4)
V = LY - 2 * WALL_SIDE                 # interior width   (67.5)
HS = P_HEX * math.sqrt(3) / 2          # spacing between nest lines
RC = (P_HEX + NEST_CLR) / math.sqrt(3) # corner radius of a nest cell

SEG = 72

FONT_NUM = "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf"

# =============================================================================
# 2D / 3D helpers
# =============================================================================


def rect(x0, y0, x1, y1):
    return CrossSection.square((x1 - x0, y1 - y0)).translate((x0, y0))


def rrect(x0, y0, x1, y1, r):
    r = min(r, (x1 - x0) / 2 - 1e-3, (y1 - y0) / 2 - 1e-3)
    if r <= 0.01:
        return rect(x0, y0, x1, y1)
    return rect(x0 + r, y0 + r, x1 - r, y1 - r).offset(r, JoinType.Round, 2.0, 48)


def circle(d, seg=SEG):
    return CrossSection.circle(d / 2, seg)


def poly(pts):
    a = np.asarray(pts, dtype=float)
    area2 = np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1])
    return CrossSection([a if area2 > 0 else a[::-1]])


def hexagon(f2f, flat_axis):
    """Regular hexagon. flat_axis is the direction measured flat to flat ('u' or 'v')."""
    r = f2f / math.sqrt(3)
    a0 = 0.0 if flat_axis == "v" else 30.0
    return poly([(r * math.cos(math.radians(a0 + 60 * k)), r * math.sin(math.radians(a0 + 60 * k))) for k in range(6)])


def cs_union(items):
    items = list(items)
    return CrossSection.batch_boolean(items, OpType.Add) if items else CrossSection()


def m_union(items):
    items = list(items)
    return Manifold.batch_boolean(items, OpType.Add) if items else Manifold()


def prism(cs, z0, z1):
    return cs.extrude(z1 - z0).translate((0, 0, z0))


def at(cs, u, v):
    return cs.translate((u, v))


# --- text ---------------------------------------------------------------------

_font_prop = None


def _font():
    global _font_prop
    if _font_prop is None:
        from matplotlib.font_manager import FontProperties

        _font_prop = FontProperties(fname=FONT_NUM) if os.path.exists(FONT_NUM) else FontProperties(family="DejaVu Sans", weight="bold")
    return _font_prop


def text_cs(s, cap, max_w=None):
    """Text outline centred on the origin; cap = capital-letter height in mm."""
    from matplotlib.textpath import TextPath

    fp = _font()
    ref = TextPath((0, 0), "H", size=100, prop=fp).get_extents()
    k = cap / ref.height
    polys = [np.asarray(p)[:-1] for p in TextPath((0, 0), s, size=100, prop=fp).to_polygons() if len(p) >= 4]
    cs = CrossSection(polys, FillRule.EvenOdd)
    b = cs.bounds()
    cs = cs.translate((-(b[0] + b[2]) / 2, -(ref.y0 + ref.height / 2))).scale((k, k))
    w = (b[2] - b[0]) * k
    if max_w and w > max_w:
        cs = cs.scale((max_w / w, max_w / w))
    return cs


# =============================================================================
# Tray
# =============================================================================


class Tray:
    """Collects decks (solid regions), cuts (pockets) and ghost pieces in interior (u, v) coordinates."""

    def __init__(self, key):
        self.key = key
        self.decks = []      # (cross-section, top z)
        self.cuts = []       # manifolds
        self.pockets = []    # (name, cross-section) footprints for the layout check
        self.ghosts = []     # dict(kind, solid, top, retain)
        self.notes = []

    # --- building blocks ------------------------------------------------------

    def deck(self, u0, v0, u1, v1, z):
        self.decks.append((rect(u0, v0, u1, v1), z))

    def _ghost(self, kind, solid, top, retain):
        self.ghosts.append(dict(kind=kind, solid=solid, top=top, retain=retain))

    def g_hex(self, u, v, flat_axis, z=FLOOR, retain=None):
        retain = Z_NEST_DECK if retain is None else retain
        self._ghost("hex", prism(at(hexagon(HEX_F2F, flat_axis), u, v), z, z + HEX_H), z + HEX_H, retain)

    def g_disc(self, u, v, z, retain):
        self._ghost("disc", prism(at(circle(DISC_D, 40), u, v), z, z + DISC_T), z + DISC_T, retain)

    def g_cyl(self, u, v, retain, z=FLOOR):
        self._ghost("cyl", prism(at(circle(CYL_D, 40), u, v), z, z + CYL_H), z + CYL_H, retain)

    def g_cube(self, u, v):
        z, retain = Z_CUBE_FLOOR, Z_CUBE_DECK
        h = CUBE / 2
        self._ghost("cube", prism(rect(u - h, v - h, u + h, v + h), z, z + CUBE), z + CUBE, retain)

    def nest(self, name, u0, v0, lines, flat_axis):
        """Honeycomb nest. lines = cells per line; odd lines are shifted half a pitch.

        flat_axis 'v': lines are columns that advance along u, cells stack along v.
        flat_axis 'u': lines are rows that advance along v, cells sit side by side along u.
        Returns the cell centres.
        """
        cells = []
        half = (P_HEX + NEST_CLR) / 2
        for k, count in enumerate(lines):
            for j in range(count):
                along = half + j * P_HEX + (P_HEX / 2 if k % 2 else 0.0)
                across = RC + k * HS
                cells.append((u0 + across, v0 + along) if flat_axis == "v" else (u0 + along, v0 + across))
        cavity = cs_union(at(hexagon(P_HEX + NEST_CLR, flat_axis), u, v) for u, v in cells)
        self.cuts.append(prism(cavity, FLOOR, H + 1))
        self.pockets.append((name, cavity))
        rings = cs_union(at(hexagon(P_HEX + 0.02, flat_axis) - hexagon(P_HEX - ETCH_W, flat_axis), u, v) for u, v in cells)
        etch = rings ^ cavity.offset(-(ETCH_W / 2 + NEST_CLR / 2 + 0.05), JoinType.Miter, 2.0)
        self.cuts.append(prism(etch, FLOOR - ETCH_D, FLOOR + 0.1))
        for u, v in cells:
            self.g_hex(u, v, flat_axis)
        return cells

    def nest_extent(self, lines, flat_axis):
        """(du, dv) size of a nest's cavity."""
        across = (len(lines) - 1) * HS + 2 * RC
        along = max(c * P_HEX + (P_HEX / 2 if k % 2 else 0.0) for k, c in enumerate(lines)) + NEST_CLR
        return (across, along) if flat_axis == "v" else (along, across)

    def disc_spot(self, u, v, label, pocket=True, card=False):
        """One space of a bases track: a pocket with its number on the floor.

        pocket=False draws the space as a flat outline (for 0, which never holds a base).
        card=True frames the number in a card outline: that space lets the Church draw a card.
        """
        z = Z_DISC_FLOOR if pocket else Z_PLAT
        if pocket:
            self.cuts.append(prism(at(circle(DISC_POCKET), u, v), Z_DISC_FLOOR, H + 1))
            cham = Manifold.cylinder(0.2, DISC_POCKET / 2, DISC_POCKET / 2 + 0.2, SEG).translate((u, v, Z_PLAT - 0.2))
            self.cuts.append(cham)
            self.pockets.append((f"disc {label}", at(circle(DISC_POCKET), u, v)))
            self.g_disc(u, v, Z_DISC_FLOOR, Z_PLAT)
        marks = []
        if card:
            frame = rrect(-CARD_W / 2, -CARD_H / 2, CARD_W / 2, CARD_H / 2, 1.3)
            marks.append(frame - frame.offset(-0.8, JoinType.Round, 2.0, 24))
            marks.append(text_cs(label, 5.2, max_w=CARD_W - 3.0))
        else:
            if not pocket:
                marks.append(circle(DISC_POCKET) - circle(DISC_POCKET - 1.6))
            marks.append(text_cs(label, 6.4, max_w=9.6))
        self.cuts.append(prism(at(cs_union(marks), u, v), z - MARK_D, z + 0.1))

    def track(self, u_first, v_first, cols, rows, first=0, cards=(), zero_as_mark=True):
        """Grid of track spaces numbered in reading order; (u_first, v_first) is the top-left centre."""
        n = first
        for r in range(rows):
            for c in range(cols):
                u, v = u_first + c * DISC_PITCH, v_first - r * DISC_PITCH
                self.disc_spot(u, v, str(n), pocket=not (n == 0 and zero_as_mark), card=n in cards)
                n += 1

    def cube_grid(self, u0, v0, nu, nv, count):
        """nu x nv pockets with the lower-left pocket's corner at (u0, v0); ghosts for `count` cubes."""
        k = 0
        for j in range(nv - 1, -1, -1):
            for i in range(nu):
                u = u0 + CUBE_POCKET / 2 + i * CUBE_PITCH
                v = v0 + CUBE_POCKET / 2 + j * CUBE_PITCH
                sq = rrect(u - CUBE_POCKET / 2, v - CUBE_POCKET / 2, u + CUBE_POCKET / 2, v + CUBE_POCKET / 2, 0.6)
                self.cuts.append(prism(sq, Z_CUBE_FLOOR, H + 1))
                self.pockets.append((f"cube {i},{j}", sq))
                if k < count:
                    self.g_cube(u, v)
                k += 1

    def well(self, name, u0, v0, u1, v1, r=3.0):
        cs = rrect(u0, v0, u1, v1, r)
        self.cuts.append(prism(cs, FLOOR, H + 1))
        self.pockets.append((name, cs))

    def hex_pocket(self, u, v, flat_axis="u"):
        cs = at(hexagon(HEX_SPARE, flat_axis), u, v)
        self.cuts.append(prism(cs, FLOOR, H + 1))
        self.pockets.append(("spare hex", cs))
        self.g_hex(u, v, flat_axis, retain=Z_PLAT)

    def pill(self, u, v):
        """Recess for the faction inlay tile, with a poke-out hole underneath."""
        L, W = PILL_L + 2 * PILL_CLR, PILL_W + 2 * PILL_CLR
        cs = rrect(u - L / 2, v - W / 2, u + L / 2, v + W / 2, W / 2)
        self.cuts.append(prism(cs, Z_PLAT - PILL_T, H + 1))
        self.cuts.append(prism(at(circle(POKE_D, 32), u, v), -1, Z_PLAT))
        self.pockets.append(("inlay", cs))
        self.pill_at = (u, v)

    # --- assembly ---------------------------------------------------------------

    def build(self):
        outer = rrect(0, 0, LX, LY, R_OUT)
        body = Manifold.batch_hull([prism(outer.offset(-CHAMFER, JoinType.Round, 2.0, 48), 0, 0.02), prism(outer, CHAMFER, H)])
        body -= prism(rect(X0, Y0, X0 + U, Y0 + V), FLOOR, H + 1)
        loc = (X0, Y0, 0)
        if self.decks:
            body += m_union(prism(cs.offset(0.1, JoinType.Miter, 2.0), FLOOR - 0.3, z) for cs, z in self.decks).translate(loc)
        if self.cuts:
            body -= m_union(self.cuts).translate(loc)
        body -= lid_slot()
        body += detent_bumps()
        body += stack_rim()
        return body

    def ghost_solids(self):
        return [(g["kind"], g["solid"].translate((X0, Y0, 0))) for g in self.ghosts]


def lid_slot():
    """Dovetailed slot the lid runs in: open at the entry end, undercut on the other three sides."""
    big = rect(-3.0, -20, LX + 20, LY + 20)
    bot = rrect(-11, WEB, LX - WEB, LY - WEB, 1.0) ^ big
    top = rrect(-11, WALL_SIDE, LX - WALL_FAR, LY - WALL_SIDE, 0.3) ^ big
    slot = Manifold.batch_hull([prism(bot, ZL, ZL + 0.01), prism(top, H - 0.01, H)])
    # carry the opening straight up; hulling that in would steepen the 45 degree rail
    return slot + prism(top, H - 0.01, H + 1.0)


LID_END = LX - WEB - LID_CLR - LID_BLUNT   # x of the lid's far edge when closed; its near edge is at x = 0
LID_CX = LID_END / 2                       # centre of the closed lid


def lid_blank():
    """Plain lid in its closed position (tray coordinates).

    All four edges carry the 45 degree bevel and the shape is the same end for end,
    so the lid slides in either way round.
    """
    x1b, x1t = LX - WEB - LID_CLR, LX - WALL_FAR - LID_CLR   # far edge: wide underside, narrower top
    x0b = LID_END - x1b                                      # near edge mirrors the far edge
    x0t = x0b + (x1b - x1t)
    bot = rrect(x0b, WEB + LID_CLR, x1b, LY - WEB - LID_CLR, 1.0)
    top = rrect(x0t, WALL_SIDE + LID_CLR, x1t, LY - WALL_SIDE - LID_CLR, 0.3)
    lid = Manifold.batch_hull([prism(bot, ZL, ZL + 0.01), prism(top, H - 0.01, H)])
    e = WEB + LID_CLR + LID_BLUNT   # trim the feather edge of the dovetail so it prints cleanly
    return lid ^ prism(rrect(0, e, LID_END, LY - e, 0.8), ZL - 1, H + 1)


DETENT_X = LX - 12.0


def detent_bumps():
    out = []
    for y0, y1 in ((WEB - 0.1, WALL_SIDE), (LY - WALL_SIDE, LY - WEB + 0.1)):
        base = prism(rect(DETENT_X - 0.7, y0, DETENT_X + 0.7, y1), ZL - 0.3, ZL + 0.01)
        ridge = prism(rect(DETENT_X - 0.05, y0, DETENT_X + 0.05, y1), ZL + BUMP_H - 0.01, ZL + BUMP_H)
        out.append(Manifold.batch_hull([base, ridge]))
    return m_union(out)


def stack_rim():
    """Low rim round the top edge. The chamfered base of the tray above drops inside it."""
    outer = rrect(0, 0, LX, LY, R_OUT)
    inner = Manifold.batch_hull([
        prism(outer.offset(-(RIM_LAND + RIM_H), JoinType.Round, 2.0, 48), H - 0.02, H - 0.01),
        prism(outer.offset(-RIM_LAND, JoinType.Round, 2.0, 48), H + RIM_H, H + RIM_H + 0.01)])
    ring = prism(outer, H - 0.01, H + RIM_H) - inner
    # open at the entry end so the lid can slide out; the corners either side still locate the tray above
    return ring - prism(rect(-1, WALL_SIDE, CHAMFER + 0.5, LY - WALL_SIDE), H - 1, H + RIM_H + 1)


DOT_AT = (LID_CX, LY / 2)


def build_lid(recessed=True):
    """Lid in its closed position (tray coordinates): plain, with a round recess for the colour
    inlay and a finger groove at each end. recessed=False leaves the top face untouched."""
    lid = lid_blank()
    # notches that click over the detent bumps, at both ends so the lid works either way round
    for x in (DETENT_X, LID_END - DETENT_X):
        for y0, y1 in ((0.0, WALL_SIDE + 0.5), (LY - WALL_SIDE - 0.5, LY)):
            lid -= prism(rect(x - 1.0, y0, x + 1.0, y1), ZL - 0.1, ZL + BUMP_H + 0.2)
    if not recessed:
        return lid
    lid -= prism(at(circle(DOT_D + 2 * DOT_CLR), *DOT_AT), H - DOT_T, H + 1)
    # a finger groove across each end, square-walled so a fingertip or nail can push the lid either way
    for x in (GROOVE_IN, LID_END - GROOVE_IN):
        groove = rrect(x - GROOVE_W / 2, LY / 2 - GROOVE_L / 2, x + GROOVE_W / 2, LY / 2 + GROOVE_L / 2, GROOVE_W / 2)
        lid -= prism(groove, H - GROOVE_D, H + 1)
    return lid


def build_dot_fill():
    """Solid that exactly fills the lid's recess (tray coordinates), for printing the dot in with the AMS."""
    return prism(at(circle(DOT_D + 2 * DOT_CLR), *DOT_AT), H - DOT_T, H)


def build_dot():
    """Loose round inlay for the lid, sitting on z = 0; press or glue it into the recess."""
    return prism(circle(DOT_D), 0, DOT_T)


def build_tile(ribs=True):
    """Plain inlay tile for the tray, sitting on z = 0; that face goes down into the recess.

    Two low ribs on each long edge stand a little wider than the recess, so the tile has to
    be pressed in and then stays put. Only the ribs touch the recess wall, which keeps the
    fit forgiving of small differences between printers. They stop short of the top face,
    so the visible outline stays plain.
    """
    L, W = PILL_L, PILL_W
    base = rrect(-L / 2, -W / 2, L / 2, W / 2, W / 2)
    tile = Manifold.batch_hull([prism(base.offset(-0.3, JoinType.Round, 2.0, 48), 0, 0.02), prism(base, 0.3, PILL_T)])
    if not ribs:
        return tile
    reach = PILL_CLR + PILL_RIB          # how far a rib stands out from the edge of the tile
    z0, z1 = PILL_RIB_Z
    out = [tile]
    for x in (-PILL_RIB_AT, PILL_RIB_AT):
        for s in (-1, 1):
            # narrow crest, wide root, and a 45 degree ramp underneath to lead it into the recess
            root = prism(rect(x - 0.8, -0.3, x + 0.8, 0.0), z0, z1)
            crest = prism(rect(x - 0.2, reach - 0.01, x + 0.2, reach), z0 + reach, z1)
            rib = Manifold.batch_hull([root, crest]).translate((0, W / 2, 0))
            out.append(rib if s > 0 else rib.mirror((0, 1, 0)))
    return m_union(out)


# =============================================================================
# Faction layouts (interior coordinates, u along the long side, v along the short side)
# =============================================================================

HEADER_V = 46.6                      # v where the header strip starts on most trays
HEADER_C = (HEADER_V + V) / 2        # centre line of the header strip
TRACK_ROWS_V = (38.6, 23.4, 8.2)     # centre lines of three track rows under the header


def tray_red_dust():
    """30 Rebels (hex) + spare, 9 Bases + spare, 2 cylinders + spare."""
    t = Tray("red-dust")
    lines = [6, 6, 6, 6, 6]
    du, dv = t.nest_extent(lines, "v")
    edge = 0.4 + du + 0.8                       # where the nest zone ends
    t.deck(0, 0, edge, V, Z_NEST_DECK)
    t.nest("rebel nest", 0.4, (V - dv) / 2, lines, "v")

    chan = U - WELL_W                           # storage channel across the far end
    t.deck(edge, 0, chan, V, Z_PLAT)
    block = 3 * DISC_POCKET + 2 * (DISC_PITCH - DISC_POCKET)
    u1 = edge + (chan - 0.8 - edge - block) / 2 + DISC_POCKET / 2
    t.track(u1, TRACK_ROWS_V[0], 3, 3, first=1)

    pill_u = edge + 1.2 + (PILL_L + 2 * PILL_CLR) / 2
    t.pill(pill_u, HEADER_C)
    t.hex_pocket(pill_u + PILL_L / 2 + PILL_CLR + 1.3 + HEX_SPARE / 2, HEADER_C)

    t.well("storage channel", chan, 0, U, V, r=1.0)
    uc = chan + WELL_W / 2
    for k in range(3):
        t.g_cyl(uc, 7.6 + 14.6 * k, Z_PLAT)
    t.g_disc(uc, 7.6 + 14.6 * 3, FLOOR, Z_PLAT)
    t.notes.append("Nest holds 30 Rebels; the spare Rebel has its own pocket beside the inlay. "
                   "Channel holds 2 cylinders + spare cylinder + spare Base.")
    return t


def tray_reclaimer():
    """20 Rebels (hex) + spare, 15 Bases + spare, 1 cylinder."""
    t = Tray("reclaimer")
    lines = [3] * 7
    du, dv = t.nest_extent(lines, "u")
    edge = 0.4 + du + 0.8
    t.deck(0, 0, edge, V, Z_NEST_DECK)
    t.nest("rebel nest", 0.4, (V - dv) / 2, lines, "u")

    t.deck(edge, 0, U, V, Z_PLAT)
    block = 5 * DISC_POCKET + 4 * (DISC_PITCH - DISC_POCKET)
    u1 = edge + (U - edge - block) / 2 + DISC_POCKET / 2
    t.track(u1, TRACK_ROWS_V[0], 5, 3, first=1, cards=(1, 3, 6, 10, 15))

    zero_u = edge + 0.8 + CARD_W / 2
    t.disc_spot(zero_u, HEADER_C, "0", pocket=False, card=True)
    pill_u = zero_u + CARD_W / 2 + 0.9 + (PILL_L + 2 * PILL_CLR) / 2
    t.pill(pill_u, HEADER_C)
    w0 = pill_u + PILL_L / 2 + PILL_CLR + 1.0
    t.well("storage well", w0, HEADER_C - WELL_W / 2, U, HEADER_C + WELL_W / 2)
    t.g_cyl(w0 + 7.5, HEADER_C, Z_PLAT)
    t.g_disc(w0 + 7.5 + 14.4, HEADER_C, FLOOR, Z_PLAT)
    t.notes.append("Nest holds 20 Rebels + spare. Well holds the cylinder + spare Base; "
                   "there is no room for a spare cylinder in this tray.")
    return t


def tray_corporations():
    """8 SpecOps (hex) + spare, 16 Security (cube) + spare, 9 Bases + spare, 1 cylinder + spare."""
    t = Tray("corporations")
    band = 31.2                                  # top of the bottom band
    lines = [3, 3, 3]
    du, dv = t.nest_extent(lines, "u")
    split = 5 * DISC_PITCH + 1.2                 # track | nest boundary
    t.deck(0, 0, split, band, Z_PLAT)
    t.track(1.0 + DISC_POCKET / 2, 23.2, 5, 2, first=0)
    t.deck(split, 0, U, band, Z_NEST_DECK)
    t.nest("specops nest", split + (U - split - du) / 2, (band - dv) / 2, lines, "u")

    gw = 6 * CUBE_POCKET + 5 * (CUBE_PITCH - CUBE_POCKET)
    gh = 3 * CUBE_POCKET + 2 * (CUBE_PITCH - CUBE_POCKET)
    cz = gw + 2.2                                # cube zone | plateau boundary
    t.deck(0, band, cz, V, Z_CUBE_DECK)
    t.cube_grid(1.1, band + (V - band - gh) / 2, 6, 3, count=17)

    t.deck(cz, band, U, V, Z_PLAT)
    mid = (cz + U) / 2
    wl = 43.6
    t.well("storage well", mid - wl / 2, band + 2.4, mid + wl / 2, band + 2.4 + WELL_W)
    for k in range(2):
        t.g_cyl(mid - wl / 2 + 7.5 + 14.4 * k, band + 2.4 + WELL_W / 2, Z_PLAT)
    t.g_disc(mid - wl / 2 + 7.5 + 14.4 * 2, band + 2.4 + WELL_W / 2, FLOOR, Z_PLAT)
    t.pill(mid, band + 2.4 + WELL_W + 1.9 + (PILL_W + 2 * PILL_CLR) / 2)
    t.notes.append("Nest holds 8 SpecOps + spare; grid holds 16 Security + spare (one pocket left over). "
                   "Well holds cylinder + spare cylinder + spare Base.")
    return t


def tray_marsgov():
    """30 Troops (cube) + spare, 3 Bases + spare, 2 cylinders + spare."""
    t = Tray("marsgov")
    t.deck(0, HEADER_V, U, V, Z_PLAT)
    pill_u = 1.4 + (PILL_L + 2 * PILL_CLR) / 2
    t.pill(pill_u, HEADER_C)
    left = pill_u + PILL_L / 2 + 1.2
    u1 = left + (U - left - 4 * DISC_PITCH) / 2 + DISC_PITCH / 2
    t.track(u1, HEADER_C, 4, 1, first=0)

    gw = 8 * CUBE_POCKET + 7 * (CUBE_PITCH - CUBE_POCKET)
    gh = 4 * CUBE_POCKET + 3 * (CUBE_PITCH - CUBE_POCKET)
    cz = gw + 2.4
    t.deck(0, 0, cz, HEADER_V, Z_CUBE_DECK)
    t.cube_grid(1.2, (HEADER_V - gh) / 2, 8, 4, count=31)

    t.deck(cz, 0, U, HEADER_V, Z_PLAT)
    s = 29.6
    wu, wv = (cz + U) / 2 - s / 2, HEADER_V / 2 - s / 2
    t.well("storage well", wu, wv, wu + s, wv + s)
    for k, (i, j) in enumerate(((0, 1), (1, 1), (0, 0))):
        t.g_cyl(wu + 7.4 + 14.8 * i, wv + 7.4 + 14.8 * j, Z_PLAT)
    t.g_disc(wu + 7.4 + 14.8, wv + 7.4, FLOOR, Z_PLAT)
    t.notes.append("Grid holds 30 Troops + spare (one pocket left over). "
                   "Well holds 2 cylinders + spare cylinder + spare Base.")
    return t


def tray_earthgov():
    """16 Troops (cube) + spare, plus an open bin."""
    t = Tray("earthgov")
    gw = 6 * CUBE_POCKET + 5 * (CUBE_PITCH - CUBE_POCKET)
    gh = 3 * CUBE_POCKET + 2 * (CUBE_PITCH - CUBE_POCKET)
    cz = gw + 2.2
    t.deck(0, 0, cz, HEADER_V, Z_CUBE_DECK)
    t.cube_grid(1.1, (HEADER_V - gh) / 2, 6, 3, count=17)
    t.deck(0, HEADER_V, cz, V, Z_PLAT)
    t.pill(cz / 2, HEADER_C)
    t.deck(cz, 0, cz + 1.0, V, Z_BIN_WALL)
    t.pockets.append(("open bin", rect(cz + 1.0, 0, U, V)))
    t.notes.append("Grid holds 16 Troops + spare (one pocket left over). "
                   "Open bin is for Satellite markers or anything else Earth Government needs.")
    return t


def tray_spares():
    """Plain lidded bin that fills the sixth slot."""
    t = Tray("spares")
    t.pockets.append(("open bin", rect(0, 0, U, V)))
    t.notes.append("Single open bin for anything left over.")
    return t


FACTIONS = [
    # key, layout, piece colour for the previews
    ("red-dust", tray_red_dust, (0.80, 0.16, 0.14)),
    ("reclaimer", tray_reclaimer, (0.95, 0.52, 0.12)),
    ("corporations", tray_corporations, (0.20, 0.20, 0.22)),
    ("marsgov", tray_marsgov, (0.16, 0.36, 0.80)),
    ("earthgov", tray_earthgov, (0.93, 0.93, 0.90)),
    ("spares", tray_spares, (0.6, 0.6, 0.6)),
]

# =============================================================================
# Checks
# =============================================================================


def check(tray, body, lid, tile):
    """Return a list of problems (empty = all good) and a few measured numbers."""
    problems = []
    box = body.bounding_box()
    size = [box[i + 3] - box[i] for i in range(3)]
    for got, want, axis in zip(size, (LX, LY, H + RIM_H), "XYZ"):
        if abs(got - want) > 0.02:
            problems.append(f"outer size {axis} is {got:.2f}, expected {want}")
    if body.volume() <= 0 or lid.volume() <= 0:
        problems.append("empty solid")

    # pocket footprints: inside the interior, and at least 0.6 mm of material between any two
    inside = rect(0, 0, U, V).offset(0.02, JoinType.Miter, 2.0)
    for name, cs in tray.pockets:
        if (cs - inside).area() > 1e-3:
            problems.append(f"{name} reaches outside the tray interior")
    grown = [(n, cs.offset(0.3, JoinType.Round, 2.0, 24)) for n, cs in tray.pockets]
    for i in range(len(grown)):
        for j in range(i + 1, len(grown)):
            if (grown[i][1] ^ grown[j][1]).area() > 1e-3:
                problems.append(f"less than 0.6 mm between '{grown[i][0]}' and '{grown[j][0]}'")

    # every wooden piece must drop in without touching plastic, clear the lid, and be unable
    # to climb out of its pocket when the lid is on (box stored on edge)
    lid_gap = []
    for g in tray.ghosts:
        solid = g["solid"].translate((X0, Y0, 0))
        hit = (body ^ solid).volume()
        if hit > 1e-3:
            problems.append(f"{g['kind']} collides with the tray ({hit:.2f} mm3)")
        if (lid ^ solid).volume() > 1e-6:
            problems.append(f"{g['kind']} collides with the lid")
        head = ZL - g["top"]
        lid_gap.append((g["kind"], head))
        if head < 0.3:
            problems.append(f"{g['kind']} has only {head:.2f} mm under the lid")
        hold = g["retain"] - (g["top"] - {"hex": HEX_H, "disc": DISC_T, "cyl": CYL_H, "cube": CUBE}[g["kind"]])
        if hold - head < 0.8:
            problems.append(f"{g['kind']} could escape its pocket (pocket {hold:.1f} deep, {head:.2f} headroom)")

    # pieces must not overlap each other
    solids = [g["solid"] for g in tray.ghosts]
    if solids:
        shared = sum(s.volume() for s in solids) - m_union(solids).volume()
        if shared > 1e-2:
            problems.append(f"wooden pieces overlap each other ({shared:.2f} mm3)")

    # nothing should be too thin to print: slice at several heights and look for slivers
    # that vanish when the outline is shrunk by 0.3 mm and grown back
    for z in (2.0, 4.0, 6.5, 7.2, 9.2, 11.0, 12.6, H + RIM_H / 2):
        sl = body.slice(z)
        thin = sl - sl.offset(-0.29, JoinType.Round, 2.0, 16).offset(0.30, JoinType.Round, 2.0, 16)
        slivers = [c.area() for c in thin.decompose() if c.area() > 0.15]   # sharp corners leave specks
        if slivers:
            problems.append(f"{len(slivers)} feature(s) thinner than 0.6 mm at z = {z:.2f} (largest {max(slivers):.2f} mm2)")

    # the lid must not overlap the tray when closed
    overlap = (body ^ lid).volume()
    if overlap > 1e-3:
        problems.append(f"closed lid overlaps the tray by {overlap:.3f} mm3")
    # ... and must pull straight out along -X, catching only on the detent bumps.
    # The plain lid is convex, so its sweep is a hull.
    blank = lid_blank()
    swept = Manifold.batch_hull([blank, blank.translate((-LX - 2.0, 0, 0))])
    snag = ((body - detent_bumps()) ^ swept).volume()
    if snag > 1e-3:
        problems.append(f"lid snags on the tray while sliding ({snag:.3f} mm3)")

    # the lid must be the same turned end for end, so it goes in either way round
    turned = lid.translate((-LID_CX, -LY / 2, 0)).rotate((0, 0, 180)).translate((LID_CX, LY / 2, 0))
    differ = (lid - turned).volume() + (turned - lid).volume()
    if differ > 1e-2:
        problems.append(f"lid is not the same turned end for end ({differ:.2f} mm3 differs)")
    # the recesses in the lid's top face must leave a printable skin, and must not meet the
    # click notches underneath: seen from above, they may not remove any of the lid's outline
    if LID_T - max(GROOVE_D, DOT_T) < 0.39:
        problems.append("less than 0.4 mm of lid left under a recess")
    through = build_lid(recessed=False).project().area() - lid.project().area()
    if through > 1e-3:
        problems.append(f"a recess in the lid cuts right through it ({through:.2f} mm2)")

    # the inlay must press in: its body drops into the recess freely, and only the ribs squeeze
    if hasattr(tray, "pill_at"):
        seat = (X0 + tray.pill_at[0], Y0 + tray.pill_at[1], Z_PLAT - PILL_T)
        loose = (body ^ build_tile(ribs=False).translate(seat)).volume()
        if loose > 1e-3:
            problems.append(f"inlay body does not fit its recess ({loose:.3f} mm3 overlap)")
        bite = (body ^ tile.translate(seat)).volume()
        if not 0.02 < bite < 1.0:
            problems.append(f"inlay ribs should squeeze the recess a little; they overlap it by {bite:.3f} mm3")
        tb = tile.bounding_box()
        squeeze = ((tb[4] - tb[1]) - (PILL_W + 2 * PILL_CLR)) / 2
        if abs(squeeze - PILL_RIB) > 0.005 or abs((tb[3] - tb[0]) - PILL_L) > 0.005:
            problems.append(f"inlay is {tb[3] - tb[0]:.2f} x {tb[4] - tb[1]:.2f}; ribs squeeze {squeeze:.3f} per side, expected {PILL_RIB}")

    # a second tray must sit on this one without touching it, and the rim alone (lid off)
    # must stop it sliding in every direction
    above = body.translate((0, 0, H))
    clash = (above ^ (body + lid)).volume()
    if clash > 1e-3:
        problems.append(f"a tray stacked on top overlaps this one ({clash:.3f} mm3)")
    for dx, dy in ((0.6, 0), (-0.6, 0), (0, 0.6), (0, -0.6)):
        if (above.translate((dx, dy, 0)) ^ body).volume() < 1e-3:
            problems.append(f"a stacked tray is free to slide ({dx:+.1f}, {dy:+.1f})")

    heads = {}
    for kind, head in lid_gap:
        heads[kind] = min(head, heads.get(kind, 99))
    return problems, dict(size=size, headroom=heads, volume_cm3=body.volume() / 1000)


# =============================================================================
# Export
# =============================================================================


def mesh_of(m):
    mesh = m.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3].astype(np.float64), np.asarray(mesh.tri_verts).astype(np.int64)


def write_stl(path, m):
    v, f = mesh_of(m)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    n = n / np.where(ln > 0, ln, 1)[:, None]
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(b"Red Dust Rebellion tray".ljust(80, b" "))
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())


_CT = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
       '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
       '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
_RELS = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
         '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')


def write_3mf(path, parts):
    """parts = [(name, [solid, ...], (tx, ty))], each placed on the plate at its offset.

    Several solids under one name become one object with separate parts, so the slicer
    can give each part its own filament. Such solids are given as (part name, solid,
    filament number); Bambu Studio reads those names and filament numbers from the
    model_settings.config written alongside, and other slicers ignore that file.
    """
    res, build, settings, oid = [], [], [], 0
    for name, solids, (tx, ty) in parts:
        ids = []
        for m in solids:
            part_name, m, filament = m if isinstance(m, tuple) else (name, m, 1)
            oid += 1
            ids.append((oid, part_name, filament))
            v, f = mesh_of(m)
            res.append(f'<object id="{oid}" name="{part_name}" type="model"><mesh><vertices>')
            res.append("".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in v.tolist()))
            res.append("</vertices><triangles>")
            res.append("".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f.tolist()))
            res.append("</triangles></mesh></object>")
        if len(ids) > 1:
            oid += 1
            comps = "".join(f'<component objectid="{i}"/>' for i, _, _ in ids)
            res.append(f'<object id="{oid}" name="{name}" type="model"><components>{comps}</components></object>')
            settings.append(f'  <object id="{oid}">\n    <metadata key="name" value="{name}"/>\n    <metadata key="extruder" value="1"/>')
            for i, part_name, filament in ids:
                settings.append(f'    <part id="{i}" subtype="normal_part">\n      <metadata key="name" value="{part_name}"/>\n'
                                f'      <metadata key="extruder" value="{filament}"/>\n    </part>')
            settings.append("  </object>")
        build.append(f'<item objectid="{oid}" transform="1 0 0 0 1 0 0 0 1 {tx:.3f} {ty:.3f} 0"/>')
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
           "<resources>", *res, "</resources><build>", *build, "</build></model>"]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CT)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("3D/3dmodel.model", "\n".join(out))
        if settings:
            z.writestr("Metadata/model_settings.config",
                       "\n".join(['<?xml version="1.0" encoding="UTF-8"?>', "<config>", *settings, "</config>"]) + "\n")


# =============================================================================
# Main
# =============================================================================

TRAY_RGB = (0.74, 0.75, 0.78)
LID_RGB = (0.66, 0.68, 0.72)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-previews", action="store_true")
    ap.add_argument("--only", help="build a single tray by key")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    models, stl_dir, prev = root / "models", root / "models" / "stl", root / "previews"
    stl_dir.mkdir(parents=True, exist_ok=True)
    prev.mkdir(exist_ok=True)

    # the lid and both inlays are the same for every tray
    lid = build_lid()
    dot_fill = build_dot_fill()
    box = lid.bounding_box()
    to_bed = (-box[0], -box[1], -box[2])          # lid lies wide face down, as it prints
    lid_bed, dot_fill_bed = lid.translate(to_bed), dot_fill.translate(to_bed)
    dot, tile = build_dot(), build_tile()
    write_stl(stl_dir / "lid.stl", lid_bed)
    write_stl(stl_dir / "lid-dot.stl", dot)
    write_stl(stl_dir / "inlay.stl", tile)
    write_3mf(models / "lid-ams.3mf", [("lid", [("lid", lid_bed, 1), ("dot", dot_fill_bed, 2)], (68.5, 93.0))])

    built = []
    failed = False
    for key, builder, rgb in FACTIONS:
        if args.only and key != args.only:
            continue
        tray = builder()
        body = tray.build()
        problems, info = check(tray, body, lid, tile)

        write_stl(stl_dir / f"{key}-tray.stl", body)
        parts = [(f"{key} tray", [body], (68.5, 30.0)), ("lid", [lid_bed], (68.5, 112.0)), ("lid dot", [dot], (90.0, 205.0))]
        if hasattr(tray, "pill_at"):
            parts.append(("inlay", [tile], (128.0, 205.0)))
        write_3mf(models / f"{key}.3mf", parts)

        sx, sy, sz = info["size"]
        head = ", ".join(f"{k} {v:.2f}" for k, v in sorted(info["headroom"].items()))
        print(f"\n{key}: tray {sx:.1f} x {sy:.1f} x {sz:.1f} mm (rim included), {body.num_tri()} triangles, "
              f"~{info['volume_cm3']:.0f} cm3 solid volume")
        if head:
            print(f"  headroom under lid (mm): {head}")
        for note in tray.notes:
            print(f"  {note}")
        counts = {}
        for g in tray.ghosts:
            counts[g["kind"]] = counts.get(g["kind"], 0) + 1
        if counts:
            print("  pieces test-fitted: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
        if problems:
            failed = True
            for p in dict.fromkeys(problems):
                print(f"  PROBLEM: {p}")
        else:
            print("  checks passed")
        built.append((key, tray, body, rgb))

    print(f"\nstack: 3 trays x {H} mm + {RIM_H} mm rim = {3 * H + RIM_H:.1f} mm; "
          f"two columns = {2 * LY:.1f} x {LX:.1f} mm footprint")

    if not args.no_previews:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import preview

        def shifted(objs, dx, dy, dz):
            return [(v + np.array([dx, dy, dz]), f, c) for v, f, c in objs]

        lid_mesh, dot_mesh = mesh_of(lid), mesh_of(dot_fill)
        loaded_all, closed_all = [], []
        for key, tray, body, rgb in built:
            v, f = mesh_of(body)
            empty = [(v, f, TRAY_RGB)]
            loaded = list(empty)
            for kind, solid in tray.ghost_solids():
                loaded.append((*mesh_of(solid), rgb))
            if hasattr(tray, "pill_at"):
                loaded.append((*mesh_of(tile.translate((X0 + tray.pill_at[0], Y0 + tray.pill_at[1], Z_PLAT - PILL_T))), rgb))
            closed = [(v, f, TRAY_RGB), (*lid_mesh, LID_RGB), (*dot_mesh, rgb)]
            loaded_all.append(loaded)
            closed_all.append(closed)
            top = preview.hstack([preview.render(empty, "top", ppm=9), preview.render(loaded, "top", ppm=9)])
            iso = preview.hstack([preview.render(loaded, (-28, 38), ppm=9), preview.render(closed, (-28, 38), ppm=9)])
            preview.save(prev / f"{key}.png", preview.vstack([top, iso]))
            print(f"preview: previews/{key}.png")

        if len(built) == len(FACTIONS):
            # every tray, loaded, seen from above
            rows = [preview.hstack([preview.render(loaded_all[i], "top", ppm=7), preview.render(loaded_all[i + 1], "top", ppm=7)])
                    for i in range(0, len(loaded_all), 2)]
            preview.save(prev / "overview.png", preview.vstack(rows))
            print("preview: previews/overview.png")
            # the six trays as they sit in the box: two columns, three high
            stack = []
            for i, closed in enumerate(closed_all):
                col, layer = i // 3, i % 3
                stack += shifted(closed, 0, col * (LY + 1.0), layer * H)
            preview.save(prev / "box-stack.png", preview.render(stack, (-32, 34), ppm=8))
            print("preview: previews/box-stack.png")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
