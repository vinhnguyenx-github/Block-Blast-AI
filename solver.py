# solver.py
# Block-puzzle solver with cascading line clears.
# - Board: np.ndarray[H,W] of 0/1
# - Pieces: list of np.ndarray (e.g., 3×5 editors)
# - No reflections. Rotation optional (default False).
# - solve()  : place in GIVEN ORDER
# - solve_any_order(): try all permutations, then map back to original indices

from __future__ import annotations
import numpy as np
from itertools import permutations
from typing import List, Dict

# ---------- helpers ----------
def _trim(p: np.ndarray) -> np.ndarray:
    p = (p > 0).astype(np.uint8)
    rows = np.where(p.any(1))[0]
    cols = np.where(p.any(0))[0]
    if rows.size == 0 or cols.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    return p[rows.min():rows.max()+1, cols.min():cols.max()+1]

def _orientations(p: np.ndarray, allow_rotate: bool) -> list[np.ndarray]:
    p = _trim(p)
    if p.size == 0:
        return [p]
    cands = [p]
    if allow_rotate:
        cands += [np.rot90(p, k) for k in (1, 2, 3)]
    uniq, seen = [], set()
    for x in cands:
        x = x.astype(np.uint8, copy=False)
        key = (x.shape, x.tobytes())
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    return uniq

def _can_place(B: np.ndarray, P: np.ndarray, r: int, c: int) -> bool:
    h, w = P.shape
    H, W = B.shape
    if r < 0 or c < 0 or r + h > H or c + w > W:
        return False
    return np.all(B[r:r+h, c:c+w] + P <= 1)

def _place(B: np.ndarray, P: np.ndarray, r: int, c: int) -> np.ndarray:
    nb = B.copy()
    h, w = P.shape
    nb[r:r+h, c:c+w] |= P
    return nb

def _clear_lines_cascade(B: np.ndarray) -> np.ndarray:
    nb = B.copy()
    while True:
        full_rows = np.all(nb == 1, axis=1)
        full_cols = np.all(nb == 1, axis=0)
        if not (full_rows.any() or full_cols.any()):
            return nb
        if full_rows.any():
            nb[full_rows, :] = 0
        if full_cols.any():
            nb[:, full_cols] = 0

def _place_and_clear(B: np.ndarray, P: np.ndarray, r: int, c: int) -> np.ndarray:
    return _clear_lines_cascade(_place(B, P, r, c))

# ---------- core DFS ----------
def _dfs(B: np.ndarray, oris: list[list[np.ndarray]], k: int, res: List[Dict]) -> bool:
    if k == len(oris):
        return True
    H, W = B.shape
    # try larger shapes first
    for P in sorted(oris[k], key=lambda x: x.sum(), reverse=True):
        if P.size == 0:
            if _dfs(B, oris, k+1, res):
                res.append({'piece_index': k, 'row': 0, 'col': 0, 'shape': P})
                return True
            continue
        if P.sum() > (B == 0).sum():  # prune: not enough free cells
            continue
        h, w = P.shape
        for r in range(H - h + 1):
            for c in range(W - w + 1):
                if _can_place(B, P, r, c):
                    nb = _place_and_clear(B, P, r, c)
                    if _dfs(nb, oris, k+1, res):
                        res.append({'piece_index': k, 'row': r, 'col': c, 'shape': P})
                        return True
    return False

# ---------- public APIs ----------
def solve(board: np.ndarray, pieces: list[np.ndarray], allow_rotate: bool = False):
    """
    Place pieces IN GIVEN ORDER. Returns list of placements in play order or None.
    """
    B = (board > 0).astype(np.uint8)
    oris = [_orientations(p, allow_rotate) for p in pieces]
    res: List[Dict] = []
    ok = _dfs(B, oris, 0, res)
    if not ok:
        return None
    res.reverse()
    return res

def solve_any_order(board: np.ndarray, pieces: list[np.ndarray], allow_rotate: bool = False):
    """
    Try all piece orders (e.g., 3! = 6). Returns placements mapped to original indices.
    """
    for order in permutations(range(len(pieces))):
        ordered = [pieces[i] for i in order]
        sol = solve(board, ordered, allow_rotate=allow_rotate)
        if sol is not None:
            # remap indices back to original piece numbering
            for s in sol:
                s['piece_index'] = order[s['piece_index']]
            return sol
    return None