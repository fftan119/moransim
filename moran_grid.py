#!/usr/bin/env python3
"""
Generate Moran-process GRID values for target rho pairs around 0.5.

Given:
- N
- a delta, so the target pair is (0.5-delta, 0.5+delta)
- an optional r-threshold

the script solves for r such that

    rho_i(r) = (1 - r^{-i}) / (1 - r^{-N})      if r != 1
    rho_i(1) = i / N

for each i = 1, ..., N-1.

It prints a Python-style GRID block to the terminal:
    GRID = [
        (r_value, i0),   # rho = target
        ...
    ]

Any pair whose solved r exceeds the threshold is omitted.
"""

from __future__ import annotations

import argparse
import math
from typing import Optional


def rho_i(i: int, N: int, r: float) -> float:
    """Standard Moran-process fixation probability."""
    if abs(r - 1.0) < 1e-14:
        return i / N
    return (1.0 - r ** (-i)) / (1.0 - r ** (-N))


def solve_r_for_target(
    i: int,
    N: int,
    target_rho: float,
    lo: float = 1e-9,
    hi: float = 1e6,
    tol: float = 1e-12,
    max_iter: int = 1000,
) -> Optional[float]:
    """
    Solve rho_i(r) = target_rho by bisection.

    rho_i(r) is monotone increasing in r for fixed i,N in the Moran process,
    so bisection is appropriate.
    """
    f_lo = rho_i(i, N, lo) - target_rho
    f_hi = rho_i(i, N, hi) - target_rho

    if abs(f_lo) < tol:
        return lo
    if abs(f_hi) < tol:
        return hi

    if f_lo * f_hi > 0:
        return None

    left, right = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = rho_i(i, N, mid) - target_rho

        if abs(f_mid) < tol or (right - left) < tol:
            return mid

        if f_lo * f_mid <= 0:
            right = mid
            f_hi = f_mid
        else:
            left = mid
            f_lo = f_mid

    return 0.5 * (left + right)


def format_float(x: float, decimals: int = 4) -> str:
    return f"{x:.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Moran-process GRID values for target rho pairs."
    )
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="Population size N.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        required=True,
        help="Target pair is (0.5-delta, 0.5+delta). Example: 0.1 gives 0.4 and 0.6.",
    )
    parser.add_argument(
        "--r-threshold",
        type=float,
        default=None,
        help="Omit any solved pair with r > threshold.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimals to print for r.",
    )

    args = parser.parse_args()

    N = args.N
    delta = args.delta
    r_threshold = args.r_threshold
    decimals = args.decimals

    if N < 2:
        raise ValueError("N must be at least 2.")

    if not (0.0 <= delta <= 0.5):
        raise ValueError("delta must satisfy 0 <= delta <= 0.5.")

    low_target = 0.5 - delta
    high_target = 0.5 + delta

    if not (0.0 <= low_target <= 1.0 and 0.0 <= high_target <= 1.0):
        raise ValueError("Target rho values must lie in [0,1].")

    print("GRID = [")
    for i in range(1, N):
        for target in (low_target, high_target):
            r_val = solve_r_for_target(i, N, target)
            if r_val is None:
                continue
            if r_threshold is not None and r_val > r_threshold:
                continue

            print(
                f"    ({format_float(r_val, decimals)}, {i}),   # rho = {target:.{decimals}f}"
            )
    print("]")


if __name__ == "__main__":
    main()