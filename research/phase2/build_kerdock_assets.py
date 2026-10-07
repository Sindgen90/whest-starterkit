"""Offline generalization of Skye Nygaard's MIT Kerdock chirp construction.

Source: https://github.com/SkyeNygaard/ARC-Whitebox/blob/main/
arc_whitebox/scripts/eval_kerdock_design.py (downloaded 2026-09-26).
No network weights, identities, or target means enter asset generation.
"""
from pathlib import Path
import math
import numpy as np
from threadpoolctl import threadpool_limits


def chirps(width):
    degree = width.bit_length() - 2
    # Irreducible binary polynomials for odd extension degrees 1,3,5,7,9.
    polynomial = {1: 0b11, 3: 0b1011, 5: 0b100101,
                  7: 0b10000011, 9: 0b1000010001}[degree]
    size = 1 << degree
    def mul(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & size:
                a ^= polynomial
        return result
    table = np.array([[mul(a,b) for b in range(size)] for a in range(size)])
    trace = np.zeros(size, dtype=np.int64)
    term = np.arange(size)
    for _ in range(degree):
        trace ^= term
        term = table[term,term]
    assert set(trace.tolist()) <= {0,1}
    value = np.arange(size)
    power = value.copy()
    quadratic = np.zeros(size, dtype=np.int64)
    for _ in range(degree // 2):
        power = table[power,power]
        quadratic ^= table[power,value]
    low = trace[quadratic[table]]
    high = low ^ trace[table]
    return (1 - 2*np.concatenate((low,high),axis=1)).astype(np.float32)


def main():
    assets = {}
    for width in (4,16,64,256,1024):
        bank = chirps(width)
        # Fixed target-independent ordering, with u=0 also eligible.
        rng = np.random.default_rng(20260926 + width)
        assets[f"chirps_{width}"] = bank[rng.permutation(len(bank))]
        assets[f"radius_{width}"] = np.array(math.sqrt(2)*math.exp(
            math.lgamma((width+1)/2)-math.lgamma(width/2)), dtype=np.float64)
        with threadpool_limits(4):
            q,r = np.linalg.qr(np.random.default_rng(3).standard_normal((width,width)))
        q *= np.where(np.diag(r)<0,-1,1)[None,:]
        assets[f"rotation_{width}"] = q.astype(np.float32)
    path = Path(__file__).parent / "kerdock_baseline" / "design.npz"
    np.savez_compressed(path, **assets)
    print(path, path.stat().st_size)


if __name__ == "__main__":
    main()
