"""Research-only complete transport of the paired fourth-tensor projection.

The retained tensor has U[i,i,i,i]=Q[i,i], U[i,i,j,j]=Q[i,j] for i!=j
and all permutations, with every other index pattern zero. This projection
does not retain the full fourth tensor of the activation distribution.
"""
import flopscope.numpy as fnp


def transport(paired, weight, mode='gram'):
    """Return the marginal fourth cumulant and transformed paired plane."""
    square = weight*weight
    gram = fnp.einsum('ij,ia,jb->ab',paired,square,square)
    diagonal = 3*fnp.diag(gram)-2*((square*square).T@fnp.diag(paired))
    if mode == 'gram':
        plane = gram
    elif mode == 'complete':
        off = fnp.copy(paired)
        fnp.fill_diagonal(off,0)
        # H-D: each omitted cross-pairing has i != j. This is O(n^4).
        cross = fnp.einsum('ij,ia,ib,ja,jb->ab',off,weight,weight,weight,weight)
        plane = gram+2*cross
    else:
        raise ValueError(f'Unknown fourth transport: {mode}')
    # Exact for this paired-only model; identical in both variants.
    fnp.fill_diagonal(plane,diagonal)
    return diagonal,plane
