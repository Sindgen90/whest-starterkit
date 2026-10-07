"""Algebra checks for the A716 reconstruction choices."""
import itertools

import numpy as np
import flopscope.numpy as fnp
from puffi_reference import fourth_transport, new_third_source, responses, third_slices


def test_positive_part_response_identities():
    mean=fnp.asarray([-.7,.2,1.3],dtype=fnp.float32)
    var=fnp.asarray([.4,1.1,2.0],dtype=fnp.float32)
    moments,f,_,_,_=responses(mean,var)
    np.testing.assert_allclose(np.asarray(f[2][1]),2*np.asarray(moments[1]),rtol=1e-6)
    np.testing.assert_allclose(np.asarray(f[4][4]),24*np.asarray(moments[0]),rtol=1e-6)


def test_third_source_matches_requested_repeated_slices():
    diag=fnp.asarray([1.,2.,3.],dtype=fnp.float32)
    repeated=fnp.asarray([[1.,4.,5.],[6.,2.,7.],[8.,9.,3.]],dtype=fnp.float32)
    a,b=new_third_source(diag,repeated)
    got_diag,got_repeated=third_slices(a,b)
    np.testing.assert_allclose(np.asarray(got_diag),np.asarray(diag),rtol=1e-6)
    np.testing.assert_allclose(np.asarray(got_repeated),np.asarray(repeated),rtol=1e-6)


def test_fourth_projection_diagonal_matches_explicit_symmetric_tensor():
    paired=np.asarray([[1.2,-.3],[-.3,.7]],dtype=np.float32)
    weight=np.asarray([[.4,-.8],[1.1,.2]],dtype=np.float32)
    # Build the retained symmetric tensor whose (i,i,j,j) plane is `paired`.
    tensor=np.zeros((2,2,2,2),dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for axes in set(itertools.permutations((i,i,j,j))):
                tensor[axes]=paired[i,j]
    expected=np.einsum("ijkl,ia,ja,ka,la->a",tensor,weight,weight,weight,weight)
    diagonal,_,_=fourth_transport(fnp.asarray(paired),fnp.asarray(weight))
    np.testing.assert_allclose(np.asarray(diagonal),expected,rtol=2e-6,atol=2e-7)
