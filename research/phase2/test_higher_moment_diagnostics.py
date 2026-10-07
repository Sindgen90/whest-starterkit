import numpy as np
from diagnose_higher_moments import corrections,cumulants,second_raw_correction


def test_correction_agrees_with_integrated_hermite_density():
    # Independent numerical integration verifies signs, scaling, and the H6 term.
    nodes,weights = np.polynomial.legendre.leggauss(180)
    for mean,sigma in ((-.7,1.3),(0.,.5),(1.2,.8)):
        lower = -mean/sigma
        x = lower+(nodes+1)*(12-lower)/2
        mass = weights*(12-lower)/2*np.exp(-x*x/2)/np.sqrt(2*np.pi)
        hermites = (x**3-3*x,x**4-6*x*x+3,x**6-15*x**4+45*x*x-15)
        skew,excess = .13,.24
        numerical = np.array([np.sum(mass*(mean+sigma*x)*h)*factor
                              for h,factor in zip(hermites,(skew/6,excess/24,skew**2/72))])
        analytical = corrections(mean,sigma*sigma,skew*sigma**3,excess*sigma**4)
        np.testing.assert_allclose(analytical,numerical,atol=2e-14)
        numerical_second = np.sum(mass*(mean+sigma*x)**2*(skew*hermites[0]/6+excess*hermites[1]/24))
        np.testing.assert_allclose(second_raw_correction(mean,sigma*sigma,skew,excess),numerical_second,atol=2e-14)


def test_gaussian_cumulants_give_zero_correction():
    np.testing.assert_array_equal(corrections(np.array([-2.,0.,1.]),np.ones(3),0.,0.),np.zeros((3,3)))


def test_cumulants_are_invariant_to_centering_reference():
    x = np.array([-2.,-.1,.2,.7,1.,3.,4.])
    raw = lambda y:np.array([np.mean(y**p) for p in range(1,5)])
    _,a,b = cumulants(raw(x),len(x))
    _,c,d = cumulants(raw(x+5),len(x))
    np.testing.assert_allclose([a,b],[c,d],rtol=1e-10)
