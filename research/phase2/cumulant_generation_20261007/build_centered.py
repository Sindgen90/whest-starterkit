from pathlib import Path
R=Path(__file__).resolve().parent;base=(R/'unclip34/estimator.py').read_text();helper=(R/'centered_functions.py').read_text()
def replace(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b)
for name in ('centered','coherent'):
    coherent=name=='coherent';s=base+'\n'+helper
    s=replace(s,'            rows.append(mu)','            stable_mu,stable_var,stable3,stable4 = centered_update(m,v,moments,f,sigma,phi,d3,d4,coherent='+str(coherent)+')\n            mu = stable_mu\n            rows.append(mu)')
    s=replace(s,'marginal_var = fnp.maximum(corrected[2]-mu*mu,1e-12)','marginal_var = stable_var')
    s=replace(s,'                cov = g11\n','                cov = g11\n                fnp.fill_diagonal(cov,stable_var)\n')
    s=replace(s,'post3 = corrected[3]-3*mu*corrected[2]+2*mu*mu*mu','post3 = stable3')
    s=replace(s,'post4 = (corrected[4]-4*mu*corrected[3]-3*corrected[2]*corrected[2]\n                             +12*mu*mu*corrected[2]-6*fnp.power(mu,4))','post4 = stable4')
    if coherent:
        anchor='                delta = (fnp.stack([f[k][3] for k in range(1,5)])*(d3/6)[None,:]'
        s=replace(s,anchor,'                attenuation = coherent_factor(m,v,moments,f,sigma,phi,d3,d4)\n                d3 = d3*attenuation\n                d4 = d4*attenuation\n                pair3 = pair3*attenuation\n                pair4 = pair4*attenuation\n'+anchor)
        s=replace(s,'rows.append(final_mean(m,v,d3,d4))','mom,f,sig,_,ph = responses(m,v)\n                attenuation = coherent_factor(m,v,mom,f,sig,ph,d3,d4)\n                rows.append(centered_update(m,v,mom,f,sig,ph,d3*attenuation,d4*attenuation,coherent=True)[0])')
    d=R/name;d.mkdir(exist_ok=True);assert not (d/'estimator.py').exists();(d/'estimator.py').write_text(s)
print('Built centered and coherent, no target fitting')
