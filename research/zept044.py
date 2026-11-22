"""Minimal ZEPT-044 computational reproduction. Synthetic data only."""
import numpy as np
s=np.linspace(-1,1,8);s/=np.linalg.norm(s)
b=np.ones(8);rng=np.random.default_rng(44044)
for eps in np.linspace(0,1,21):
    B=b+eps*s;C=np.outer(B,B)+.04*np.eye(8)
    z=np.linalg.solve(C,s);w=z/(s@z)
    y=rng.normal(size=(100000,1))*B+rng.normal(scale=.2,size=(100000,8))+.25*s
    for label,a in [('differential',s),('endpoint',np.eye(8)[-1]/s[-1]),('oracle',w)]:
        assert abs(a@s-1)<1e-10
        print(eps,label,a@C@a,np.exp(-.5*a@C@a),abs(np.exp(1j*(y@a)).mean()))
