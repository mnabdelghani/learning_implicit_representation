import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from common import ImplicitLearner,rowspace_error
import contact,fourier_online

class DifferentialTests(unittest.TestCase):
    def test_contact_differential(self):
        rng=np.random.default_rng(31)
        for t in (20,75):
            c,B=contact.truth(t)
            for _ in range(10):
                z=rng.normal(size=8);eps=1e-6
                fd=np.column_stack([(contact.residual(z+eps*np.eye(8)[j],c,B)-contact.residual(z-eps*np.eye(8)[j],c,B))/(2*eps) for j in range(8)])
                np.testing.assert_allclose(contact.differential(z,c,B),fd,rtol=1e-7,atol=1e-6)
    def test_fourier_differential_including_clock(self):
        rng=np.random.default_rng(72);c=np.array(fourier_online.CONFIG['true_coefficients'])
        for _ in range(20):
            z=rng.normal(size=4);eps=1e-6
            fd=np.array([(fourier_online.residual(z+eps*np.eye(4)[j],c)-fourier_online.residual(z-eps*np.eye(4)[j],c))/(2*eps) for j in range(4)])
            np.testing.assert_allclose(fourier_online.differential(z[0],z[1],z[3],c),fd,atol=1e-9)
    def test_contact_physical_relation(self):
        c,B=contact.truth(10);state=np.array([.2,-.02,4]);u=np.array([.1,-.03])
        rates=contact.physical_rhs(state,u,c,B)
        np.testing.assert_allclose(contact.residual(np.r_[state,rates[:2],rates[2],u],c,B),0,atol=1e-12)
    def test_contact_gauge_invariance(self):
        c,B=contact.truth(20);F=contact.differential([.2,-.01,4,.3,-.02,.1,.2,.1],c,B)
        G=np.array([[2,.2,-.3],[.5,1,.1],[0,.4,1.5]])@F
        self.assertLess(rowspace_error(F,G,contact.CONFIG['context_scales']),1e-12)
    def test_fourier_nonzero_rate_partial(self):
        ph=np.linspace(0,2*np.pi,2000)
        lower=np.asarray(fourier_online.CONFIG['lower']);upper=np.asarray(fourier_online.CONFIG['upper'])
        self.assertGreater(lower[0]-np.hypot(max(abs(lower[1]),abs(upper[1])),max(abs(lower[2]),abs(upper[2]))),.57)
    def test_contact_actuation_bound(self):
        self.assertGreater(.5*.5-.3*.3,0)
    def test_contact_rk4_step_refinement(self):
        c,B=contact.truth(10);s=np.array([.2,-.02,4]);u=np.array([.1,-.03]);h=.01
        a=contact.rk4(s,u,h,c,B,4)
        b=contact.rk4(contact.rk4(s,u,h/2,c,B,4),u,h/2,c,B,4)
        self.assertLess(np.linalg.norm(a-b),2e-8)

class LearningTests(unittest.TestCase):
    def learner(self):return ImplicitLearner([.5,-.2],[1.,.3],[-2.,-2.],[2.,2.],.01,1.,steps=8)
    def test_projected_memory_descent(self):
        rng=np.random.default_rng(4);L=self.learner()
        for _ in range(100):
            d=L.observe(rng.normal(size=(10,2)),rng.normal(size=10))
            self.assertLessEqual(d['memory_after'],d['memory_before']+1e-12)
            self.assertTrue(np.all(L.theta>=L.lower));self.assertTrue(np.all(L.theta<=L.upper))
    def test_current_candidate_history_identity(self):
        rng=np.random.default_rng(42);L=self.learner();vs=[]
        for _ in range(10):
            x=rng.normal(size=2);s=rng.normal();vs.append(L.channels(x,s)[0]);L.observe(x,s)
        q=np.r_[np.array([.2,.8]),1.]
        direct=sum((1-L.beta)*L.beta**(len(vs)-1-i)*(v@q)**2/2 for i,v in enumerate(vs))
        self.assertAlmostEqual(direct,q@L.R@q/2,places=13)
    def test_energy_gradient(self):
        L=self.learner();L.observe([[1.,2.],[.1,.5]],[.2,.3]);q=np.r_[L.theta,1.];g=(L.R@q)[:-1];eps=1e-6
        def energy(th):
            v=np.r_[th,1.];return v@L.R@v/2
        fd=np.array([(energy(L.theta+eps*np.eye(2)[j])-energy(L.theta-eps*np.eye(2)[j]))/(2*eps) for j in range(2)])
        np.testing.assert_allclose(g,fd,atol=1e-10)
    def test_preupdate_report(self):
        L=self.learner();v=L.channels([1,2],.3)[0];q=np.r_[L.theta,1];expected=(v@q)**2/2
        d=L.observe([1,2],.3);self.assertAlmostEqual(d['energy'],expected,places=12)
    def test_frozen_coefficients(self):
        L=self.learner();prior=L.coefficients.copy();L.observe([1,2],4,adapt=False)
        np.testing.assert_array_equal(prior,L.coefficients)
    def test_single_sample_exception(self):
        q=np.array([[1.,0.]]);v=np.array([1.,0.]);xi=q@v
        update=xi[:,None]@v[None,:]-(xi[:,None]@xi[None,:])@q
        np.testing.assert_array_equal(update,0);self.assertEqual(float(xi@xi),1)
    def test_multiroot_stationary_exception(self):
        R=np.diag([0.,0.,1.,2.]);Q=np.array([[0.,0.,1.,0.],[0.,0.,0.,1.]])
        np.testing.assert_array_equal(Q@R-(Q@R@Q.T)@Q,0)
        self.assertGreater(np.trace(Q@R@Q.T),0)
    def test_short_contact_deterministic(self):
        cfg=contact.CONFIG.copy();cfg['duration']=.2
        a,_=contact.simulate(10,'learned',cfg);b,_=contact.simulate(10,'learned',cfg)
        np.testing.assert_allclose([r['x'] for r in a],[r['x'] for r in b],atol=0,rtol=0)
    def test_short_fourier_frozen(self):
        cfg=fourier_online.CONFIG.copy();cfg['duration']=.2
        rows=fourier_online.simulate(10,'frozen',cfg)
        for r in rows:np.testing.assert_array_equal([r[f'c{i}'] for i in range(6)],cfg['initial'])

if __name__=='__main__':unittest.main()
