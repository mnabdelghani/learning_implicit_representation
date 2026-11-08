"""Supplementary excitation protocol checks, independent of main-paper runs."""
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import contact,contact_excitation

class ExcitationTests(unittest.TestCase):
 def test_probing_is_bounded_and_zero_at_freeze(self):
  t=np.linspace(0,60,601)
  p=np.array([contact_excitation.probing(x) for x in t])
  self.assertTrue(np.all(np.isfinite(p)))
  self.assertLessEqual(np.max(np.abs(p[:,0])),.12+1e-12)
  self.assertLessEqual(np.max(np.abs(p[:,1])),.02+1e-12)
 def test_task_reference_changes_only_at_transfer(self):
  self.assertEqual(contact_excitation.reference(59),(0.,0.,4.,0.))
  self.assertEqual(contact_excitation.reference(60),contact.refs(0.))
 def test_frozen_parameters_remain_constant(self):
  cfg=contact.CONFIG.copy();cfg.update(duration=.3,change_time=1e9)
  rows,_=contact.simulate(10,'learned',cfg,reference=contact_excitation.reference,
    probe=contact_excitation.probing,freeze_after=.1,extra_diagnostics=True)
  names=['s1','s2','a','r1','r2','b','B11','B12','B21','B22']
  frozen=np.array([[x[k] for k in names] for x in rows if x['time']>=.1])
  np.testing.assert_array_equal(frozen,np.broadcast_to(frozen[0],frozen.shape))
if __name__=='__main__':unittest.main()
