"""
Test function related to frames.
"""
import proxddp
import numpy as np
import pinocchio as pin

from proxddp import manifolds


model = pin.buildSampleModelHumanoid()
rdata: pin.Data = model.createData()
space = manifolds.MultibodyConfiguration(model)

ndx = space.ndx
nq = model.nq
nu = model.nv


def test_frame_placement():
    fr_name1 = "larm_shoulder2_body"
    print(model.frames.tolist())
    fr_id1 = model.getFrameId(fr_name1)

    x0 = space.neutral()
    u0 = np.zeros(nu)
    q0 = x0[:nq]

    pin.framesForwardKinematics(model, rdata, q0)
    fr_plc1 = rdata.oMf[fr_id1]

    fun = proxddp.FramePlacementResidual(ndx, nu, model, fr_plc1, fr_id1)
    assert fr_id1 == fun.frame_id
    assert fr_plc1 == fun.getReference()

    fdata = fun.createData()
    fun.evaluate(x0, u0, x0, fdata)

    assert np.allclose(fdata.value, 0.0)
    print(fdata.value)

    print("Error:", fdata.rMf)
    print("pindata from fdata:")
    pdata_f = fdata.pin_data
    print(pdata_f.oMf[fr_id1])


if __name__ == "__main__":
    import pytest
    import sys

    retcode = pytest.main(sys.argv)
