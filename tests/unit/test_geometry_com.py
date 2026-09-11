import numpy as np
import pytest

from dvi.geometry.com import estimate_com
from tests.fixtures.synthetic.skeletons import build_synthetic_standing_pose


class TestEstimateCom:
    def test_com_lies_between_feet_and_head(self):
        pose = build_synthetic_standing_pose()
        com, valid = estimate_com(pose)

        ankle_y = pose.joints[0, pose.skeleton.joint_index("ankle_l"), 1]
        head_y = pose.joints[0, pose.skeleton.joint_index("head_top"), 1]

        assert valid[0]
        assert ankle_y < com[0, 1] < head_y

    def test_symmetric_standing_pose_has_com_near_midline(self):
        pose = build_synthetic_standing_pose()  # no roll -> left/right symmetric
        com, valid = estimate_com(pose)
        assert com[0, 0] == pytest.approx(0.0, abs=1e-9)  # X (left-right) midline

    def test_trunk_dominates_so_com_is_near_trunk_not_limbs(self):
        # Trunk alone is ~50% of body mass; COM should sit well within the
        # pelvis-neck span, not drift out toward a limb extremity.
        pose = build_synthetic_standing_pose()
        com, valid = estimate_com(pose)
        pelvis_y = pose.joints[0, pose.skeleton.joint_index("pelvis"), 1]
        neck_y = pose.joints[0, pose.skeleton.joint_index("neck"), 1]
        assert pelvis_y < com[0, 1] < neck_y

    def test_partial_joint_invalidity_still_produces_an_estimate(self):
        pose = build_synthetic_standing_pose()
        valid = pose.valid.copy()
        # knock out one arm entirely; COM should still be computable from the rest
        valid[0, pose.skeleton.joint_index("wrist_l")] = False
        valid[0, pose.skeleton.joint_index("elbow_l")] = False
        pose = pose.model_copy(update={"valid": valid})

        com, com_valid = estimate_com(pose)
        assert com_valid[0]
        assert not np.isnan(com[0]).any()

    def test_all_invalid_gives_nan_and_false(self):
        pose = build_synthetic_standing_pose()
        valid = np.zeros_like(pose.valid)
        pose = pose.model_copy(update={"valid": valid})

        com, com_valid = estimate_com(pose)
        assert not com_valid[0]
        assert np.isnan(com[0]).all()
