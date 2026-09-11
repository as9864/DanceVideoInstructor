"""DVI-Canonical skeleton (ARCHITECTURE.md §4.3, ADR-0001 §2 C1 / §7 D3).

Backends disagree on joint sets (COCO-17, H36M-17, SMPL-24, BlazePose-33,
Halpe-26...). Every backend adapter maps into this one canonical joint list
so downstream code never has to know which backend produced a pose.

ADR-0001 C1: COCO-17 and H36M-17 both lack foot keypoints, which several
AXI-* metrics need (heel/toe for ground-contact and support-point geometry).
The canonical set below is the Halpe-26-compatible minimum that the posture
and axis metrics actually require — not the full 26-point format.
"""

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DVI_CANONICAL_JOINTS: tuple[str, ...] = (
    "pelvis",
    "spine_mid",
    "neck",
    "head_top",
    "nose",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
    "foot_index_l",
    "foot_index_r",
    "heel_l",
    "heel_r",
)

# Joints with no direct keypoint in most backends; adapters fill these via
# derivation rules (e.g. spine_mid = (pelvis + neck) / 2, ARCHITECTURE.md §4.3).
DEFAULT_DERIVED_JOINTS: frozenset[str] = frozenset({"spine_mid"})


class SkeletonSpec(BaseModel):
    """Which canonical joints a given backend/adapter actually populates.

    `derived_joints` marks joints filled by a derivation rule rather than
    a direct model keypoint — metrics depending on them should discount
    confidence accordingly (ARCHITECTURE.md §4.3).
    """

    model_config = ConfigDict(frozen=True)

    name: str  # e.g. "dvi-canonical-v1"
    joint_names: tuple[str, ...] = DVI_CANONICAL_JOINTS
    derived_joints: frozenset[str] = DEFAULT_DERIVED_JOINTS

    @field_validator("joint_names")
    @classmethod
    def _joints_are_canonical_and_unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate joint names: {v}")
        unknown = set(v) - set(DVI_CANONICAL_JOINTS)
        if unknown:
            raise ValueError(f"joints not in DVI-Canonical: {sorted(unknown)}")
        return v

    @model_validator(mode="after")
    def _derived_subset_of_joints(self) -> "SkeletonSpec":
        unknown = self.derived_joints - set(self.joint_names)
        if unknown:
            raise ValueError(f"derived_joints not in joint_names: {sorted(unknown)}")
        return self

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def joint_index(self, name: str) -> int:
        return self.joint_names.index(name)

    def is_derived(self, name: str) -> bool:
        return name in self.derived_joints


DVI_CANONICAL_V1 = SkeletonSpec(name="dvi-canonical-v1")
