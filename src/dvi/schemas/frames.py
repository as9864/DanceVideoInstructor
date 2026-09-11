"""Coordinate frames and role labels (ARCHITECTURE.md §4.1).

`WORLD_G` is the load-bearing concept in this project: camera-frame angles
are confounded with camera tilt and cannot answer "how far did they fall
over". Every posture/axis metric must require WORLD_G or BODY input.
"""

from enum import StrEnum


class Frame(StrEnum):
    IMAGE = "image"      # pixels, y-down
    CAMERA = "camera"    # camera-centric, meters, OpenCV convention
    WORLD_G = "world_g"  # gravity-aligned: +Y is up (away from gravity), XZ is the floor plane
    BODY = "body"        # pelvis-origin, pelvis-line yaw-aligned (for rotation-invariant metrics)


class Role(StrEnum):
    LEADER = "leader"
    FOLLOWER = "follower"
    UNKNOWN = "unknown"  # role assignment was attempted but inconclusive (ARCHITECTURE.md §5.3)
