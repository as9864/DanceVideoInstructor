"""Real-time path (ARCHITECTURE.md §11): One-Euro filter, 2D metrics,
feedback hysteresis, and LiveSession/run_live end to end with a fake pose
estimator — no camera, GPU, or model weights needed."""

import numpy as np
import pytest

from dvi.config.loader import load_live_messages, load_style_targets
from dvi.geometry.gravity import manual_calibration_from_vertical_line
from dvi.geometry.smoothing import OneEuroFilter
from dvi.perception.pose.skeleton import COCO17_SKELETON_SPEC
from dvi.pipeline.stream_runner import run_live
from dvi.realtime.display import NullDisplay
from dvi.realtime.feedback import FeedbackEngine, judge
from dvi.realtime.hud import render_hud
from dvi.realtime.metrics2d import (
    CameraView,
    Keypoints2D,
    LiveCalibration,
    axi04_2d,
    axi06_2d,
    compute_live_metrics,
    facing_sign,
    pos01_2d,
    pos02_2d,
    pos04_2d,
    pos05a_2d,
    pos05b_2d,
    shoulder_ear_ratio,
    support_side,
)
from dvi.realtime.session import LiveSession, box_from_keypoints
from dvi.schemas.metrics import TargetRange

JOINTS = COCO17_SKELETON_SPEC.joint_names
TORSO = 200.0  # px, hip_mid -> neck


def _rot_up(angle_deg: float, length: float) -> np.ndarray:
    """Image-space (y-down) vector of `length` at `angle_deg` from up, + toward image-right."""
    a = np.radians(angle_deg)
    return np.array([np.sin(a), -np.cos(a)]) * length


def make_side_pose(
    lean_deg: float = 0.0,
    facing: int = 1,
    head_forward: float = 0.0,
    shrug: float = 0.0,
    knee_deg: float = 175.0,
    ankle_offset: float = 0.0,
    image_roll_deg: float = 0.0,
) -> Keypoints2D:
    """Profile view, dancer facing image-right (facing=+1) or left (-1).

    `lean_deg`: forward torso pitch. `head_forward`: ear ahead of the
    shoulder, in torso lengths. `shrug`: fraction the shoulder-ear distance
    shrinks. `ankle_offset`: support ankle's horizontal offset from the hip
    (px, + = behind the dancer). `image_roll_deg`: rotates the whole image
    as if the camera were tilted.
    """
    p: dict[str, np.ndarray] = {}
    hip = np.array([400.0, 500.0])
    neck = hip + _rot_up(facing * lean_deg, TORSO)
    for s in ("l", "r"):
        p[f"hip_{s}"] = hip.copy()
        p[f"shoulder_{s}"] = neck.copy()
    ear_dist = 0.35 * TORSO * (1 - shrug)
    ear = neck + np.array([facing * head_forward * TORSO, -ear_dist])
    p["ear_l"] = p["ear_r"] = ear
    p["nose"] = ear + np.array([facing * 25.0, 5.0])
    # support leg: straight down from hip to ankle with a knee angle
    ankle = hip + np.array([-facing * ankle_offset, 2.2 * TORSO])
    leg = ankle - hip
    mid = hip + leg / 2
    half = np.linalg.norm(leg) / 2
    bend = half / np.tan(np.radians(knee_deg / 2))  # knee pushed forward to make the angle
    perp = np.array([-leg[1], leg[0]]) / np.linalg.norm(leg) * -facing
    knee = mid + perp * bend
    for s in ("l", "r"):
        p[f"knee_{s}"] = knee
        p[f"ankle_{s}"] = ankle
    p["elbow_l"] = p["elbow_r"] = neck + np.array([facing * 60.0, 60.0])
    p["wrist_l"] = p["wrist_r"] = neck + np.array([facing * 110.0, 40.0])
    p["pelvis"] = hip
    p["neck"] = neck
    p["spine_mid"] = (hip + neck) / 2

    xy = np.array([p[n] for n in JOINTS])
    if image_roll_deg:
        r = np.radians(image_roll_deg)
        # rotate about the hip, clockwise on screen for positive roll (y-down)
        c, s = np.cos(r), np.sin(r)
        d = xy - hip
        xy = hip + np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)
    return Keypoints2D(joint_names=JOINTS, xy=xy, valid=np.ones(len(JOINTS), dtype=bool))


def make_front_pose(lean_deg: float = 0.0, shoulder_tilt_deg: float = 0.0, stance: float = 15.0) -> Keypoints2D:
    """Facing the camera, dancer's left on image-right (no mirroring)."""
    p: dict[str, np.ndarray] = {}
    hip = np.array([400.0, 500.0])
    neck = hip + _rot_up(lean_deg, TORSO)
    half_sh = 70.0
    t = np.radians(shoulder_tilt_deg)
    sh_vec = np.array([np.cos(t), -np.sin(t)]) * half_sh  # + tilt raises the image-right shoulder
    p["shoulder_l"], p["shoulder_r"] = neck + sh_vec, neck - sh_vec
    p["hip_l"], p["hip_r"] = hip + [40.0, 0.0], hip - [40.0, 0.0]
    p["ear_l"], p["ear_r"] = neck + [20.0, -70.0], neck + [-20.0, -70.0]
    p["nose"] = neck + [0.0, -60.0]
    for s, sign in (("l", 1), ("r", -1)):
        p[f"knee_{s}"] = hip + [sign * stance, 220.0]
        p[f"ankle_{s}"] = hip + [sign * stance, 440.0]
        p[f"elbow_{s}"] = p[f"shoulder_{s}"] + [sign * 30.0, 100.0]
        p[f"wrist_{s}"] = p[f"elbow_{s}"] + [0.0, 90.0]
    p["pelvis"], p["neck"], p["spine_mid"] = hip, neck, (hip + neck) / 2
    xy = np.array([p[n] for n in JOINTS])
    return Keypoints2D(joint_names=JOINTS, xy=xy, valid=np.ones(len(JOINTS), dtype=bool))


CAL = LiveCalibration()


class TestOneEuro:
    def test_constant_signal_passes_through(self):
        f = OneEuroFilter()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        for i in range(10):
            out = f(x, i / 30)
        np.testing.assert_allclose(out, x)

    def test_reduces_jitter(self):
        rng = np.random.default_rng(0)
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        raw = 0.5 + rng.normal(0, 0.01, size=300)
        out = np.array([f(np.array([v]), i / 30)[0] for i, v in enumerate(raw)])
        assert np.std(out[30:]) < 0.5 * np.std(raw[30:])

    def test_masked_joint_restarts_from_raw(self):
        f = OneEuroFilter()
        f(np.array([[0.0, 0.0]]), 0.0)
        f(np.array([[0.0, 0.0]]), 1 / 30, mask=np.array([False]))
        out = f(np.array([[5.0, 5.0]]), 2 / 30, mask=np.array([True]))
        np.testing.assert_allclose(out, [[5.0, 5.0]])


class TestMetrics2D:
    @pytest.mark.parametrize("lean", [0.0, 5.0, -8.0, 15.0])
    @pytest.mark.parametrize("facing", [1, -1])
    def test_pos01_reads_forward_lean_either_facing(self, lean, facing):
        kp = make_side_pose(lean_deg=lean, facing=facing)
        assert facing_sign(kp) == facing
        assert pos01_2d(kp, CAL) == pytest.approx(lean, abs=1e-6)

    def test_roll_calibration_undoes_camera_tilt(self):
        kp = make_side_pose(lean_deg=4.0, image_roll_deg=6.0)
        assert pos01_2d(kp, CAL) != pytest.approx(4.0, abs=0.5)
        # a door frame as seen through the tilted camera -> the manual-vertical fallback
        r = np.radians(6.0)
        a, b = (100.0, 600.0), (100.0 + 300 * np.sin(r), 600.0 - 300 * np.cos(r))
        roll = manual_calibration_from_vertical_line(a, b).metadata["roll_deg"]
        assert pos01_2d(kp, LiveCalibration(roll_deg=roll)) == pytest.approx(4.0, abs=1e-6)

    @pytest.mark.parametrize("facing", [1, -1])
    def test_pos04_head_forward(self, facing):
        assert pos04_2d(make_side_pose(head_forward=0.0, facing=facing), CAL) == pytest.approx(0.0, abs=1e-9)
        assert pos04_2d(make_side_pose(head_forward=0.2, facing=facing), CAL) == pytest.approx(0.2, abs=1e-9)

    def test_pos05a_relative_to_baseline(self):
        baseline = shoulder_ear_ratio(make_side_pose())
        cal = LiveCalibration(shrug_baseline=baseline)
        assert pos05a_2d(make_side_pose(), cal) == pytest.approx(0.0, abs=1e-9)
        assert pos05a_2d(make_side_pose(shrug=0.2), cal) == pytest.approx(0.2, abs=1e-9)
        assert pos05a_2d(make_side_pose(), CAL) is None  # not calibrated yet

    @pytest.mark.parametrize("knee", [160.0, 172.0, 179.0])
    def test_axi04_knee_angle(self, knee):
        assert axi04_2d(make_side_pose(knee_deg=knee), CAL) == pytest.approx(knee, abs=1e-6)

    def test_axi06_axis_offset(self):
        assert axi06_2d(make_side_pose(), CAL) == pytest.approx(0.0, abs=1e-6)
        kp = make_side_pose(ankle_offset=40.0)  # ankle behind the hips -> body ahead of the foot
        expected = np.degrees(np.arctan2(40.0, 2.2 * TORSO + TORSO / 2))
        assert abs(axi06_2d(kp, CAL)) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("lean", [0.0, 4.0, -6.0])
    def test_front_view_side_lean_and_shoulders(self, lean):
        kp = make_front_pose(lean_deg=lean, shoulder_tilt_deg=5.0)
        assert pos02_2d(kp, CAL) == pytest.approx(lean, abs=1e-6)
        assert pos05b_2d(kp, CAL) == pytest.approx(5.0, abs=1e-6)
        assert support_side(kp) == "both"

    def test_missing_joints_report_none(self):
        kp = make_side_pose()
        valid = kp.valid.copy()
        valid[JOINTS.index("hip_l")] = False
        kp = Keypoints2D(JOINTS, kp.xy, valid)
        values = compute_live_metrics(kp, CameraView.SIDE, CAL)
        assert values["POS-01"] is None
        assert set(values) == {"AXI-06", "POS-01", "POS-04", "POS-05a", "AXI-04"}


TARGETS = load_style_targets("salon")
MESSAGES = load_live_messages()


class TestFeedback:
    def test_judge_bands(self):
        t = TargetRange(unit="deg", min=0.0, max=5.0, tolerance=2.0)
        assert judge("POS-01", 3.0, t) == ("ok", None)
        assert judge("POS-01", 6.5, t) == ("minor", "high")
        assert judge("POS-01", 9.0, t) == ("major", "high")
        assert judge("POS-01", -3.0, t) == ("major", "low")
        assert judge("POS-02", -9.0, TargetRange(unit="deg", max=3.0)) == ("major", "high")
        assert judge("POS-01", None, t) == ("unknown", None)

    def test_alert_needs_hold_time_and_releases(self):
        eng = FeedbackEngine(TARGETS, MESSAGES, hold_s=0.3, release_s=0.5, cooldown_s=4.0)
        fps = 30
        fired = []
        for i in range(fps):  # 1s of heavy forward lean
            up = eng.update({"POS-01": 15.0}, i / fps)
            fired += up.cues
            if i / fps < 0.29:
                assert up.headline is None
        assert len(fired) == 1 and fired[0].metric_id == "POS-01"
        assert "앞으로" in up.headline.message
        # back in range: alert clears only after release_s
        for i in range(fps, 2 * fps):
            up = eng.update({"POS-01": 2.0}, i / fps)
        assert up.headline is None

    def test_single_frame_spike_ignored(self):
        eng = FeedbackEngine(TARGETS, MESSAGES)
        cues = []
        for i in range(60):
            cues += eng.update({"POS-01": 30.0 if i == 10 else 2.0}, i / 30).cues
        assert cues == []

    def test_cue_cooldown(self):
        eng = FeedbackEngine(TARGETS, MESSAGES, hold_s=0.1, release_s=0.1, cooldown_s=4.0)
        cues = []
        t = 0.0
        for _ in range(3):  # three bad/good cycles within 4s
            for _ in range(10):
                cues += eng.update({"POS-01": 15.0}, t).cues
                t += 1 / 30
            for _ in range(10):
                cues += eng.update({"POS-01": 2.0}, t).cues
                t += 1 / 30
        assert len(cues) == 1

    @pytest.mark.parametrize("mirrored, expected", [(True, "오른쪽"), (False, "왼쪽")])
    def test_side_words_follow_mirroring(self, mirrored, expected):
        eng = FeedbackEngine(TARGETS, MESSAGES, hold_s=0.0, mirrored=mirrored)
        up = eng.update({"POS-02": 10.0}, 0.0)  # + = toward image-right
        assert up.headline.message.startswith(f"상체가 {expected}")

    def test_priority_axis_first(self):
        eng = FeedbackEngine(TARGETS, MESSAGES, hold_s=0.0)
        up = eng.update({"POS-05a": 0.5, "AXI-06": 12.0, "POS-01": 15.0}, 0.0)
        assert up.headline.metric_id == "AXI-06"


class FakeEstimator:
    """Plays back a scripted sequence of synthetic poses."""

    joint_names = JOINTS

    def __init__(self, poses: list[Keypoints2D | None]):
        self.poses = poses
        self.i = 0
        self.boxes = []

    def estimate(self, image_rgb, box):
        self.boxes.append(box)
        kp = self.poses[min(self.i, len(self.poses) - 1)]
        self.i += 1
        if kp is None:
            return np.zeros((len(JOINTS), 2)), np.zeros(len(JOINTS))
        return kp.xy.copy(), np.full(len(JOINTS), 0.9)


class FakeSource:
    def __init__(self, n: int, fps: float = 30.0):
        self.n, self.fps, self.i = n, fps, 0

    def read(self, timeout_s: float = 2.0):
        if self.i >= self.n:
            return None
        self.i += 1
        return np.zeros((720, 800, 3), dtype=np.uint8), (self.i - 1) / self.fps

    def now(self):
        return (self.i - 1) / self.fps

    def close(self):
        pass


class TestLiveSession:
    def _session(self, poses, **kw):
        kw.setdefault("calibration_s", 1.0)
        return LiveSession(FakeEstimator(poses), TARGETS, MESSAGES, **kw)

    def test_calibrates_then_warns(self):
        # 1s neutral standing, then 1s of heavy forward lean
        poses = [make_side_pose()] * 30 + [make_side_pose(lean_deg=18.0)] * 30
        session = self._session(poses)
        results = [session.process(np.zeros((720, 800, 3), np.uint8), i / 30) for i in range(60)]
        assert results[0].calibration_progress is not None and results[0].feedback is None
        assert session.calibration.shrug_baseline == pytest.approx(0.35, rel=1e-6)
        calibrated = [r for r in results if r.calibration_progress is None]
        assert calibrated and calibrated[0].feedback.headline is None
        final = results[-1].feedback
        assert final.headline is not None and final.headline.metric_id in ("POS-01", "AXI-06")
        assert final.metrics["POS-01"].status == "major"
        assert final.metrics["POS-01"].value > 10  # One-Euro converged toward 18°

    def test_tracks_box_from_previous_pose(self):
        poses = [make_side_pose()] * 3
        est = FakeEstimator(poses)
        session = LiveSession(est, TARGETS, MESSAGES)
        for i in range(3):
            session.process(np.zeros((720, 800, 3), np.uint8), i / 30)
        assert est.boxes[0] is None  # first frame: whole image
        assert est.boxes[1] is not None
        x, y, w, h = est.boxes[1]
        assert 0 <= x and 0 <= y and x + w <= 800 and y + h <= 720

    def test_nobody_in_frame(self):
        session = self._session([None] * 5, calibration_s=0.0)
        r = session.process(np.zeros((720, 800, 3), np.uint8), 0.0)
        assert r.keypoints is None and r.box is None

    def test_box_from_keypoints_needs_enough_points(self):
        kp = make_side_pose()
        valid = np.zeros(len(JOINTS), bool)
        valid[:3] = True
        assert box_from_keypoints(kp.xy, valid, (800, 720)) is None


class TestRunLive:
    def test_end_to_end_with_recording(self, tmp_path):
        poses = [make_side_pose()] * 30 + [make_side_pose(lean_deg=20.0)] * 30
        session = LiveSession(FakeEstimator(poses), TARGETS, MESSAGES, calibration_s=0.5)
        cues = []
        out = tmp_path / "live.mp4"
        stats = run_live(FakeSource(60), session, NullDisplay(), record_path=out, on_cue=cues.append)
        assert stats.frames == 60
        assert cues and stats.cues
        assert out.exists() and out.stat().st_size > 0

    def test_hud_renders_all_states(self):
        img = np.zeros((720, 1280, 3), np.uint8)
        eng = FeedbackEngine(TARGETS, MESSAGES, hold_s=0.0)
        kp = make_side_pose(lean_deg=15.0)
        fb = eng.update(compute_live_metrics(kp, CameraView.SIDE, LiveCalibration(shrug_baseline=0.35)), 0.0)
        for args in [
            dict(kp=kp, feedback=fb),
            dict(kp=kp, feedback=None, calibration_progress=0.4),
            dict(kp=None, feedback=None),
        ]:
            out = render_hud(img, args["kp"], args["feedback"], TARGETS, header="salon · side",
                             calibration_progress=args.get("calibration_progress"), fps=30.0, latency_ms=45.0)
            assert out.shape == img.shape and out.any()
        assert not img.any()  # input untouched
