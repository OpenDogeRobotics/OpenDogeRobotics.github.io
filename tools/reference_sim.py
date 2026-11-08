#!/usr/bin/env python3
"""Reference MuJoCo runner for the browser simulator's policies.

The browser build must reproduce the training-time control loop exactly:
48-field observation, 12 position actuators (kp=12, kd=0.5), action scale 0.25,
2 ms physics at decimation 2.  This script implements the same loop with the
Python MuJoCo bindings, so a new policy can be validated before it is added to
`public/policies/`.

Usage::

  python tools/reference_sim.py --policy public/policies/opendoge-flat.onnx \\
      --command 0.4 0 0 --duration 3

  python tools/reference_sim.py --policy public/policies/opendoge-getup.onnx \\
      --init fallen --duration 5

Requires `mujoco`, `numpy` and `onnxruntime` in the active Python environment.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

SITE = Path(__file__).resolve().parents[1]

JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
DEFAULT = np.array([0.0, 0.8, -1.6, 0.0, -0.8, 1.6, 0.0, 0.8, -1.6, 0.0, -0.8, 1.6])
EFFORT = [6.0, 6.0, 9.0] * 4
KP, KD, ARMATURE = 12.0, 0.5, 0.005
ACTION_SCALE = 0.25
DT, DECIMATION = 0.002, 2

# Rough policy height scan: mjlab GridPatternCfg(size=(0.7, 0.5), resolution=0.05).
SCAN_X = [-0.35 + index * 0.05 for index in range(15)]
SCAN_Y = [-0.25 + index * 0.05 for index in range(11)]
SCAN_MISS = 5.0
SCAN_SCALE = 1 / SCAN_MISS
SKILL_NAMES = ("walk", "handstand", "getup", "jump")


def build_model(xml: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.option.timestep = DT
    for joint in spec.joints:
        if joint.name == "floating_base_joint":
            continue
        joint.armature = ARMATURE
    for index, name in enumerate(JOINTS):
        act = spec.add_actuator(name=name, target=name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = KP
        act.biasprm[1] = -KP
        act.biasprm[2] = -KD
        act.ctrllimited = False
        act.forcelimited = True
        act.forcerange[:] = [-EFFORT[index], EFFORT[index]]
    for geom in spec.geoms:
        if geom.name.endswith("_foot_collision"):
            geom.condim = 3
            geom.priority = 1
            geom.friction[0] = 0.8
            geom.solimp[0:3] = [0.9, 0.95, 0.023]
        elif geom.name.endswith("_collision"):
            geom.condim = 1
            geom.conaffinity = 0
    return spec.compile()


def projected_gravity(data: mujoco.MjData) -> np.ndarray:
    w, x, y, z = data.qpos[3:7]
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return rotation.T @ np.array([0.0, 0.0, -1.0])


def height_scan(model, data) -> np.ndarray:
    body_id = int(model.body("base_link").id)
    position = np.array(data.xpos[body_id])
    rotation = np.array(data.xmat[body_id]).reshape(3, 3)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    cos, sin = math.cos(yaw), math.sin(yaw)
    geom_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    geom_id = np.zeros(1, dtype=np.int32)
    height = np.empty(len(SCAN_X) * len(SCAN_Y), dtype=np.float32)
    index = 0
    # torch.meshgrid(indexing="xy") flattens as y-outer, x-inner.
    for y in SCAN_Y:
        for x in SCAN_X:
            point = np.array([
                position[0] + cos * x - sin * y,
                position[1] + sin * x + cos * y,
                position[2],
            ])
            distance = mujoco.mj_ray(model, data, point, np.array([0.0, 0.0, -1.0]),
                                     geom_group, 1, body_id, geom_id)
            height[index] = SCAN_MISS if distance < 0 else distance
            index += 1
    return height * SCAN_SCALE


def observation(model, data, last_action, command) -> np.ndarray:
    return np.concatenate([
        data.sensor("imu_lin_vel").data,
        data.sensor("imu_ang_vel").data,
        projected_gravity(data),
        np.array([data.qpos[model.jnt(n).qposadr[0]] for n in JOINTS]) - DEFAULT,
        np.array([data.qvel[model.jnt(n).dofadr[0]] for n in JOINTS]),
        last_action,
        command,
    ]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=SITE / "public/policies/opendoge-flat.onnx")
    training = Path(os.environ.get("OPENDOGE_TRAIN", SITE.parent / "OpenDoge_train"))
    parser.add_argument(
        "--scene",
        type=Path,
        default=training / "src/assets/robots/opendoge/xmls/scene_opendoge.xml",
    )
    parser.add_argument("--command", type=float, nargs=3, default=[0.4, 0.0, 0.0],
                        metavar=("VX", "VY", "WZ"))
    parser.add_argument("--init", choices=("stand", "fallen"), default="stand")
    parser.add_argument("--init-scale", type=float, default=1.0,
                        help="Multiplier on the default joint pose (training reset range 0.85-1.15).")
    parser.add_argument("--skill", choices=SKILL_NAMES, default="walk",
                        help="Skill token for the 57-field multi-skill policy.")
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    model = build_model(args.scene)
    data = mujoco.MjData(model)
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    fallen = args.init == "fallen"
    data.qpos[0:3] = [0.0, 0.0, 0.06 if fallen else 0.158]
    data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0] if fallen else [1.0, 0.0, 0.0, 0.0]
    for index, name in enumerate(JOINTS):
        data.qpos[model.jnt(name).qposadr[0]] = DEFAULT[index] * args.init_scale
    mujoco.mj_forward(model, data)

    last_action = np.zeros(12)
    command = np.array(args.command)
    width = session.get_inputs()[0].shape[1]
    steps = int(args.duration / (DT * DECIMATION))

    for step in range(steps):
        shared = observation(model, data, last_action, command)
        if width == 213:
            obs = np.concatenate([shared, height_scan(model, data)])
        elif width == 57:
            token = np.zeros(len(SKILL_NAMES), dtype=np.float32)
            token[SKILL_NAMES.index(args.skill)] = 1.0
            # Jump twist is zero for the stationary skills; the phase clock only
            # advances for jump, so walk/handstand/getup see sin/cos = 0, 1.
            obs = np.concatenate([shared, np.zeros(3, dtype=np.float32), token,
                                  np.array([0.0, 1.0], dtype=np.float32)])
        else:
            obs = shared
        action = session.run(None, {input_name: obs[None]})[0][0]
        last_action = action
        target = DEFAULT + ACTION_SCALE * action
        for _ in range(DECIMATION):
            data.ctrl[:] = target
            mujoco.mj_step(model, data)
        if step % 100 == 0:
            print(f"t={step * DT * DECIMATION:5.2f}s x={data.qpos[0]:.3f} "
                  f"y={data.qpos[1]:.3f} z={data.qpos[2]:.3f} vx={data.qvel[0]:.3f}")

    print(f"final x={data.qpos[0]:.3f} y={data.qpos[1]:.3f} z={data.qpos[2]:.3f}")


if __name__ == "__main__":
    sys.exit(main())
