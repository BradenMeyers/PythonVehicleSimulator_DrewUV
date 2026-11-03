#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The pytest for 'test_yaw_wrap.py' can be run in the terminal using:
1) cd <path of the Python Vehicle Simulator installation>
2) pytest -k yaw_wrap -v
"""

import math
import numpy as np
import pytest
from python_vehicle_simulator.lib.gnc import attitudeEuler, ssa
from python_vehicle_simulator.vehicles import remus100, torpedo

D2R = math.pi / 180
DT = 0.05

# vehicle class and the number of actuator states in u_actual
VEHICLES = [
    pytest.param(torpedo, 5, id="torpedo"),
    pytest.param(remus100, 3, id="remus100"),
]


def run(vehicle, n_act, psi0_deg, ref_deg, T=300.0, psi_d0=None):
    """Closed-loop depth/heading simulation. Returns time histories of the
    yaw angle psi and the reference-model state psi_d."""
    v = vehicle("depthHeadingAutopilot", 0, ref_deg, 1000, 0, 0)
    eta = np.zeros(6)
    eta[5] = psi0_deg * D2R
    nu = np.array([1.5, 0, 0, 0, 0, 0], float)
    u_actual = np.zeros(n_act)
    # Start the reference model from the vehicle state (the vehicle class
    # defaults psi_d to 0, which is only consistent for an initial yaw of 0)
    v.psi_d = eta[5] if psi_d0 is None else psi_d0 * D2R

    psi, psi_d = [], []
    for _ in range(int(T / DT)):
        u_control = v.depthHeadingAutopilot(eta, nu, DT)
        nu, u_actual = v.dynamics(eta, nu, u_actual, u_control, DT)
        eta = attitudeEuler(eta, nu, DT)
        psi.append(eta[5])
        psi_d.append(v.psi_d)
    return np.array(psi), np.array(psi_d)


# (initial yaw, setpoint) in degrees. Includes pairs straddling +-180 where
# the long way round is more than 180 deg, and multi-turn setpoints.
CASES = [
    (170, -170),    # short way is +20 (through 180)
    (-170, 170),    # short way is -20 (through 180)
    (179, -179),
    (-179, 179),
    (10, 350),      # short way is -20 (through 0)
    (350, 10),      # initial yaw > 180
    (90, -90 + 360),
    (0, 90),
    (0, -90),
    (-135, 135),    # 90 deg, through 180
    (0, 720 + 30),  # setpoint many turns away: only 30 deg to go
    (720 + 5, 20),  # vehicle many turns away
]


@pytest.mark.parametrize("vehicle,n_act", VEHICLES)
@pytest.mark.parametrize("psi0,ref", CASES)
def test_yaw_wrap_takes_shortest_way_without_reversing(vehicle, n_act, psi0, ref):
    psi, psi_d = run(vehicle, n_act, psi0, ref)
    err0 = ssa((ref - psi0) * D2R)           # signed shortest error (rad)
    direction = np.sign(err0)

    # The reference state moves monotonically in the shortest direction
    # (tolerance for numerical noise), i.e. it never reverses.
    d_psi_d = np.diff(psi_d)
    assert np.all(direction * d_psi_d > -1e-6), "reference reversed direction"

    # Total vehicle yaw travel is the shortest error, not the long way
    # (small allowance for overshoot).
    travel = psi.max() - psi.min()
    assert travel < abs(err0) * 1.15 + 2 * D2R, \
        f"travelled {travel / D2R:.1f} deg, shortest is {abs(err0) / D2R:.1f}"

    # Vehicle never goes the wrong way past its start
    assert np.all(direction * (psi - psi[0]) > -2 * D2R)

    # Converges onto the setpoint (modulo 2 pi)
    assert abs(ssa(psi[-1] - ref * D2R)) < 1.0 * D2R


@pytest.mark.parametrize("vehicle,n_act", VEHICLES)
def test_yaw_wrap_exactly_opposite_commits_to_one_direction(vehicle, n_act):
    """With a 180 deg error either direction is 'shortest'. It must pick one
    and stick with it instead of dithering."""
    psi, psi_d = run(vehicle, n_act, 0.0, 180.0)
    d = np.diff(psi_d)
    moving = d[np.abs(d) > 1e-6]
    assert np.all(moving > 0) or np.all(moving < 0), "reference changed direction"
    assert abs(ssa(psi[-1] - math.pi)) < 1.0 * D2R


@pytest.mark.parametrize("vehicle,n_act", VEHICLES)
def test_yaw_wrap_setpoint_sweep_never_dithers(vehicle, n_act):
    """Sweep a grid of initial yaw / setpoint pairs and count sign changes of
    the commanded reference velocity: a well-behaved reference has at most
    one (accelerate, then decelerate to rest through zero is not a reversal)."""
    for psi0 in range(-180, 181, 45):
        for ref in range(-180, 181, 45):
            if abs(ssa((ref - psi0) * D2R)) < 1e-9:
                continue
            _, psi_d = run(vehicle, n_act, psi0, ref, T=200.0)
            d = np.diff(psi_d)
            d = d[np.abs(d) > 1e-6]
            sign_changes = np.sum(np.diff(np.sign(d)) != 0)
            assert sign_changes <= 1, (psi0, ref, sign_changes)
