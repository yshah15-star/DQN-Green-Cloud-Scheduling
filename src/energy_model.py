"""
Energy model (utilization-based linear power model).

    P(u) = P_idle + (P_max - P_idle) * u      when the server is awake (has work)
    P    = P_sleep                            when the server has no running or queued task

Energy = integral of power over time (joules); reported in kWh.
This is the single place where energy is calculated.
"""

J_PER_KWH = 3.6e6


def server_power(server_type, utilization, awake):
    """Instantaneous power draw (W) of one server."""
    if not awake:
        return server_type.p_sleep
    return server_type.p_idle + (server_type.p_max - server_type.p_idle) * utilization


def energy_joules(power_w, seconds):
    return power_w * seconds


def to_kwh(joules):
    return joules / J_PER_KWH


def marginal_energy(server_type, cpu, exec_time, awake_extension):
    """
    Estimated extra energy (J) caused by placing one task on a server:
      dynamic part: the task's share of (P_max - P_idle) for its run time
      static part : the extra time the server must stay awake instead of sleeping
    Used for the DQN's per-step reward and features.
    """
    dynamic = (server_type.p_max - server_type.p_idle) * (cpu / server_type.cores) * exec_time
    static = (server_type.p_idle - server_type.p_sleep) * max(0.0, awake_extension)
    return dynamic + static
