"""Four comparators used by the evaluations harness."""
from .firmware_behavior import compare as firmware_behavior_compare
from .hardware_diff import compare as hardware_diff_compare
from .simulation_diff import compare as simulation_diff_compare
from .spec_coverage import compare as spec_coverage_compare

__all__ = [
    "spec_coverage_compare",
    "hardware_diff_compare",
    "firmware_behavior_compare",
    "simulation_diff_compare",
]
