"""Per-task check functions, collected into one dispatch map."""

from .digit_span import DS_COLS, check_digit_span
from .gng import GNG_COLS, check_gng
from .maze import MAZE_COLS, MAZE_TIMEOUT_MS, check_maze
from .SoA import SOA_COLS, SOA_TIMEOUT_MS, check_SoA_part_two
from .stroop import STROOP_COLS, STROOP_TIMEOUT_MS, check_stroop

CHECK_TASK_MAP = {
    "Maze": check_maze,
    "GNG": check_gng,
    "Stroop": check_stroop,
    "DigitSpan": check_digit_span,
    "SoA": check_SoA_part_two,
}

# raw variables each task owns -- used to map a column back to its task
# (e.g. for unexpected-value issues surfaced during preprocessing)
TASK_COLUMN_GROUPS = {
    "Maze": MAZE_COLS,
    "GNG": GNG_COLS,
    "Stroop": STROOP_COLS,
    "DigitSpan": DS_COLS,
    "SoA": SOA_COLS,
}

__all__ = [
    "CHECK_TASK_MAP",
    "TASK_COLUMN_GROUPS",
    "check_maze",
    "check_gng",
    "check_stroop",
    "check_digit_span",
    "check_SoA_part_two",
    "MAZE_COLS",
    "MAZE_TIMEOUT_MS",
    "STROOP_TIMEOUT_MS",
    "GNG_COLS",
    "STROOP_COLS",
    "DS_COLS",
    "SOA_COLS",
    "SOA_TIMEOUT_MS",
]
