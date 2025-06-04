import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.alphagenius.memory.memory_module import MemoryModule


def test_memory_module_persistence(tmp_path):
    filepath = tmp_path / "mem.jsonl"
    memory = MemoryModule(filepath=str(filepath))

    memory.add_experience(
        sub_goal="goal1",
        script="script1",
        stdout="output1",
        stderr="",
        fle_score=1.0,
        novelty_score=0.5,
        success=True,
    )
    memory.add_experience(
        sub_goal="goal2",
        script="script2",
        stdout="output2",
        stderr="error",
        fle_score=0.0,
        novelty_score=0.2,
        success=False,
    )

    new_memory = MemoryModule()
    loaded = new_memory.load_experiences(str(filepath))

    assert loaded == 2
    assert new_memory.experiences == memory.experiences


