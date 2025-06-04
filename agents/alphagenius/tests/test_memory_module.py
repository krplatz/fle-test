import json
from agents.alphagenius.memory.memory_module import MemoryModule


def test_memory_module_persistence(tmp_path):
    mem_file = tmp_path / "mem.jsonl"
    mem = MemoryModule(filepath=str(mem_file))
    mem.add_experience("task1", "script1", "out1", "", 1.0, 0.5, True)
    mem.add_experience("task2", "script2", "", "err", None, None, False)
    mem.save_experiences()

    loaded = MemoryModule(filepath=str(mem_file))
    assert len(loaded.experiences) == 2
    assert loaded.experiences[0]["sub_goal"] == "task1"
    assert loaded.experiences[1]["stderr"] == "err"
