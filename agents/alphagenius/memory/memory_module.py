import json
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

class MemoryModule:
    """
    A simple module to store and retrieve experiences for AlphaGenius.
    V1: In-memory list, optional JSON file persistence (append-only for saves during a run).
    """
    def __init__(self, filepath: Optional[str] = None):
        self.experiences: List[Dict[str, Any]] = []
        self.filepath = filepath
        if self.filepath and os.path.exists(self.filepath):
            self.load_experiences(self.filepath)
        print(f"MemoryModule initialized. Loaded {len(self.experiences)} experiences from {self.filepath if self.filepath else 'in-memory only'}.")

    def add_experience(self, 
                       sub_goal: str, 
                       script: str, 
                       stdout: str, 
                       stderr: str, 
                       fle_score: Optional[float], # Can be None if not applicable
                       novelty_score: Optional[float], # Can be None
                       success: bool,
                       timestamp: Optional[str] = None):
        """Adds an experience to the memory."""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        
        experience = {
            "timestamp": timestamp,
            "sub_goal": sub_goal,
            "script": script,
            "stdout": stdout,
            "stderr": stderr,
            "fle_score": fle_score,
            "novelty_score": novelty_score,
            "success": success
        }
        self.experiences.append(experience)
        # print(f"MemoryModule: Added experience for sub-goal '{sub_goal}'. Success: {success}. FLE Score: {fle_score}, Novelty: {novelty_score}")

        if self.filepath:
            # Append to file immediately (simple V1 persistence)
            try:
                with open(self.filepath, 'a') as f:
                    f.write(json.dumps(experience) + '\n')
            except IOError as e:
                print(f"MemoryModule: Error appending experience to file {self.filepath}: {e}")

    def load_experiences(self, filepath: str):
        """Loads experiences from a JSONL file."""
        self.filepath = filepath # Ensure filepath is set if called directly
        loaded_count = 0
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        self.experiences.append(json.loads(line))
                        loaded_count +=1
                    except json.JSONDecodeError:
                        print(f"MemoryModule: Skipping malformed line in {filepath}: {line.strip()}")
            # print(f"MemoryModule: Successfully loaded {loaded_count} experiences from {filepath}.")
        except FileNotFoundError:
            print(f"MemoryModule: File {filepath} not found. Starting with empty memory.")
        except IOError as e:
            print(f"MemoryModule: Error loading experiences from {filepath}: {e}")
        return loaded_count

    def save_experiences(self, filepath: Optional[str] = None):
        """Saves all current in-memory experiences to a JSONL file, overwriting it."""
        path_to_save = filepath if filepath is not None else self.filepath
        if not path_to_save:
            print("MemoryModule: No filepath specified, cannot save experiences.")
            return 0
        
        self.filepath = path_to_save # Update current filepath if a new one is given
        try:
            with open(path_to_save, 'w') as f:
                for experience in self.experiences:
                    f.write(json.dumps(experience) + '\n')
            print(f"MemoryModule: Successfully saved {len(self.experiences)} experiences to {path_to_save}.")
            return len(self.experiences)
        except IOError as e:
            print(f"MemoryModule: Error saving experiences to {path_to_save}: {e}")
            return 0

    def get_recent_experiences(self, count: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent 'count' experiences."""
        return self.experiences[-count:]

if __name__ == '__main__':
    # Test MemoryModule
    mem_file = "test_memory.jsonl"
    if os.path.exists(mem_file): os.remove(mem_file)

    memory = MemoryModule(filepath=mem_file)
    memory.add_experience("mine iron", "script1", "out1", "", 10.0, 1.0, True)
    memory.add_experience("craft furnace", "script2", "out2", "error1", -1.0, 0.0, False)
    
    print(f"Recent experiences: {memory.get_recent_experiences(5)}")
    
    # Test saving and loading
    memory.save_experiences() # Saves to mem_file
    
    new_memory = MemoryModule(filepath=mem_file) # Loads on init
    print(f"Experiences in new_memory (loaded): {len(new_memory.experiences)}")
    assert len(new_memory.experiences) == 2
    
    new_memory.add_experience("research automation", "script3", "out3", "", 5.0, 1.0, True)
    # new_memory.save_experiences() # Appends via add_experience, save_experiences overwrites

    print(f"Final experiences in new_memory: {len(new_memory.experiences)}")
    if os.path.exists(mem_file): os.remove(mem_file)
```
