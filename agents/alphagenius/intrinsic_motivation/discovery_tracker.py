# agents/alphagenius/intrinsic_motivation/discovery_tracker.py

class DiscoveryTracker:
    """Simple novelty tracker used for intrinsic motivation."""

    def __init__(self):
        self.events = set()
        print("DiscoveryTracker initialized.")

    def add_event(self, name: str) -> bool:
        """Record a named event and return True if it was not seen before."""
        if name not in self.events:
            self.events.add(name)
            print(f"DiscoveryTracker: New event '{name}'.")
            return True
        return False

    def has_event(self, name: str) -> bool:
        return name in self.events

if __name__ == '__main__':
    tracker = DiscoveryTracker()
    tracker.add_event("hello")
    tracker.add_event("world")
    tracker.add_event("hello")  # Already known
    print(f"Has 'world' been seen? {tracker.has_event('world')}")
