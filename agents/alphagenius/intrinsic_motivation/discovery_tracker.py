# agents/alphagenius/intrinsic_motivation/discovery_tracker.py

class DiscoveryTracker:
    """
    Tracks new discoveries, such as items crafted or technologies researched for the first time.
    This is a basic component for intrinsic motivation.
    """
    def __init__(self):
        self.discovered_items_crafted = set()
        self.discovered_technologies_researched = set()
        # Could add more sets for other types of discoveries, e.g., areas explored.
        print("DiscoveryTracker initialized.")

    def add_item_crafted(self, item_name: str) -> bool:
        """
        Records that an item has been crafted.
        Returns True if the item is a new discovery, False otherwise.
        """
        if item_name not in self.discovered_items_crafted:
            self.discovered_items_crafted.add(item_name)
            print(f"DiscoveryTracker: New item crafted - '{item_name}'! (Intrinsic Reward Hint)")
            return True
        print(f"DiscoveryTracker: Item '{item_name}' crafted again (already known).")
        return False

    def check_item_crafted(self, item_name: str) -> bool:
        """
        Checks if an item has been recorded as crafted.
        """
        return item_name in self.discovered_items_crafted

    def add_technology_researched(self, tech_name: str) -> bool:
        """
        Records that a technology has been researched.
        Returns True if the technology is a new discovery, False otherwise.
        """
        if tech_name not in self.discovered_technologies_researched:
            self.discovered_technologies_researched.add(tech_name)
            print(f"DiscoveryTracker: New technology researched - '{tech_name}'! (Intrinsic Reward Hint)")
            return True
        print(f"DiscoveryTracker: Technology '{tech_name}' researched again (already known).")
        return False

    def check_technology_researched(self, tech_name: str) -> bool:
        """
        Checks if a technology has been recorded as researched.
        """
        return tech_name in self.discovered_technologies_researched

if __name__ == '__main__':
    tracker = DiscoveryTracker()
    tracker.add_item_crafted("iron-plate")
    tracker.add_item_crafted("copper-cable")
    tracker.add_item_crafted("iron-plate") # Already known
    
    tracker.add_technology_researched("automation")
    tracker.add_technology_researched("automation") # Already known
    print(f"Has iron-plate been crafted? {tracker.check_item_crafted('iron-plate')}")
    print(f"Has steel-plate been crafted? {tracker.check_item_crafted('steel-plate')}")
