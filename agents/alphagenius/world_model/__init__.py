class WorldModel:
    """A minimal world model for storing simple key-value facts."""

    def __init__(self):
        self._facts = {}

    def set_fact(self, key, value):
        """Stores or updates a fact."""
        self._facts[key] = value

    def get_fact(self, key, default=None):
        """Retrieves a fact by key."""
        return self._facts.get(key, default)

    def all_facts(self):
        """Returns a dictionary of all known facts."""
        return dict(self._facts)
