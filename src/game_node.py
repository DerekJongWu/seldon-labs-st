class Node:
    """Represents a game state, supporting sequential moves."""
    def __init__(self, players=None):
        self.players = players if players else set()
        self.actions = {}  # Maps action names to child nodes
        self.payoff = None  # Stores outcome if terminal

    def add_action(self, action, child_node):
        """Adds an action leading to a child node."""
        self.actions[action] = child_node
