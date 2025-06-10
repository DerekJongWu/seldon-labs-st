import sys
sys.path.append("../")
from src.game import Game

class Solver:
    def __init__(self, game):
        """Initialize with a game instance."""
        if not isinstance(game, Game):
            raise TypeError("Solver expects an instance of Game.")
        self.game = game
        self.equilibrium = {}

    def solve(self):
        """Base method to be overridden by specific solvers."""
        raise NotImplementedError("Solve method must be implemented in subclasses.")

    def get_equilibrium(self):
        """Return the computed equilibrium."""
        return self.equilibrium
    
