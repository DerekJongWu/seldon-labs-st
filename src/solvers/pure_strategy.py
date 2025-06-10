from .solver import Solver
from itertools import product


class PureStrategyNashSolver(Solver):
    def __init__(self, game):
        """Initialize the Pure Strategy Nash solver with a game."""
        super().__init__(game)
        self.debug = False
        self.equilibria = []  # Store all found Nash equilibria
        self.strategy_profiles = {}  # Store all possible strategy profiles
        self.payoff_matrix = {}  # Store payoffs for each strategy profile

    def solve(self):
        """
        Solve the game to find all Pure Strategy Nash Equilibria.
        Returns a list of equilibria, where each equilibrium is a dictionary
        mapping players to their equilibrium strategies.
        """
        # Generate all possible strategy profiles
        self._generate_strategy_profiles()
        
        # Compute payoffs for each strategy profile
        self._compute_payoffs()
        
        # Find Nash equilibria
        self._find_nash_equilibria()
        
        # Store the first equilibrium in self.equilibrium for compatibility
        if self.equilibria:
            self.equilibrium = self.equilibria[0]
        else:
            self.equilibrium = {}
            
        return self.equilibria

    def _generate_strategy_profiles(self):
        """Generate all possible pure strategy profiles for the game."""
        # Use game.players attribute instead of get_all_players()
        players = self.game.players
        
        # Find all available actions for each player
        player_actions = {}
        for player in players:
            # Collect all possible actions for this player
            actions = self._collect_player_actions(player)
            if actions:
                player_actions[player] = actions
                
        if self.debug:
            print(f"Player actions: {player_actions}")
            
        # Generate all possible strategy profiles using Cartesian product
        action_items = list(player_actions.items())
        players_list = [p for p, _ in action_items]
        action_lists = [actions for _, actions in action_items]
        
        # Generate all combinations of actions
        for actions_combo in product(*action_lists):
            strategy = {players_list[i]: action for i, action in enumerate(actions_combo)}
            strategy_key = self._strategy_to_key(strategy)
            self.strategy_profiles[strategy_key] = strategy
            
        if self.debug:
            print(f"Generated {len(self.strategy_profiles)} strategy profiles")

    def _collect_player_actions(self, player):
        """Collect all possible actions for a player throughout the game tree."""
        actions = set()
        
        def traverse(node):
            # If player is active at this node, collect their actions
            if player in node.players:
                actions.update(node.actions.keys())
                
            # Continue traversal for all children
            for child in node.actions.values():
                traverse(child)
        
        # Start traversal from the root
        traverse(self.game.root)
        return list(actions)

    def _compute_payoffs(self):
        """Compute payoffs for each strategy profile."""
        for strategy_key, strategy in self.strategy_profiles.items():
            # Simulate the game with this strategy profile
            payoffs = self._simulate_game(strategy)
            self.payoff_matrix[strategy_key] = payoffs
            
            if self.debug:
                print(f"Strategy {strategy_key} yields payoffs {payoffs}")

    def _simulate_game(self, strategy):
        """
        Simulate the game with a given strategy profile and return the payoffs.
        
        Parameters:
        strategy: Dictionary mapping players to their chosen actions
        
        Returns:
        Tuple of payoffs for all players
        """
        # Start at the root node
        node = self.game.root
        
        # Follow the game tree according to the strategy profile
        while node and node.actions:
            # If no players at this node, break
            if not node.players:
                break
                
            # Get the player making the decision at this node
            current_player = next(iter(node.players))
            
            # Get the action for this player from the strategy profile
            if current_player in strategy:
                action = strategy[current_player]
                
                # Check if this action is available at this node
                if action in node.actions:
                    # Move to the next node according to the strategy
                    node = node.actions[action]
                else:
                    # Action not available, break the simulation
                    if self.debug:
                        print(f"Invalid action {action} for player {current_player} at node")
                    return None
            else:
                # No strategy defined for this player
                if self.debug:
                    print(f"No strategy defined for player {current_player}")
                return None
        
        # Return the payoffs at the terminal node
        return node.payoff if node else None

    def _find_nash_equilibria(self):
        """
        Find all Pure Strategy Nash Equilibria.
        A strategy profile is a Nash equilibrium if no player can improve
        their payoff by unilaterally changing their strategy.
        """
        players = self.game.players
        player_indices = {player: self.game.get_player_index(player) for player in players}
        
        for strategy_key, strategy in self.strategy_profiles.items():
            is_nash = True
            payoffs = self.payoff_matrix.get(strategy_key)
            
            # Skip if this strategy profile doesn't have valid payoffs
            if payoffs is None:
                continue
                
            # Check if any player can improve by deviating
            for player, current_action in strategy.items():
                player_idx = player_indices[player]
                current_payoff = payoffs[player_idx]
                
                # Try each alternative action for this player
                alternative_actions = self._collect_player_actions(player)
                for alt_action in alternative_actions:
                    if alt_action == current_action:
                        continue
                        
                    # Create an alternative strategy with this player's action changed
                    alt_strategy = strategy.copy()
                    alt_strategy[player] = alt_action
                    alt_key = self._strategy_to_key(alt_strategy)
                    
                    # Get payoffs for the alternative strategy
                    alt_payoffs = self.payoff_matrix.get(alt_key)
                    
                    # Skip if this alternative doesn't have valid payoffs
                    if alt_payoffs is None:
                        continue
                        
                    # Check if player would get higher payoff by deviating
                    alt_payoff = alt_payoffs[player_idx]
                    if alt_payoff > current_payoff:
                        is_nash = False
                        if self.debug:
                            print(f"Not Nash: Player {player} can improve by switching from {current_action} to {alt_action}")
                        break
                
                if not is_nash:
                    break
            
            # If no player can improve by deviating, it's a Nash equilibrium
            if is_nash:
                self.equilibria.append(strategy)
                if self.debug:
                    print(f"Found Nash equilibrium: {strategy}")
        
        if self.debug:
            print(f"Found {len(self.equilibria)} Nash equilibria")

    def _strategy_to_key(self, strategy):
        """Convert a strategy profile to a hashable key."""
        # Sort by player to ensure consistent key generation
        return tuple(sorted((player, action) for player, action in strategy.items()))
        
    def get_player_at_index(self, index):
        """Get player name from index."""
        if 0 <= index < len(self.game.players):
            return self.game.players[index]
        return f"Player{index}"

    def print_equilibria(self):
        """Print all found Nash equilibria in a readable format."""
        if not self.equilibria:
            self.solve()
            
        print(f"Found {len(self.equilibria)} Pure Strategy Nash Equilibria:")
        
        for i, eq in enumerate(self.equilibria):
            print(f"\nEquilibrium {i+1}:")
            for player, action in eq.items():
                print(f"  Player {player}: {action}")
                
            # Print payoffs for this equilibrium
            eq_key = self._strategy_to_key(eq)
            payoffs = self.payoff_matrix.get(eq_key)
            if payoffs:
                payoff_str = ", ".join(f"{self.get_player_at_index(i)}: {p}" 
                                     for i, p in enumerate(payoffs))
                print(f"  Payoffs: {payoff_str}")

    def visualize_equilibria(self, highlight_index=0):
        """
        Visualize the Nash equilibria as a strategic form grid with the specified 
        equilibrium highlighted.
        
        Parameters:
        highlight_index: Index of the equilibrium to highlight (default: first equilibrium)
        """
        
        if not self.equilibria:
            self.solve()
            
        if not self.equilibria:
            print("No Nash equilibria found to visualize.")
            return
            
        if highlight_index >= len(self.equilibria):
            highlight_index = 0
            
        # Get the equilibrium to highlight
        highlight_eq = self.equilibria[highlight_index]
        
        # Extract player information from the game
        players = self.game.players
        
        # Group players by their available actions
        player_actions = {}
        for player in players:
            actions = self._collect_player_actions(player)
            if actions:
                player_actions[player] = actions
        
        if len(player_actions) <= 2:
            # For 2-player games, create a standard grid visualization
            self._visualize_two_player_grid(player_actions, highlight_eq, highlight_index)
        else:
            # For games with more than 2 players, create a tabular visualization
            self._visualize_multi_player_table(player_actions, highlight_eq, highlight_index)
            
    def _visualize_two_player_grid(self, player_actions, highlight_eq, highlight_index):
        """Create a grid visualization for a 2-player game with combined payoffs and no heatmap.
        Player 1 is on the right axis (vertical) and Player 2 is on the top axis (horizontal)."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        
        plt.figure(figsize=(6, 4))
        
        # Get the two players
        players = list(player_actions.keys())
        if len(players) < 2:
            print("Not enough players with actions to visualize as a grid.")
            return
        
        player1, player2 = players[0], players[1]
        actions1 = player_actions[player1]
        actions2 = player_actions[player2]
        
        # Create a grid with player 2 actions as columns (horizontal) and 
        # player 1 actions as rows (vertical)
        nrows, ncols = len(actions1), len(actions2)
        
        # Track Nash equilibria positions
        nash_positions = []
        highlight_position = None
        
        # Create a matrix to store cell data (payoff values as strings)
        cell_texts = np.empty((nrows, ncols), dtype=object)
        cell_colors = np.full((nrows, ncols), 'white', dtype=object)
        
        # Build payoff matrices
        for i, action2 in enumerate(actions2):  # Columns - player 2
            for j, action1 in enumerate(actions1):  # Rows - player 1
                strategy = {player1: action1, player2: action2}
                strategy_key = self._strategy_to_key(strategy)
                payoffs = self.payoff_matrix.get(strategy_key)
                
                # Skip if strategy doesn't have valid payoffs
                if payoffs is None:
                    cell_texts[j, i] = "N/A"
                    continue
                    
                # Get player indices
                p1_idx = self.game.get_player_index(player1)
                p2_idx = self.game.get_player_index(player2)
                
                # Store payoffs as a formatted string: "p1_payoff, p2_payoff"
                cell_texts[j, i] = f"{payoffs[p1_idx]:.1f}, {payoffs[p2_idx]:.1f}"
                
                # Check if this is a Nash equilibrium
                is_nash = False
                for eq in self.equilibria:
                    if eq.get(player1) == action1 and eq.get(player2) == action2:
                        is_nash = True
                        nash_positions.append((j, i))
                        if eq == highlight_eq:
                            highlight_position = (j, i)
                            cell_colors[j, i] = 'lightgreen'  # Shade highlighted equilibrium green
                        elif is_nash:
                            cell_colors[j, i] = 'lightblue'  # Shade other equilibria blue
                        break
        
        # Create the plot
        ax = plt.gca()
        
        # Set up the grid dimensions
        ax.set_xlim(-0.5, ncols - 0.5)
        ax.set_ylim(-0.5, nrows - 0.5)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(actions2)))
        ax.set_yticks(np.arange(len(actions1)))
        ax.set_xticklabels(actions2)
        ax.set_yticklabels(actions1)
        
        # Important: Invert the y-axis to put player 2 at the top
        ax.invert_yaxis()
        
        # Rotate the tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add grid
        ax.set_xticks(np.arange(-.5, len(actions2), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(actions1), 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        
        # Add cell backgrounds
        for i in range(ncols):
            for j in range(nrows):
                rect = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=True, 
                                    color=cell_colors[j, i], alpha=0.3)
                ax.add_patch(rect)
        
        # Add text annotations for payoffs
        for i in range(ncols):
            for j in range(nrows):
                if cell_texts[j, i] != "N/A":
                    ax.text(i, j, cell_texts[j, i], ha="center", va="center", fontsize=10)
        
        # Highlight the selected equilibrium with a bold border
        if highlight_position:
            rect = plt.Rectangle((highlight_position[1] - 0.5, highlight_position[0] - 0.5), 1, 1, 
                               fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(rect)
        
        # Add title and labels - position the labels on the top and right
        plt.title(f"Nash Equilibrium {highlight_index+1} of {len(self.equilibria)}")
        
        # Move x-axis label to top
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        
        # Set labels
        plt.xlabel(f"Player {player2} Actions")
        plt.ylabel(f"Player {player1} Actions")
        
        # Simplified legend with only Nash Equilibrium
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.3, edgecolor='black', 
                         label='Nash Equilibrium')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()

    ## See if this needs to be fixed
    def _visualize_multi_player_table(self, player_actions, highlight_eq, highlight_index):
        """Create a grid-style visualization for games with more than 2 players."""
        
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        from matplotlib.patches import Rectangle
        
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        # Get all strategy profiles
        profiles = sorted(self.strategy_profiles.values(), 
                        key=lambda s: tuple(s.get(p, '') for p in self.game.players))
        
        # Filter out profiles without valid payoffs
        valid_profiles = []
        for strategy in profiles:
            strategy_key = self._strategy_to_key(strategy)
            payoffs = self.payoff_matrix.get(strategy_key)
            if payoffs is not None:
                valid_profiles.append((strategy, payoffs))
        
        # If no valid profiles, show a message
        if not valid_profiles:
            plt.text(0.5, 0.5, "No valid strategy profiles to display", 
                   ha='center', va='center', fontsize=14)
            plt.tight_layout()
            plt.show()
            return
        
        # Create grid data
        num_rows = len(valid_profiles)
        num_cols = len(self.game.players) * 2   # Action + Payoff for each player + Equilibrium column
        
        # Prepare data arrays
        cell_texts = np.empty((num_rows, num_cols), dtype=object)
        cell_colors = np.full((num_rows, num_cols), 'white', dtype=object)
        
        # Column labels
        col_labels = []
        for player in self.game.players:
            col_labels.append(f"{player} Action")
            col_labels.append(f"{player} Payoff")
        
        # Fill data
        for i, (strategy, payoffs) in enumerate(valid_profiles):
            col_idx = 0
            
            # Add actions and payoffs
            for j, player in enumerate(self.game.players):
                cell_texts[i, col_idx] = strategy.get(player, 'N/A')
                col_idx += 1
                
                # Add payoff
                player_idx = self.game.get_player_index(player)
                cell_texts[i, col_idx] = f"{payoffs[player_idx]:.1f}"
                col_idx += 1
            
            # Determine if this is a Nash equilibrium
            is_nash = False
            is_highlighted = False
            
            for eq in self.equilibria:
                eq_match = True
                for player in self.game.players:
                    if eq.get(player) != strategy.get(player):
                        eq_match = False
                        break
                
                if eq_match:
                    is_nash = True
                    if eq == highlight_eq:
                        is_highlighted = True
                    break
            
            # Add equilibrium status
            if is_highlighted:
                # Color the entire row for highlighted equilibrium
                for j in range(num_cols):
                    cell_colors[i, j] = 'lightgreen'
            elif is_nash:
                cell_texts[i, col_idx] = "✓ Nash"
                # Color the entire row for nash equilibrium
                for j in range(num_cols):
                    cell_colors[i, j] = 'lightblue'
        
        # Set up the grid dimensions
        ax.set_xlim(-0.5, num_cols - 0.5)
        ax.set_ylim(-0.5, num_rows - 0.5)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(num_cols))
        ax.set_yticks(np.arange(num_rows))
        ax.set_xticklabels(col_labels)
        
        # Create row labels from strategy combinations
        row_labels = []
        for i, (strategy, _) in enumerate(valid_profiles):
            row_labels.append(f"S{i+1}")
        
        ax.set_yticklabels(row_labels)
        
        # Invert the y-axis to display strategies from top to bottom
        ax.invert_yaxis()
        
        # Adjust tick label formatting
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
        
        # Add grid
        ax.set_xticks(np.arange(-.5, num_cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, num_rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        
        # Add cell backgrounds
        for i in range(num_rows):
            for j in range(num_cols):
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, 
                                    color=cell_colors[i, j], alpha=0.3)
                ax.add_patch(rect)
        
        # Add text annotations
        for i in range(num_rows):
            for j in range(num_cols):
                if cell_texts[i, j] is not None:
                    ax.text(j, i, cell_texts[i, j], ha="center", va="center", fontsize=9)
        
        # Highlight the selected equilibrium with a bold border
        for i, (strategy, _) in enumerate(valid_profiles):
            is_highlighted = True
            for player in self.game.players:
                if strategy.get(player) != highlight_eq.get(player, None):
                    is_highlighted = False
                    break
            
            if is_highlighted:
                for j in range(num_cols):
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                       fill=False, edgecolor='green', linewidth=2)
                    ax.add_patch(rect)
        
        # Add title
        plt.title(f"Nash Equilibrium {highlight_index+1} of {len(self.equilibria)}")
        
        # Add legend (only for Nash Equilibrium) outside the plot
        # 1. Remove the standard Nash Equilibrium from legend
        # 2. Rename "Highlighted Equilibrium" to "Nash Equilibrium"
        # 3. Position it outside the grid
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.3, edgecolor='black', 
                         label='Nash Equilibrium')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Add a strategy key explanation below the plot with more space
        strategy_key = "Strategy Key:"
        for i, (strategy, _) in enumerate(valid_profiles):
            strategy_str = ", ".join([f"{p}: {a}" for p, a in strategy.items()])
            strategy_key += f"\nS{i+1}: {strategy_str}"
        
        # Position the strategy key with more space
        plt.figtext(0.01, -0.1, strategy_key, fontsize=8, verticalalignment='top')
        
        # Adjust layout to make room for the legend on the right and strategy key below
        plt.tight_layout()
        plt.subplots_adjust(right=0.85, bottom=0.25)  # Make room for legend and strategy key
        plt.show()
    
    def record_equilibrium(self): 
        return self.equilibria[0]