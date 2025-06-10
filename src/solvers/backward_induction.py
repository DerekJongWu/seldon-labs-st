from .solver import Solver
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

class BackwardInductionSolver(Solver):
    def __init__(self, game):
        """Initialize the backward induction solver with a game."""
        super().__init__(game)
        self.optimal_actions = {}  # Dictionary to store optimal actions at each node
        self.node_values = {}      # Dictionary to store computed values for each node
        # Debug mode to print detailed information during solving
        self.debug = False

    def solve(self):
        """Solve the game using backward induction."""
        # Start from the root and solve recursively
        self._backward_induction(self.game.root)           
        return self.optimal_actions

    def _backward_induction(self, node, depth=0):
        """
        Recursive backward induction algorithm.
        Returns the value (payoff) of the current node.
        
        Parameters:
        node: Current game node
        depth: Current depth in the tree (for debugging)
        """
        node_id = id(node)
        
        # Base case: terminal node (no actions)
        if not node.actions:
            if self.debug:
                print("  " * depth + f"Terminal node with payoff: {node.payoff}")
            self.node_values[node_id] = node.payoff
            return node.payoff
        
        # Get the player making the decision at this node
        if not node.players:
            if self.debug:
                print("  " * depth + "No players at this node")
            return None
        
        current_player = next(iter(node.players))
        if self.debug:
            print("  " * depth + f"Player {current_player} at depth {depth}")
        
        # Use the game's player indexing system instead of hardcoded values
        player_idx = self.game.get_player_index(current_player)
        
        if self.debug:
            print("  " * depth + f"Player index: {player_idx}")
        
        # Get values of all children
        best_payoff = float('-inf')
        best_action = None
        best_value = None
        
        for action, child_node in node.actions.items():
            if self.debug:
                print("  " * depth + f"Trying action: {action}")
            
            child_value = self._backward_induction(child_node, depth + 1)
            
            if child_value is None:
                continue
            
            # Extract the current player's payoff from the tuple
            player_payoff = child_value[player_idx]
            
            if self.debug:
                print("  " * depth + f"Action {action} gives payoff {player_payoff} to {current_player}")
            
            if player_payoff > best_payoff:
                best_payoff = player_payoff
                best_action = action
                best_value = child_value
                
                if self.debug:
                    print("  " * depth + f"New best action: {best_action} with payoff {best_payoff}")
        
        # Store optimal action for this node
        if best_action is not None:
            self.optimal_actions[node_id] = best_action
            if self.debug:
                print("  " * depth + f"Optimal action for node {node_id}: {best_action}")
        
        # Store node value
        self.node_values[node_id] = best_value
        
        return best_value

    def get_subgame_perfect_equilibrium(self):
        """Return the subgame perfect equilibrium strategies."""
        if not self.optimal_actions:
            self.solve()
        
        # Format the equilibrium strategies by player
        equilibrium = {}
        
        # Traverse the tree to determine which nodes are reachable
        def traverse(node, path=[]):
            node_id = id(node)
            
            # Skip terminal nodes
            if not node.actions:
                return
            
            # For each player at this node, record their optimal action
            for player in node.players:
                if player not in equilibrium:
                    equilibrium[player] = {}
                
                # Store the optimal action for this player at this information set
                if node_id in self.optimal_actions:
                    equilibrium[player][tuple(path)] = self.optimal_actions[node_id]
            
            # Continue traversal with the optimal child
            if node_id in self.optimal_actions:
                optimal_action = self.optimal_actions[node_id]
                child_node = node.actions.get(optimal_action)
                if child_node:
                    traverse(child_node, path + [optimal_action])
        
        # Start traversal from the root
        traverse(self.game.root)
        
        self.equilibrium = equilibrium
        return equilibrium
    
    def visualize_equilibrium(self):
        """Visualize the game tree with equilibrium strategies highlighted."""
        if not self.optimal_actions:
            self.solve()
            
        # Create a new directed graph
        graph = nx.DiGraph()
        node_labels = {}
        
        # Add nodes and edges to the graph
        def add_nodes_and_edges(node, parent=None, action=None):
            node_id = id(node)
            
            # Create label for the node
            if not node.actions:  # Terminal node
                label = f"Payoff: {node.payoff}"
            else:
                players_str = ", ".join(sorted(node.players))
                optimal = self.optimal_actions.get(node_id, "N/A")
                label = f"{players_str}\nOptimal: {optimal}"
            
            node_labels[node_id] = label
            graph.add_node(node_id)
            
            # Add edge from parent if applicable
            if parent is not None:
                # Check if this edge is part of the equilibrium path
                parent_optimal = self.optimal_actions.get(parent)
                is_optimal = (parent_optimal == action)
                
                # Add the edge with attributes
                graph.add_edge(parent, node_id, 
                               action=action,
                               color="red" if is_optimal else "black",
                               width=2.0 if is_optimal else 1.0)
            
            # Process all children of this node
            for act, child in node.actions.items():
                add_nodes_and_edges(child, node_id, act)
        
        # Build the graph starting from the root
        add_nodes_and_edges(self.game.root)
        
        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = graphviz_layout(graph, prog="dot")
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=3000, node_color="lightblue")
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10)
        
        # Draw edges with appropriate colors and widths
        for (u, v, data) in graph.edges(data=True):
            nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], 
                                  edge_color=data["color"], 
                                  width=data["width"])
        
        # Add edge labels
        edge_labels = {(u, v): data["action"] for u, v, data in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("Game Tree with Equilibrium Path Highlighted")
        plt.axis("off")
        plt.show()
        
    def print_equilibrium(self):
        """Print the equilibrium strategies in a readable format."""
        if not self.equilibrium:
            self.get_subgame_perfect_equilibrium()
        
        print("Subgame Perfect Equilibrium Strategies:")
        for player, strategies in self.equilibrium.items():
            print(f"Player {player}:")
            for path, action in strategies.items():
                path_str = " → ".join(["Root"] + list(path)) if path else "Root"
                print(f"  At '{path_str}', choose '{action}'")
        
        print("\nEquilibrium Path:")
        node = self.game.root
        path = ["Root"]
        
        while node and node.actions:
            node_id = id(node)
            if node_id in self.optimal_actions:
                next_action = self.optimal_actions[node_id]
                path.append(next_action)
                
                # Find the child with this action
                node = node.actions.get(next_action)
            else:
                break
        
        print(" → ".join(path))
        
        if node and node.payoff is not None:
            print(f"Terminal payoffs: {node.payoff}")

    def record_equilibrium(self): 
        """Create dictionary of the equilibrium.""" 
        if not self.equilibrium:
            self.get_subgame_perfect_equilibrium()

        player_actions = {}
        for player, strategies in self.equilibrium.items():
            player_actions[player] = {}
            for path, action in strategies.items():
                player_actions[player] = action
                
        return player_actions
    