import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from src.game_node import Node

class Game:
    """Represents a game theory structure with players, actions, and payoffs."""
    def __init__(self):
        self.root = Node()
        self.current_nodes = [self.root]  # Track leaf nodes for expansion
        self.players = []  # List to track players in the order they're added
        self.player_indices = {}  # Maps player names to their indices
    
    def add_player(self, player):
        """Add a player to the game if not already present."""
        if player not in self.player_indices:
            self.players.append(player)
            self.player_indices[player] = len(self.players) - 1
        return self.player_indices[player]
    
    def get_player_index(self, player):
        """Return the index of the player in payoff tuples."""
        if player not in self.player_indices:
            raise ValueError(f"Player {player} not found in game")
        return self.player_indices[player]
    
    def add_moves(self, player, actions):
        """Adds moves for a player at all current leaf nodes."""
        # Add player to the tracking system if not already added
        self.add_player(player)
        
        new_nodes = []
        for node in self.current_nodes:
            node.players.add(player)
            for action in actions:
                child_node = Node()
                node.add_action(action, child_node)
                new_nodes.append(child_node)
        self.current_nodes = new_nodes
    
    def add_outcomes(self, outcomes):
        """Assigns payoffs to the current leaf nodes."""
        if len(outcomes) != len(self.current_nodes):
            raise ValueError("Number of outcomes must match the number of terminal nodes.")
        for node, payoff in zip(self.current_nodes, outcomes):
            node.payoff = payoff
    
    def display_tree(self):
        """Recursively prints the game tree."""
        def recurse(node, depth=0):
            payoff_text = f", Payoff: {node.payoff}" if node.payoff is not None else ""
            print("  " * depth + f"Players: {node.players}{payoff_text}")
            for action, child in node.actions.items():
                print("  " * depth + f"Action: {action}")
                recurse(child, depth + 1)
        recurse(self.root)
    
    def visualize_tree(self):
        """Visualizes the game tree with improved spacing using Graphviz."""
        graph = nx.DiGraph()
        node_labels = {}
        
        def add_edges(node, parent=None, action_label=None):
            """Recursively add nodes and edges to the graph."""
            node_id = id(node)  # Unique identifier
            label = f"{', '.join(node.players)}" if node.actions else f"Payoff: {node.payoff}"
            node_labels[node_id] = label

            if parent is not None:
                graph.add_edge(parent, node_id, action=action_label)

            for action, child in node.actions.items():
                add_edges(child, node_id, action)

        add_edges(self.root)

        # Use Graphviz DOT layout for better hierarchy
        pos = graphviz_layout(graph, prog="dot")

        # Draw graph
        plt.figure(figsize=(10, 6))
        nx.draw(graph, pos, with_labels=True, labels=node_labels, node_color="lightblue", edge_color="black", 
                node_size=3000, font_size=8, font_weight="bold", arrowsize=15)

        # Add action labels on edges
        edge_labels = {(u, v): data["action"] for u, v, data in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, 
                                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

        plt.title("Game Tree Visualization")
        plt.show()