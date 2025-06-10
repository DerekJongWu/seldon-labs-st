import streamlit as st
import pandas as pd
from src.sampling import sample_from_distribution
from src import Game,BackwardInductionSolver
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

class Player(): 
    def __init__(self, data, player_name):
        self.data = data 
        self.name = player_name

    def calculate_payoff(self, scenario, payoff_func):
        # To Do: Can I make this more generalizable and less hardcoded? 
        var1 = float(sample_from_distribution(self.data[scenario]['var1']['mean'], self.data[scenario]['var1']['stdev'], num_samples=1)[0])
        var2 = float(sample_from_distribution(self.data[scenario]['var2']['mean'], self.data[scenario]['var2']['stdev'], num_samples=1)[0])
        var3 = float(sample_from_distribution(self.data[scenario]['var3']['mean'], self.data[scenario]['var3']['stdev'], num_samples=1)[0])
        var4 = float(sample_from_distribution(self.data[scenario]['var4']['mean'], self.data[scenario]['var4']['stdev'], num_samples=1)[0])
        var5 = float(sample_from_distribution(self.data[scenario]['var5']['mean'], self.data[scenario]['var5']['stdev'], num_samples=1)[0])
        variables_used = {'var1': var1, "var2": var2, "var3": var3, "var4": var4, 'var5': var5}
        return variables_used, payoff_func(var1, var2, var3, var4, var5)
    
def payoffs_formula(var1, var2, var3, var4, var5): 
    return var1+var2+var3+var4+var5

def convert_dict(input_dict):
    result = {}
    
    for key1, value1 in input_dict.items():
        result[key1] = {}
        
        for key2, value2 in value1.items():
            result[key1][key2] = {}
            
            # Extract mean and std values
            mean_values = value2["mean"]
            std_values = value2["std"]
            
            # Create the new structure
            for i in range(len(mean_values)):
                var_key = f"var{i+1}"
                result[key1][key2][var_key] = {
                    "mean": mean_values[i],
                    "stdev": std_values[i]
                }
    
    return result

def convert_dict_to_df(dictionary): 
    # First, let's create an empty list to store our data
    data = []
    
    # Iterate through each key-value pair in the outer dictionary
    for scenario, players in dictionary.items():
        # Create a row dictionary for this scenario
        row = {'simulation': scenario}
        
        # Iterate through each player
        for player, variables in players.items():
            # Iterate through each variable for this player
            for var_name, var_value in variables.items():
                # Create a column name by combining player and variable names
                column_name = f"{player}{var_name}"
                # Add this data to our row
                row[column_name] = var_value
        
        # Add the completed row to our data list
        data.append(row)
    
    # Create a DataFrame from our list of dictionaries
    df = pd.DataFrame(data)
    
    # Set the scenario column as the index
    df.set_index('simulation', inplace=True)
    return df

def create_game(outcomes):
    # Create a new game where "China" is the root player
    game = Game()
    
    # Add moves: China chooses between "Tariff" or "No Tariff"
    game.add_moves(player="Player1", actions=["Defect", "Cooperate"])
    
    # Add moves: The US responds to China's move
    game.add_moves(player="Player2", actions=["Defect", "Cooperate"])
    
    game.add_outcomes(outcomes)

    return game

# Initialize session state for the four scenarios and two players per scenario
if 'game_data' not in st.session_state:
    st.session_state.game_data = {
        (i, j): {1: {'mean': [0, 0, 0, 0, 0], 'std': [0, 0, 0, 0, 0]}, 
                 2: {'mean': [0, 0, 0, 0, 0], 'std': [0, 0, 0, 0, 0]}}  # (row, col): {player: {'mean': [values], 'std': [values]}}
        for i in range(2) for j in range(2)
    }

def update_values(scenario, player, new_means, new_stds):
    st.session_state.game_data[scenario][player]['mean'] = new_means
    st.session_state.game_data[scenario][player]['std'] = new_stds

def collect_data():
    player_data = {1: {}, 2: {}}
    for scenario, players in st.session_state.game_data.items():
        for player, values in players.items():  
            player_data[player][str(scenario)] = {'mean': values['mean'], 'std': values['std']}
    return player_data

def render_scenario(row, col):
    scenario = (row, col)
    if row == 0 and col == 0: 
        st.subheader("Defect, Defect")
    if row == 1 and col == 0: 
        st.subheader("Cooperate, Defect")
    if row == 0 and col == 1: 
        st.subheader("Defect, Cooperate")
    if row == 1 and col == 1: 
        st.subheader("Cooperate, Cooperate")

    col1, col2 = st.columns([1, 1])
    
    for player in [1, 2]:
        with (col1 if player == 1 else col2):
            if st.button(f"Player {player}", key=f"btn_{row}_{col}_{player}"):
                st.session_state.selected_scenario = scenario
                st.session_state.selected_player = player
                st.session_state.modal_open = True
    
    for player in [1, 2]:
        total_mean = sum(st.session_state.game_data[scenario][player]['mean'])
        st.write(f"Player {player} total mean: {total_mean}")

st.title("Seldon Labs Demo")

grid_layout = st.columns([2, 0.5, 2])  # Add spacing between columns
for i in range(2):
    with grid_layout[0]:
        render_scenario(i, 0)
        st.write("\n")
        st.write("\n")
    with grid_layout[2]:
        render_scenario(i, 1)
        st.write("\n")
        st.write("\n")
    st.write("\n")  # Add spacing between rows

# Handle modal pop-up for variable adjustment
if st.session_state.get("modal_open", False):
    scenario = st.session_state.selected_scenario
    player = st.session_state.selected_player
    
    st.sidebar.title(f"Adjust Variables for Scenario {scenario} - Player {player}")
    new_means = st.session_state.game_data[scenario][player]['mean'][:]
    new_stds = st.session_state.game_data[scenario][player]['std'][:]
    
    for idx in range(5):
        mean_col, std_col = st.sidebar.columns(2)
        new_means[idx] = mean_col.number_input(f"Mean {idx+1}", value=new_means[idx], key=f"mean_{scenario}_{player}_{idx}")
        new_stds[idx] = std_col.number_input(f"Std {idx+1}", value=new_stds[idx], key=f"std_{scenario}_{player}_{idx}")
    
    if st.sidebar.button("Save"):
        update_values(scenario, player, new_means, new_stds)
        st.session_state.modal_open = False

# Input for number of simulations and process button
num_simulations = st.number_input("Number of simulations", min_value=1, value=100, step=1)
if st.button("Process"):
    data = collect_data()
    player_data = convert_dict(data)
    # convert player_data 
    player1 = Player(player_data[1], "Player 1")
    player2 = Player(player_data[2], "Player 2")

    dd = {} 
    dc = {} 
    cd = {} 
    cc = {} 
    results = {}

    for i in (range(num_simulations)): 
        # Sample Player 1 Payoffs 
        p1_tt_variables, p1_tt = player1.calculate_payoff('(0, 0)', payoffs_formula)
        p1_tnt_variables, p1_tnt = player1.calculate_payoff('(0, 1)', payoffs_formula)
        p1_ntt_variables, p1_ntt = player1.calculate_payoff('(1, 0)', payoffs_formula)
        p1_ntnt_variables, p1_ntnt = player1.calculate_payoff('(1, 1)', payoffs_formula)

        player1_payoffs = {'dd': p1_tt, 'dc': p1_tnt, 'cd': p1_ntt, 'cc': p1_ntnt} 
        
        # Sample Player 2 payoffs
        p2_tt_variables, p2_tt = player2.calculate_payoff('(0, 0)', payoffs_formula)
        p2_tnt_variables, p2_tnt = player2.calculate_payoff('(0, 1)', payoffs_formula)
        p2_ntt_variables, p2_ntt = player2.calculate_payoff('(1, 0)', payoffs_formula)
        p2_ntnt_variables, p2_ntnt = player2.calculate_payoff('(1, 1)', payoffs_formula)
        
        player2_payoffs = {'dd': p2_tt, 'dc': p2_tnt, 'cd': p2_ntt, 'cc': p2_ntnt} 

        outcomes = [
            (int(p1_tt), int(p2_tt)),  # Both impose tariffs
            (int(p1_tnt), int(p2_tnt)),    # China tariffs, US does not
            (int(p1_ntt), int(p2_ntt)),   # China does not tariff, US does
            (int(p1_ntnt), int(p2_ntnt))   # Neither imposes tariffs
        ]

        dd[i] = {'player1': p1_tt_variables,
                'player2': p2_tt_variables}

        dc[i] = {'player1': p1_tnt_variables,
                'player2': p2_tnt_variables}

        cd[i] = {'player1': p1_ntt_variables,
                'player2': p2_ntt_variables}
        
        cc[i] = {'player1': p1_ntnt_variables,
                'player2': p2_ntnt_variables}

        g = create_game(outcomes)
        solver = BackwardInductionSolver(g)
        solver.solve()
        sim_result = solver.record_equilibrium() # What happens if there are two equilibria? 

        results[i] = {'player1': player1_payoffs,
                    'player2': player2_payoffs, 
                    'actions': sim_result
                    }
        
    tt_df = convert_dict_to_df(dd) 
    tnt_df = convert_dict_to_df(dc)
    ntt_df = convert_dict_to_df(cd)
    ntnt_df = convert_dict_to_df(cc) 
    results_df = convert_dict_to_df(results)

    with pd.ExcelWriter('output.xlsx') as writer:  
        tt_df.to_excel(writer, sheet_name='Defect Defect')
        tnt_df.to_excel(writer, sheet_name='Defect Cooperate')
        ntt_df.to_excel(writer, sheet_name='Cooperate Defect')
        ntnt_df.to_excel(writer, sheet_name='Cooperate Cooperate')
        results_df.to_excel(writer, sheet_name='Game Results')

    df = pd.read_excel('output.xlsx', sheet_name = 'Game Results')
    col1 = 'actionsPlayer1'
    col2 = 'actionsPlayer2'

    tuples = list(zip(df[col1], df[col2]))
    unique_tuples = list(set(tuples))

    tuple_counts = {}
    for tup in unique_tuples:
        tuple_counts[tup] = tuples.count(tup)

    plt.figure(figsize=(10, 6))
    labels = [f"{col1[7:]} {t[0]}, {col2[7:]} {t[1]}" for t in unique_tuples]
    counts = list(tuple_counts.values())

    sorted_indices = np.argsort(counts)[::-1]  # Descending order
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    bars = plt.bar(range(len(sorted_labels)), sorted_counts)
    plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=45, ha='right')
    plt.xlabel('Unique Value Pairs')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Outcomes')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    plt.tight_layout()
    st.pyplot(plt)



