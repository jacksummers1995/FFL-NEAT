# %% -- libs

import pandas as pd
import neat

# %% -- Read in Data

# Meta Data
meta = pd.read_csv('../2. Data/meta_data.csv')
feature_types_dict = dict(zip(meta['feature'], meta['feature_type']))

apply_stats_features = meta[meta['apply_stats'] == "TRUE"]['feature'].tolist()

# Current Season
data_22_23 = pd.read_csv('../2. Data/22-23 FFL.csv', dtype=feature_types_dict)

# %% -- Apply stats to Current Season

# Create a new DataFrame to hold the stats
new_stats_df = pd.DataFrame()

# Loop through each player in the dataset
for player in data_22_23['name'].unique():
    player_data = data_22_23[data_22_23['name'] == player]
    
    # Sort by game week to ensure the data is in order
    player_data = player_data.sort_values(by=['GW'])

    # Find the earliest game week for the player
    earliest_gw = player_data['GW'].min()
    
    # Create missing rows for earlier game weeks
    if earliest_gw > 1:
        missing_data = pd.DataFrame({
            'name': [player] * (earliest_gw - 1),
            'GW': range(1, earliest_gw),
            'player_available': [False] * (earliest_gw - 1)
        })
        
        # Add in the other columns from data_22_23 and set them to 0
        for column in data_22_23.columns:
            if column not in missing_data.columns:
                missing_data[column] = 0

        # Set the features specified in apply_stats_features to 0
        for feature in apply_stats_features:
            missing_data[feature] = 0
                
        player_data = pd.concat([missing_data, player_data])
    
    # Add player_available flag for existing records
    player_data['player_available'] = True
    
    # Calculate the mean and 5 GW rolling average for features where apply_stats=True,
    # but only for the weeks when the player is available
    for feature in apply_stats_features:
        player_data[f'{feature}_mean_upto_GW'] = player_data.apply(lambda row: row[feature].expanding().mean() if row['player_available'] else None, axis=1)
        player_data[f'{feature}_rolling_5GW'] = player_data.apply(lambda row: row[feature].rolling(window=5).mean() if row['player_available'] else None, axis=1)

    new_stats_df = pd.concat([new_stats_df, player_data])

# Replace the old DataFrame with the new one that contains calculated stats
data_22_23 = new_stats_df.reset_index(drop=True)





# %%
current_season_features = ['value']
ALL_PLAYERS = data_22_23.name.unique()

# Split out Game Weeks
gw_values = list(range(0, 39))  # 0 to 38
data_22_23_gw_dict = {}
for gw in gw_values[1:]:
    data_22_23_gw_dict[gw] = data_22_23[data_22_23.GW == gw][['name','position','team']+current_season_features]


# Last Season
last_season_features = ['value','goals_conceded','goals_scored','assists','saves','clean_sheets','minutes','selected','total_points','GW']
data_21_22 = pd.read_csv('21-22 FFL.csv', usecols=['name']+last_season_features)

data_21_22_summary = data_21_22.groupby('name').agg(
    last_season_value_mean              =('value', 'mean'),
    last_season_value_max               =('value', 'max'),
    last_season_value_min               =('value', 'min'),
    last_season_goals_conceded_total    =('goals_conceded', 'sum'),
    last_season_goals_scored_total      =('goals_scored', 'sum'),
    last_season_assists_total           =('assists', 'sum'),
    last_season_saves_total             =('saves', 'sum'),
    last_season_clean_sheets_total      =('clean_sheets', 'sum'),
    last_season_minutes_total           =('minutes', 'sum'),
    last_season_total_points_total      =('total_points', 'sum'),
).reset_index()

# Set data for Game Week 0
# Join Data together on name
gw0_player_data = data_22_23_gw_dict[1].merge(data_21_22_summary,'left','name') # Take data from GW 1 as data pre season
gw0_player_data['new_player'] = gw0_player_data['last_season_value_mean'].isna().astype(int)

gw0_player_data['gw'] = 0

# Fill missing values in all columns with 0
gw0_player_data.fillna(0, inplace=True)

data_22_23_gw_dict[0] = gw0_player_data

# Set max / mean / std of each column to normalize for each Game Week
max_gw_player_data = {}
means_gw_player_data = {}
stds_gw_player_data = {}

for gw in gw_values:
    means_gw_player_data[gw] = data_22_23_gw_dict[gw].drop(columns=['name','position','team']).mean().to_dict()
    stds_gw_player_data[gw] = data_22_23_gw_dict[gw].drop(columns=['name','position','team']).std().to_dict()

# Create one-hot encoding for char columns
one_hot_encoding_dicts = {}
for col in ['position','team']:
    unique_values = gw0_player_data[col].unique()
    one_hot_encoding_dicts[col] = {value: [int(value == unique_val) for unique_val in unique_values] for value in unique_values}

# Manually create one-hot encoding for GW
one_hot_encoding_dicts['GW'] = {gw: [int(gw == unique_gw) for unique_gw in gw_values] for gw in gw_values}

# Create Dict of Player attributes for start
# Convert the DataFrame to a list of dictionaries, one for each row
gw0_player_data_dict = gw0_player_data.to_dict(orient='records')

# Create a dictionary using the 'name' field as the key for each record
gw0_player_data_dict = {record['name']: record for record in gw0_player_data_dict}
gw0_player_data_dict


# %% -- Create Player Object
class Player:
    def __init__(self, **kwargs):
        self.points = 0
        for key, value in kwargs.items():
            setattr(self, key, value)

# Create list of Player Objects
all_available_players = []

for player in gw0_player_data_dict:
    player_to_add = Player(**gw0_player_data_dict[player])
    all_available_players.append(player_to_add)

# %% -- Create Team Object

MAX_POSITIONS = {
    'GK': 1,
    'DEF': 4,
    'MID': 4,
    'FWD': 2
}

class Team:
    def __init__(self):
        self.original_budget = 100_000_000
        self.dynamic_budget = self.original_budget # Budget that gets updated each game week as players value changes
        self.players = []
        self.team_points = 0

    def add_player(self, player):
        if player.value <= self.dynamic_budget:
            self.players.append(player)
            self.dynamic_budget -= player.value

    def remove_player(self, player):
        self.players.remove(player)
        self.dynamic_budget += player.value

    def update_team_points(self):
        self.team_points += sum(player.gw_points for player in self.players)

    def can_add_player_based_on_position(self, player):
        # Check the number of players of this position in the current team and compare with maximum allowed
        players_of_same_position = [p for p in self.players if p.position == player.position]
        return len(players_of_same_position) < MAX_POSITIONS[player.position]

# %% -- Helper Functions

# Function to prepare input for a single player
def prepare_input_for_player(player,gw):
    player_input = []
    
    # Normalize numeric features
    for feature in last_season_features + current_season_features:
        feature_to_add = (getattr(player, feature) - means_gw_player_data[gw]) / stds_gw_player_data[gw]
        player_input.append(feature_to_add)

    # Non-normalize numeric features
    for feature in ['new_player']:
        feature_to_add = getattr(player, feature)
        player_input.append(feature_to_add)

    # One-hot encoded character features
    for feature in ['position', 'team', 'gw']:
        one_hot_vector = one_hot_encoding_dicts[feature][getattr(player, feature)]
        player_input.extend(one_hot_vector)

    return player_input


# %% -- Setup NEAT structure

# Load NEAT config
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config.txt')

# Create the population
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(neat.StatisticsReporter())


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Initialize fitness
        genome.fitness = 0
        
        # Create Team Object
        team = Team()

        # GAME WEEK 0 | INITIAL SELECTION
        output_dict = {}

        # Pass each Player Inputs to genome | Save Output in Dict
        for player in all_available_players:
            player_input = prepare_input_for_player(player,0)
            output = net.activate(player_input)
            output_dict[player.name] = output

        # Create a list of tuples (player, output) from the network output
        indexed_output = [(player, output_dict[player.name]) for player in all_available_players]

        # Sort the list by value in descending order
        indexed_output.sort(key=lambda x: x[1], reverse=True)

        # Keep track of the number of players added
        num_players_added = 0

        # Iterate over the sorted list, adding players to the team
        for player, _ in indexed_output:
            # Select player
            player = all_available_players[player]
            
            # Check if the player can be added based on position and budget
            if team.can_add_player_based_on_position(player) and team.dynamic_budget >= player.value:
                team.add_player(player)
                
                # Increment the number of players added
                num_players_added += 1
                
                # Stop if 11 players have been added
                if num_players_added == 11:
                    break

        # GAME WEEK 1+ | PLAY GAME
        # Loop through each Game Week
        for game_week in gw_values[1:]:
            # Update Player stats for Game Week
            for player in all_available_players:
                # Fetch the data for this player and this game week
                gw_data = data_22_23_gw_dict[game_week]
                player_data_for_gw = gw_data[gw_data['name'] == player.name]
                
                # If there is no data for this player for this game week, continue to the next player
                if player_data_for_gw.empty:
                    continue
                
                # Update player stats for this game week
                for stat in current_season_features:
                    setattr(player, stat, player_data_for_gw[stat].values[0])

                # Update Game Week 
                setattr(player, stat, game_week)



            # Update budget based on new player values
            team.dynamic_budget = sum(player.value for player in team.players) - team.original_budget


            output_dict = {}

            # Pass each Player Inputs to genome | Save Output in Dict
            for player in all_available_players:
                player_input = prepare_input_for_player(player,0)
                output = net.activate(player_input)
                output_dict[player.name] = output

            genome.fitness = team.team_points

# Run until a solution is found.
winner = p.run(eval_genomes, 50) 


