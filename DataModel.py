import pandas as pd
from matplotlib import colors
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
from collections import OrderedDict
import math

class DataModel:
    def __init__(self, controller, path_to_unclean_set = 'games_detailed_info.csv'):
        self.controller = controller
        self.unclean_set= pd.read_csv(path_to_unclean_set)
        self.language_value = {
            'NO' : 1,
            'SOME' : 2,
            'MODERATE' : 3,
            'EXTENSIVE' : 4,
            'UNPLAYABLE' : 5
        }

        self.result_value = {
            'BEST' : 3,
            'RECOMMENDED' : 2,
            'NOTRECOMMENDED' : 1
        }

        self.random_seed = 87
        self.cmap = colors.ListedColormap(['r','g','b','c','m', 'y'])
        self.clean_data_set = self.create_clean_data_set(self.unclean_set)


        #create scaler and scale total data set
        self.scaler_model = StandardScaler().fit(self.clean_data_set)
        self.scaled_data_set = self.scaler_model.transform(self.clean_data_set)

        #create pca model and find pca of full set
        pca = PCA(n_components=2, random_state=self.random_seed)
        self.pca_model = pca.fit(self.scaled_data_set)
        pca_data_array = self.pca_model.transform(self.scaled_data_set)
        self.pca_data_set = pd.DataFrame(data=pca_data_array, columns = ['PC1', 'PC2'])

        self.train_set, self.test_set = train_test_split(self.pca_data_set, test_size=0.2, random_state=self.random_seed, shuffle=True)

        # create kmeans_model and find kmeans of full set
        self.kmeans_model = KMeans(n_clusters=6, random_state=self.random_seed)
        self.kmeans_model.fit(self.train_set)
        self.kmeans_data_set = self.kmeans_model.predict(self.train_set)
        self.kmeans_test_set = self.kmeans_model.predict(self.test_set)

        #create linear regression model
        self.linear_regression_model = LinearRegression().fit(self.train_set, self.kmeans_data_set)
        print(self.linear_regression_model.coef_)

        self.add_cluster_numbers()
        self.name_index_set = pd.DataFrame(self.unclean_set['primary'], self.clean_data_set.index.values, columns=['index', 'name'])


    def add_cluster_numbers(self):
        all_games_kmeans = self.kmeans_model.predict(self.pca_data_set)
        self.clean_data_set['cluster'] = all_games_kmeans.tolist()
        self.unclean_set['cluster'] = self.clean_data_set['cluster']

    def add_brackets_and_eval(self, string_item):
        string_item = str(string_item)
        if string_item[0] != '[':
            string_item = '[' + str(string_item)
        if string_item[-1] != ']':
            string_item = str(string_item) + ']'
        return eval(string_item)

    def calculate_players_from_ordered_dict(self, game):
        playercount_normalized_results = {}
        for playercount in game:
            if not 'result' in playercount:
                return 0
            total_votes = 0
            result_total_result = 0
            result_normalized_result = 0
            for result in playercount['result']:
                # current_player_count = result['@value']
                multiplier = self.result_value[result['@value'].upper().replace(' ', '')]
                total_votes += int(result['@numvotes'])
                result_total_result += int(result['@numvotes']) * multiplier
            if(total_votes):
                result_normalized_result = result_total_result / total_votes
            else:
                result_normalized_result = 0
                result_normalized_result = 0
            playercount_normalized_results[playercount['@numplayers']]=result_normalized_result
        sum_of_weighted_players = 0
        sum_of_keys = 0
        num_of_keys = 0
        for key in playercount_normalized_results.keys():
            if not key.isnumeric():
                num = int(key[:-1]) + 1
            else:
                num = int(key)
            sum_of_weighted_players += num * (playercount_normalized_results[key]/3)
            sum_of_keys += num
            num_of_keys += 1
        return (sum_of_weighted_players)/(sum_of_keys/num_of_keys)

    def calculate_age_from_ordered_dict(self, game):
        weighted_age_sum = 0
        total_votes = 0
        for age in game:
            votes = age['@numvotes']
            if age['@value'].isnumeric():
                age_num = age['@value']
            else:
                age_num = 25
            weighted_age_sum += int(age_num) * int(votes) 
            total_votes += int(votes)
        if total_votes:
            return weighted_age_sum/total_votes
        else:
            return 0

    def normalize_language_dependence(self, game):
        sum_of_weighted_language = 0
        total_num_votes = 0
        for level in game:
            value_key = level['@value'].split(' ')[0].upper()
            language_multiplier = self.language_value[value_key]
            votes = int(level['@numvotes'])
            total_num_votes += votes
            sum_of_weighted_language += language_multiplier * votes
        return (sum_of_weighted_language / total_num_votes) if total_num_votes > 0 else 0

    def create_clean_data_set(self, unclean_set):
        clean_data_set = None
        clean_data_set = unclean_set.drop(['Unnamed: 0', 'type', 'id', 'thumbnail', 'image', 'primary',
            'alternate', 'description', 'yearpublished', 'boardgameexpansion', 'boardgameimplementation',
            'boardgamedesigner', 'boardgameartist', 'boardgamepublisher', 'boardgameintegration', 'boardgamecompilation', 'boardgamecategory',
            'boardgamemechanic', 'boardgamefamily', 'RPG Item Rank', 'Accessory Rank', 'Video Game Rank', 'Amiga Rank',
            'Commodore 64 Rank', 'Arcade Rank', 'Atari ST Rank'], axis=1)
        clean_data_set = clean_data_set.fillna(0)
        clean_data_set['suggested_num_players'] = clean_data_set['suggested_num_players'].apply(self.add_brackets_and_eval)
        clean_data_set['suggested_num_players'] = clean_data_set['suggested_num_players'].apply(self.calculate_players_from_ordered_dict)
        clean_data_set = clean_data_set[clean_data_set['suggested_playerage'] != 0]
        clean_data_set['suggested_playerage'] = clean_data_set['suggested_playerage'].apply(self.add_brackets_and_eval)
        clean_data_set['suggested_playerage'] = clean_data_set['suggested_playerage'].apply(self.calculate_age_from_ordered_dict)
        clean_data_set = clean_data_set[clean_data_set['suggested_language_dependence'] != 0]
        clean_data_set['suggested_language_dependence'] = clean_data_set['suggested_language_dependence'].apply(self.add_brackets_and_eval)
        clean_data_set['suggested_language_dependence'] = clean_data_set['suggested_language_dependence'].apply(self.normalize_language_dependence)
        clean_data_set = clean_data_set[clean_data_set['suggested_num_players'] > 0]
        clean_data_set = clean_data_set[clean_data_set['suggested_playerage'] > 0]
        clean_data_set = clean_data_set[clean_data_set['suggested_language_dependence'] > 0]
        return clean_data_set

    def predict_game_cluster(self, game_id_list):
        number_of_features = len(self.clean_data_set.columns) - 1
        games = np.empty((0,number_of_features), float)
        for game_id in game_id_list:
            games = np.append(games, np.array([self.clean_data_set.iloc[game_id].drop('cluster')]), axis=0)
        average_of_games = np.average(games, axis=0)
        scaled_average = self.scaler_model.transform(average_of_games.reshape(1, -1))
        pca_average = self.pca_model.transform(scaled_average)
        predicted_cluster = self.linear_regression_model.predict(pca_average)
        if predicted_cluster - math.floor(predicted_cluster) <= .5:
            predicted_cluster = math.floor(predicted_cluster)
        else:
            predicted_cluster = math.ceil(predicted_cluster)
        if predicted_cluster < 0:
            predicted_cluster = 0
        elif predicted_cluster > 5:
            predicted_cluster = 5

        return predicted_cluster

    def choose_result(self, predicted_cluster):
        if predicted_cluster >=0 and predicted_cluster <=5:
            return self.unclean_set[self.unclean_set['cluster'] == predicted_cluster].sample(n=1)
    
    def get_names_and_ids(self):
        name_index_set = {}
        for k,v in zip(self.unclean_set['primary'], self.clean_data_set.index.values):
            name_index_set[k] = v
        return name_index_set
