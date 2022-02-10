import Dashboard
from numpy.random import random_integers
import pandas as pd
import random
import tkinter as tk
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn as sk
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import time
import math
from matplotlib import colors
import DataModel


class Controller():
    def __init__(self):
        self.model = DataModel.DataModel(self)
        self.view = Dashboard.Dashboard(self)
        # view.create_dropdowns(self.model.get_names_and_ids())
        # {'Sam':1, 'Justin': 2, 'Triston': 3, 'Lilly' : 4, 'Jehu': 5}
        self.names_index_dict = self.model.get_names_and_ids()
        self.view.create_dropdowns(self.names_index_dict)

        self.view.window.mainloop()
    def getPrediction(self, list_of_titles):
        list_of_index = []
        for i in list_of_titles:
            item = self.names_index_dict[i]
            list_of_index.append(item)
        predicted_cluster = self.model.predict_game_cluster(list_of_index)
        result = self.model.choose_result(predicted_cluster)
        title = result['primary'].iloc[0]
        self.view.set_result(title)

    def sum_of_array(an_array):
        sum = 0 
        for i in an_array:
            sum += i
        return sum