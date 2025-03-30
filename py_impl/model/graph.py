from typing import Union
import numpy as np
import random


"""
Graph Travel:

Task:
- must support Json file
- importing a json file or txt file that print the result

"""







class QLearningGraphTravel:


    """
        A Q-learning-based agent for finding optimal paths in a graph of cities.

        Attributes:
            graph_cities (dict): Dictionary representing the graph structure, where keys are city names
                                and values are lists of neighboring cities (possible actions).
            Q_table (dict): A Q-value table storing rewards for state-action pairs.
            travel (list): List storing the sequence of visited cities in training/testing.
            epsilon (float): Probability of choosing a random action (exploration vs. exploitation).
            history_epsilon (float): Initial epsilon value for decay.
            state (str): Current city during training/testing.
            next_best_action (str): The next action chosen based on the Q-learning policy.
    """
    

    def __init__(self, graphcities, epsilon = 0.8, value_init = 0):
        
        """
        Initializes the Q-learning agent.

        Args:
            graphcities (dict): A dictionary representing the graph structure.
            epsilon (float, optional): Initial probability of taking a random action (default: 0.8).
            value_init (float, optional): Initial Q-values for all state-action pairs (default: 0).
        """
        self.graph_cities = graphcities
        self.Q_table = {city: {action: value_init for action in graphcities[city]} for city in graphcities}
        self.travel = []
        self.history_epsilon = epsilon
        self.epsilon = epsilon
        self.state = None
        self.next_best_action = None
    

    def train(self,beginning,end, epochs = 10, limit = 1000, lr = 0.001,gamma = 0.9, reward = -6, verbose = True):
        """
        Trains the Q-learning agent to find the best route between two cities.

        Args:
            beginning (str): The starting city.
            end (str): The target city.
            epochs (int, optional): Number of training iterations (default: 10).
            limit (int, optional): Maximum steps allowed per episode to prevent infinite loops (default: 1000).
            lr (float, optional): Learning rate for updating Q-values (default: 0.001).
            gamma (float, optional): Discount factor for future rewards (default: 0.9).
            reward (int, optional): Default reward for non-goal states (default: -6).
            verbose (bool, optional): If True, prints training progress (default: True).
        """
          
        for x in range(epochs):
            self.epsilon = self.history_epsilon
            self.state = beginning
            nb_travel_cities = 0
            total_reward = 0
            step_reward = 0
            while self.state != end and nb_travel_cities < limit:
                self.travel.append(self.state)
                self.choiceAction()
                step_reward = 0.1 if self.state == end else reward
                max_next_Q = max(self.Q_table[self.next_best_action].values()) if self.Q_table[self.next_best_action] else 0
                self.Q_table[self.state][self.next_best_action] += lr*(step_reward + gamma* max_next_Q - self.Q_table[self.state][self.next_best_action])
                self.state = self.next_best_action
                total_reward += step_reward
                nb_travel_cities +=1
                if self.state not in self.graph_cities.keys():
                    continue

        
            
            if verbose:
                print(f"Epoch {x + 1}/{epochs} - Total Rewards: {total_reward}")
                print(f"Path: {self.travel}")     
                

    def test(self,beginning,end, max_iterations = 100):

        """
        Tests the trained Q-learning model by finding the best path from a starting city to a destination.

        Args:
            beginning (str): The starting city.
            end (str): The target city.
            max_iterations (int, optional): Maximum number of steps allowed to prevent infinite loops (default: 100).

        Returns:
            list: The optimal path from the starting city to the target city.
        """
        
        self.state = beginning
        iter = 0
        self.travel = [beginning]
        while self.state != end and iter < max_iterations:
            if self.state == end:
                print(f"Arrivé : {self.state}/{end}")
            self.state = max(self.Q_table[self.state], key=self.Q_table[self.state].get)
            self.travel.append(self.state)

            iter += 1
        return self.travel

    def choiceAction(self, decay =  0.001):
        self.epsilon = max(0.1, self.epsilon * np.exp(-decay))         
        x = np.random.uniform(0,1)
        if x < self.epsilon:
            self.next_best_action = random.choice(self.graph_cities[self.state])
        else:
            self.next_best_action = max(self.Q_table[self.state], key=self.Q_table[self.state].get)
        return self.next_best_action





