from typing import Union
import numpy as np
import random







class QLearningGraphTravel:



    def __init__(self, graphcities, epsilon = 0.8, value_init = 0):
        self.graph_cities = graphcities
        self.Q_table = {city: {action: value_init for action in graphcities[city]} for city in graphcities}
        self.travel = []
        self.history_epsilon = epsilon
        self.epsilon = epsilon
        self.state = None
        self.next_best_action = None
    

    def train(self,beginning,end, epochs = 10, limit = 1000, lr = 0.001,gamma = 0.9, reward = -6, verbose = True):
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
                    break

        
            
            if verbose:
                print(f"Epoch {x + 1}/{epochs} - Total Rewards: {total_reward}")
                print(f"Path: {self.travel}")     
                

    def test(self,beginning,end, max_iterations = 100):
        
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





