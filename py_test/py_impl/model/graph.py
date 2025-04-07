import sys
import os
sys.path.insert(0, os.path.abspath(""))

from py_impl.model.graph import QLearningGraphTravel


city_graph_weighted = {
    "Paris": {"Lyon": 465, "Bruxelles": 310},
    "Lyon": {"Marseille": 315},
    "Bruxelles": {"Amsterdam": 210, "Berlin": 765},
    "Marseille": {},
    "Amsterdam": {"Paris": 500},
    "Berlin": {"Marseille":800}
}

city_graph = {
    "Paris": ["Lyon", "Bruxelles"],
    "Lyon": ["Marseille"],
    "Bruxelles": ["Amsterdam", "Berlin"],
    "Marseille": [],
    "Berlin":["Marseille"],
    "Amsterdam": ["Paris"]
}



def custom_reward_function(state, next_action, end, graph):
    """
    Fonction de récompense pour le problème du voyageur de commerce.
    - Récompense positive pour l'atteinte de l'objectif (Marseille).
    - Pénalité basée sur la distance et les étapes.
    """
    if next_action == end:
        return 10  # Grande récompense si l'agent atteint Marseille.
    
    # Pénalité pour chaque étape
    step_penalty = -1
    
    # Récompense en fonction de la distance : inversement proportionnelle à la distance
    if next_action in graph[state]:
        distance_to_end = graph.get(next_action, {}).get(end, float('inf'))  # Distance de la ville suivante à l'objectif
        reward = step_penalty + 1 / (distance_to_end + 1)  # Plus la distance est petite, plus la récompense est élevée
    else:
        reward = step_penalty  # Pénalité si l'action n'est valide
    
    return reward


# Créer l'agent Q-learning pour le graphe pondéré
ql_agent_weighted = QLearningGraphTravel(city_graph_weighted, epsilon=0.8, value_init=0, compute_rewards=custom_reward_function)

# Entraîner l'agent avec le graphe pondéré
print("Entraînement avec graphe pondéré...")
ql_agent_weighted.train("Paris", "Marseille", epochs=10, limit=1000, lr=0.1, gamma=0.9, reward=-6, verbose=True)

# Sauvegarder la Q-table dans un fichier JSON pour le graphe pondéré
ql_agent_weighted.save("q_table_weighted_custom_reward.json")


# Créer l'agent Q-learning pour le graphe pondéré
ql_agent_weighted = QLearningGraphTravel(city_graph_weighted, epsilon=0.8, value_init=0)

# Entraîner l'agent avec le graphe pondéré
print("Entraînement avec graphe pondéré...")
ql_agent_weighted.train("Paris", "Marseille", epochs=10, limit=1000, lr=0.1, gamma=0.9, reward=-6, verbose=True)

# Sauvegarder la Q-table dans un fichier JSON pour le graphe pondéré
ql_agent_weighted.save("q_table_weighted.json")

# Tester l'agent sur le chemin trouvé avec le graphe pondéré
print("\nTest avec graphe pondéré...")
optimal_path_weighted = ql_agent_weighted.test("Paris", "Marseille")
print(f"Optimal path (weighted): {optimal_path_weighted}")

# Créer l'agent Q-learning pour le graphe non pondéré
ql_agent_unweighted = QLearningGraphTravel(city_graph, epsilon=0.8, value_init=0)

# Entraîner l'agent avec le graphe non pondéré
print("\nEntraînement avec graphe non pondéré...")
ql_agent_unweighted.train("Paris", "Marseille", epochs=10, limit=1000, lr=0.1, gamma=0.9, reward=-6, verbose=True)

# Sauvegarder la Q-table dans un fichier JSON pour le graphe non pondéré
ql_agent_unweighted.save("q_table_unweighted.json")

# Tester l'agent sur le chemin trouvé avec le graphe non pondéré
print("\nTest avec graphe non pondéré...")
optimal_path_unweighted = ql_agent_unweighted.test("Paris", "Marseille")
print(f"Optimal path (unweighted): {optimal_path_unweighted}")

# Charger et tester la Q-table sauvegardée du graphe pondéré
ql_agent_loaded = QLearningGraphTravel(city_graph_weighted)
ql_agent_loaded.load("q_table_weighted.json")

# Tester la Q-table chargée du graphe pondéré
print("\nTest avec Q-table chargée (graphe pondéré)...")
optimal_path_loaded_weighted = ql_agent_loaded.test("Paris", "Marseille")
print(f"Optimal path (loaded weighted): {optimal_path_loaded_weighted}")

# Charger et tester la Q-table sauvegardée du graphe non pondéré
ql_agent_loaded_unweighted = QLearningGraphTravel(city_graph)
ql_agent_loaded_unweighted.load("q_table_unweighted.json")

# Tester la Q-table chargée du graphe non pondéré
print("\nTest avec Q-table chargée (graphe non pondéré)...")
optimal_path_loaded_unweighted = ql_agent_loaded_unweighted.test("Paris", "Marseille")
print(f"Optimal path (loaded unweighted): {optimal_path_loaded_unweighted}")