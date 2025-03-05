import env
import agent
import reward
import simulate

import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_counts(data):

    x = np.arange(len(data))  # Indices de la liste
    y1, y2 = [], []

    # Calculer les sommes cumulées pour chaque composante des tuples
    cumulative_sum_1 = 0
    cumulative_sum_2 = 0
    for t in data:
        cumulative_sum_1 += t[0]
        cumulative_sum_2 += t[1]
        y1.append(cumulative_sum_1)
        y2.append(cumulative_sum_2)

    # Tracer les courbes cumulées
    plt.plot(x, y1, label="Cumsum evacuated agent", color='green')
    plt.plot(x, y2, label="cumsum deactivated agent", color='red')

    # Ajouter des labels et une légende
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Value")
    plt.legend()
    plt.show()

trained_agent, all_rewards, all_results = simulate.train('config.json')

# # Plot the cumulated rewards per episode
simulate.plot_cumulated_rewards(all_rewards)



# Afficher le graphiq

plot_cumulative_counts(all_results)