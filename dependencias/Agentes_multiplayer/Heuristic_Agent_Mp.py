from abc import ABC, abstractmethod
from random import choice

class Heuristic_Agent_Mp(ABC):
    def __init__(self, id):
        self.posible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.rewards = {
            (0, 1): 0,
            (0, -1): 0,
            (1, 0): 0,
            (-1, 0): 0
        }
        self.id = id

    def get_rewards(self, state, pos):
        """Este metodo devuelve un diccionario con los pesos asociados a los 4 movimientos posibles, en funcion del heuristico."""

    def get_action(self, state, pos):
        """Este metodo devuelve la accion a tomar en funcion del estado actual."""
        # Devolvemos la accion con mayor peso en caso de empate uno al azar de los que tengan el mayor peso

        self.get_rewards(state, pos)


        max_reward = max(self.rewards.values())
        best_actions = [action for action, reward in self.rewards.items() if reward == max_reward]

        return choice(best_actions)
    