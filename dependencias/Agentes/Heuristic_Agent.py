from dependencias.Agentes.Base_Agent import Agent
from abc import ABC, abstractmethod
from random import choice

class Heuristic_Agent(ABC):
    def __init__(self):
        self.posible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.rewards = {
            (0, 1): 0,
            (0, -1): 0,
            (1, 0): 0,
            (-1, 0): 0
        }

    def get_rewards(self, state):
        """Este metodo devuelve un diccionario con los pesos asociados a los 4 movimientos posibles, en funcion del heuristico."""

    def get_action(self, state):
        """Este metodo devuelve la accion a tomar en funcion del estado actual."""
        # Devolvemos la accion con mayor peso en caso de empate uno al azar de los que tengan el mayor peso

        self.get_rewards(state)


        max_reward = max(self.rewards.values())
        best_actions = [action for action, reward in self.rewards.items() if reward == max_reward]

        return choice(best_actions)

