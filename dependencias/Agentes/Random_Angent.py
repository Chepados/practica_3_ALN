from dependencias.Agentes.Base_Agent import Agent
from random import choice

class RandomAgent(Agent):
    def get_action(self, state):
        POSIBLE_ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return choice(POSIBLE_ACTIONS)