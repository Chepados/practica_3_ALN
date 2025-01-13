from dependencias.Agentes.Heuristic_Agent import Heuristic_Agent

class Combined_agent(Heuristic_Agent):
    ## Agent that tries to catch the food as soon as possible

    def __init__(self, agentes : [Heuristic_Agent], weights = None):
        super().__init__()
        self.agentes = agentes
        self.weights_list = weights


    def get_rewards(self, state):

        #Reiniciamos el estado de los pesos
        self.rewards = {
            (0, 1): 0,
            (0, -1): 0,
            (1, 0): 0,
            (-1, 0): 0
        }

        if self.weights_list is None:
            self.weights_list = [1 for _ in self.agentes]

        for i, agente in enumerate(self.agentes):
            agente.get_rewards(state)
            for accion in agente.rewards:
                self.rewards[accion] += self.weights_list[i] * agente.rewards[accion]