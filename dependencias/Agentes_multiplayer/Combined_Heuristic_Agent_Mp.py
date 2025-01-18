from dependencias.Agentes_multiplayer.Heuristic_Agent_Mp import Heuristic_Agent_Mp

class Combined_agent_Mp(Heuristic_Agent_Mp):

    def __init__(self, agentes : [Heuristic_Agent_Mp], id, weights = None):
        super().__init__(id)
        self.agentes = agentes
        self.weights_list = weights


    def get_rewards(self, state, pos):

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
            agente.get_rewards(state, pos)
            for accion in agente.rewards:
                self.rewards[accion] += self.weights_list[i] * agente.rewards[accion]