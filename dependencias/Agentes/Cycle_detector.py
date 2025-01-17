from dependencias.Agentes.Heuristic_Agent import Heuristic_Agent


class Tail_Chasser(Heuristic_Agent):
    ## Este agente detectara si el agente esta desarrollando un comportaiento ciclico, y si es as seleccionara el mejor movimiento que se salga sel ciclo

    def __init__(self):
        super().__init__()
        self.puntuacion = None
        self.ciclo = []

    def get_rewards(self, state):


        if len(state.snake) == self.puntuacion:
            #En este caso podemos estar en un ciclo
            if state.snake[0] in self.ciclo:
                #podemos estar en un cliclo




        REWARD = 1

        #Reiniciamos el estado de los pesos
        self.rewards = {
            (0, 1): 0,
            (0, -1): 0,
            (1, 0): 0,
            (-1, 0): 0
        }



