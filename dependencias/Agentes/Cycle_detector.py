from dependencias.Agentes.Heuristic_Agent import Heuristic_Agent
from random import choice, random
from dependencias.Agentes.Busqueda_anchura import Busqueda_anchura
from dependencias.Agentes.Chaser import ChaserAgent


class Cycle_detector(Heuristic_Agent):
    ## Este agente detectara si el agente esta desarrollando un comportaiento ciclico, y si es as seleccionara el mejor movimiento que se salga sel ciclo

    def __init__(self, avoid_skipable_loops = False):
        super().__init__()
        self.puntuacion = None
        self.secuencia_principal = []
        self.secuencia_secundaria = []
        self.punto_repeticion = None
        self.in_loop = False
        self.ciclo = []
        self.avoid_skipable_loops = avoid_skipable_loops


    def get_rewards(self, state):

        REWARD = 200

        def reset_sequence():
            self.puntuacion = len(state.snake)
            self.punto_repeticion = None
            self.secuencia_secundaria = []
            self.secuencia_principal = []
            self.in_loop = False
            self.ciclo = []
            self.rewards = {
                (0, 1): 0,
                (0, -1): 0,
                (1, 0): 0,
                (-1, 0): 0
            }


        def reward():
            self.rewards = {
                (0, 1): 0,
                (0, -1): 0,
                (1, 0): 0,
                (-1, 0): 0
            }

            if self.avoid_skipable_loops:

                aux_anchura = Busqueda_anchura()
                aux_anchura.get_rewards(state)
                aux_chasser = ChaserAgent()
                aux_chasser.get_rewards(state)


                for mov, reward in aux_anchura.rewards.items():
                    self.rewards[mov] = reward * REWARD * random() + aux_chasser.rewards[mov] * REWARD

            else:
                self.rewards[choice(list(self.rewards.keys()))] = REWARD

            '''
            premiar de forma aleatorio sale del bucle pero muchas veces mata al agente
            '''
            #print("Trying to break the loop")

        if state.n_moviminetos == 0:
            reset_sequence()

        if self.in_loop:
            if state.snake[0] in self.secuencia_secundaria:
                reward()
            else:
                reset_sequence()

        else:

            #añadimos la cabeza a la secuencia principal
            self.secuencia_principal.append(state.snake[0])

            #no se ha comido comida
            if len(state.snake) == self.puntuacion:
                # comprobaos si acabamos de encontrar un punto de repeticion


                # buscamos si estamos en una casilla visitada y guardamos el ciclo
                if state.snake[0] in self.secuencia_principal[:-1] and not self.punto_repeticion:
                    self.punto_repeticion = state.snake[0]
                    i_0, i_f = [i for i in range(len(self.secuencia_principal)) if self.secuencia_principal[i] == self.punto_repeticion][:2]
                    self.ciclo = self.secuencia_principal[i_0:i_f + 1].copy()

                # En caso de que estemos evaluando una repeticion, comprobamos si hemos vuelto a la secuencia principal
                if self.punto_repeticion:
                    self.secuencia_secundaria.append(state.snake[0])

                    #comprobamos si hemos cerrado el ciclo
                    if self.secuencia_secundaria == self.ciclo:
                        self.in_loop = True
                    #coprobamos si seguimos recorriendo el ciclo.
                    elif self.secuencia_secundaria == self.ciclo[:len(self.secuencia_secundaria)]:
                        pass
                    # nos hemos salido del ciclo
                    else:
                        reset_sequence()

            # cuando come consideraos que se reinicia la busqueda de ciclos
            else:
                reset_sequence()


        if self.in_loop:
            reward()