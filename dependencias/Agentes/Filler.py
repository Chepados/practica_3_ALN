from dependencias.Agentes.Heuristic_Agent import Heuristic_Agent

class Filler(Heuristic_Agent):
    ## Este agente intentara rellenar el tablero de la forma mas optima posible

    def __init__(self):
        super().__init__()

    def get_rewards(self, state):

        REWARD = 1


        #Reiniciamos el estado de los pesos
        self.rewards = {
            (0, 1): 0,
            (0, -1): 0,
            (1, 0): 0,
            (-1, 0): 0
        }

        cabeza = state.snake[0]

        MOVIMIENTOS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        next_possible_pos = [(cabeza[0] + movimiento[0], cabeza[1] + movimiento[1]) for movimiento in MOVIMIENTOS]

        for start_pos, movimiento in zip(next_possible_pos, MOVIMIENTOS):
            if ((0 <= start_pos[0] < state.shape[0]) and (0 <= start_pos[1] < state.shape[1]) and (start_pos not in state.snake[:-1])):
                for next_mov in MOVIMIENTOS:
                    new_pos = (start_pos[0] + next_mov[0], start_pos[1] + next_mov[1])
                    if new_pos in state.snake[:-1]:
                        if self.rewards[movimiento] < 2:
                            self.rewards[movimiento] += REWARD








