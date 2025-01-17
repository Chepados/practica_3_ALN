from dependencias.Agentes.Heuristic_Agent import Heuristic_Agent

class Avoid_inmediate_death(Heuristic_Agent):
    ## It avoids the snake to die in the next step

    def __init__(self):
        super().__init__()

    def get_rewards(self, state):

        #Reiniciamos el estado de los pesos
        self.rewards = {
            (0, 1): 0,
            (0, -1): 0,
            (1, 0): 0,
            (-1, 0): 0
        }

        PUNISHMENT = -1

        POSIBLE_ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        head = state.snake[0]

        # Castigamos las acciones que nos llevan a chocar con la serpiente

        if (head[0] + 1, head[1]) in state.snake[:-1]:
            self.rewards[(1, 0)] += PUNISHMENT
        if (head[0] - 1, head[1]) in state.snake[:-1]:
            self.rewards[(-1, 0)] += PUNISHMENT
        if (head[0], head[1] + 1) in state.snake[:-1]:
            self.rewards[(0, 1)] += PUNISHMENT
        if (head[0], head[1] - 1) in state.snake[:-1]:
            self.rewards[(0, -1)] += PUNISHMENT

        # Castigamos las acciones que nos llevan a la pared

        if head[0] == 0:
            self.rewards[(-1, 0)] += PUNISHMENT
        if head[0] == state.shape[0] - 1:
            self.rewards[(1, 0)] += PUNISHMENT
        if head[1] == 0:
            self.rewards[(0, -1)] += PUNISHMENT
        if head[1] == state.shape[1] - 1:
            self.rewards[(0, 1)] += PUNISHMENT

