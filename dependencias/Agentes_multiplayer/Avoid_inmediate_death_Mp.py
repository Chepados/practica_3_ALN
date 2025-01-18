from dependencias.Agentes_multiplayer.Heuristic_Agent_Mp import Heuristic_Agent_Mp

class Avoid_inmediate_death_Mp(Heuristic_Agent_Mp):
    ## It avoids the snake to die in the next step

    def __init__(self, id):
        super().__init__(id)

    def get_rewards(self, state, pos):

        #Reiniciamos el estado de los pesos
        self.rewards = {
            (0, 1): 0,
            (0, -1): 0,
            (1, 0): 0,
            (-1, 0): 0
        }

        PUNISHMENT = -1

        POSIBLE_ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        head = state.snakes[pos][0]

        # Castigamos las acciones que nos llevan a chocar con la serpiente

        if (head[0] + 1, head[1]) in state.bodies:
            self.rewards[(1, 0)] += PUNISHMENT
        if (head[0] - 1, head[1]) in state.bodies:
            self.rewards[(-1, 0)] += PUNISHMENT
        if (head[0], head[1] + 1) in state.bodies:
            self.rewards[(0, 1)] += PUNISHMENT
        if (head[0], head[1] - 1) in state.bodies:
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