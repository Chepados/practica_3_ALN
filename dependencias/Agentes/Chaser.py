from dependencias.Agentes.Heuristic_Agent import Heuristic_Agent

class ChaserAgent(Heuristic_Agent):
    ## Agent that tries to catch the food as soon as possible

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

        REWARD = 1

        POSIBLE_ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        head = state.snake[0]

        # Calculamos la distancia a la comida en cada dirección y nos quedamos con la mínima

        min_distance = float('inf')

        for comida in state.food_list:
            dist = abs(head[0] - comida[0]) + abs(head[1] - comida[1])
            if dist < min_distance:
                min_distance = dist
                best_food = comida

        acciones_posibles = POSIBLE_ACTIONS.copy()


        # Favorecemos las acciones que nos acercan a la comida

        for accion in acciones_posibles:
            if abs(head[0] + accion[0] - best_food[0]) + abs(head[1] + accion[1] - best_food[1]) < min_distance:
                self.rewards[accion] += REWARD