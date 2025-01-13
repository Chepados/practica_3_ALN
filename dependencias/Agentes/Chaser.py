from dependencias.Agentes.Base_Agent import Agent

class ChaserAgent(Agent):
    ## Agent that tries to catch the food as soon as possible

    def get_action(self, state):
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

        # Eliminamos las acciones que nos llevan a chocar con la serpiente

        if (head[0] + 1, head[1]) in state.snake:
            acciones_posibles.remove((1, 0))
        if (head[0] - 1, head[1]) in state.snake:
            acciones_posibles.remove((-1, 0))
        if (head[0], head[1] + 1) in state.snake:
            acciones_posibles.remove((0, 1))
        if (head[0], head[1] - 1) in state.snake:
            acciones_posibles.remove((0, -1))

        # Eliminamos las acciones que nos llevan a la pared

        if head[0] == 0:
            acciones_posibles.remove((-1, 0))
        if head[0] == state.shape[0] - 1:
            acciones_posibles.remove((1, 0))
        if head[1] == 0:
            acciones_posibles.remove((0, -1))
        if head[1] == state.shape[1] - 1:
            acciones_posibles.remove((0, 1))




        # de los movimientos restantes no quedamos con uno que nos acerue a best food

        if not acciones_posibles:
            return (1, 0)

        for accion in acciones_posibles:
            if abs(head[0] + accion[0] - best_food[0]) + abs(head[1] + accion[1] - best_food[1]) < min_distance:
                return accion
        return acciones_posibles[0]

