from dependencias.Agentes_multiplayer.Heuristic_Agent_Mp import Heuristic_Agent_Mp

class ChaserAgentMp(Heuristic_Agent_Mp):
    ## Agent that tries to catch the food as soon as possible

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

        REWARD = 1

        POSIBLE_ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        head = state.snakes[pos][0]

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
        

        # Con tal de hacer el chaser un poco mas inteligente para pruebas (esto se debe hacer combinando agentes más adelante) 
        for accion in acciones_posibles:
            new_head = (head[0] + accion[0], head[1] + accion[1])
            if 0 < new_head[0] < state.shape[0] and 0 < new_head[1] < state.shape[1]:
                self.rewards[accion] += REWARD
            
            if new_head in state.bodies:
                self.rewards[accion] -= 10