from dependencias.Agentes_multiplayer.Heuristic_Agent_Mp import Heuristic_Agent_Mp

class Tail_Chasser_Mp(Heuristic_Agent_Mp):
    ## Este agente intenta no perder de vista la cola de la serpiente de modo que siempre pueda alcanzarla.

    def __init__(self, id):
        super().__init__(id)

    def get_rewards(self, state, pos):

        REWARD = 1

        def tail_reachable(start_pos):
            POSIBLES_ACCIONES = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            # Búsqueda en anchura para calcular las casillas accesibles desde start_pos
            visited = set()
            queue = [start_pos]
            visited.add(start_pos)
            tail_pos = state.snakes[pos][-1]


            while queue:
                posit = queue.pop(0)

                if posit == tail_pos:
                    return True

                for accion in POSIBLES_ACCIONES:
                    new_pos = (posit[0] + accion[0], posit[1] + accion[1])

                    if new_pos == tail_pos:
                        return True

                    if (0 <= new_pos[0] < state.shape[0]) and (0 <= new_pos[1] < state.shape[1]) and (
                            new_pos not in state.bodies) and (new_pos not in visited):
                        queue.append(new_pos)
                        visited.add(new_pos)

            return False

        #Reiniciamos el estado de los pesos
        self.rewards = {
            (0, 1): 0,
            (0, -1): 0,
            (1, 0): 0,
            (-1, 0): 0
        }

        cabeza = state.snakes[pos][0]

        MOVIMIENTOS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        lista_start_pos = [(cabeza[0] + movimiento[0], cabeza[1] + movimiento[1]) for movimiento in MOVIMIENTOS]


        for start_pos, movimiento in zip(lista_start_pos, MOVIMIENTOS):
            if ((0 <= start_pos[0] < state.shape[0]) and (0 <= start_pos[1] < state.shape[1]) and (start_pos not in state.bodies)):
                if tail_reachable(start_pos):
                    self.rewards[movimiento] += REWARD