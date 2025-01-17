from dependencias.Agentes.Heuristic_Agent import Heuristic_Agent

class Aantiloop(Heuristic_Agent):
    ## Agent that tries to catch the food as soon as possible

    def __init__(self):
        super().__init__()

    def get_rewards(self, state):

        def count_BFS(start_pos):
            POSIBLES_ACCIONES = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            # Búsqueda en anchura para calcular las casillas accesibles desde start_pos
            visited = set()
            queue = [start_pos]
            visited.add(start_pos)

            while queue:
                pos = queue.pop(0)

                for accion in POSIBLES_ACCIONES:
                    new_pos = (pos[0] + accion[0], pos[1] + accion[1])
                    if (0 <= new_pos[0] < state.shape[0]) and (0 <= new_pos[1] < state.shape[1]) and (
                            new_pos not in state.snake[:-1]) and (new_pos not in visited):
                        queue.append(new_pos)
                        visited.add(new_pos)

            return len(visited)

        #Reiniciamos el estado de los pesos
        self.rewards = {
            (0, 1): 0,
            (0, -1): 0,
            (1, 0): 0,
            (-1, 0): 0
        }

        cabeza = state.snake[0]

        MOVIMIENTOS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        lista_start_pos = [(cabeza[0] + movimiento[0], cabeza[1] + movimiento[1]) for movimiento in MOVIMIENTOS]

        for start_pos, movimiento in zip(lista_start_pos, MOVIMIENTOS):
            if ((0 <= start_pos[0] < state.shape[0]) and (0 <= start_pos[1] < state.shape[1]) and (start_pos not in state.snake[:-1])):
                self.rewards[movimiento] += count_BFS(start_pos)

        casillas_accesibles = sum(self.rewards.values())

        if casillas_accesibles:
            for key in self.rewards.keys():
                self.rewards[key] /= casillas_accesibles


