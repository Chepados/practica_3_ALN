from random import choice, random
import numpy as np
from IPython.display import clear_output
from os import system
import pygame
from pygame.locals import QUIT
from tqdm import tqdm



def clear_console():
    clear_output(wait=True)
    system('cls')

class Game_state:

    ## Class to represent the state of the game

    def __init__(self, n_food, shape = (15, 17)):
        self.shape = shape
        self.n_food = n_food
        self.reset()
        self.is_game_over = False
        self.n_moviminetos =  0
        self.has_won = False
        self.ate_food = False

    def generate_food(self):

        posibles_posiciones = []

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if (i, j) not in self.snake and (i, j) not in self.food_list:
                    posibles_posiciones.append((i, j))

        if posibles_posiciones:
            return choice(posibles_posiciones)
        else:
            return None


    def update_food_list(self):
        # Hay ue tener en cuenta que la comida no puede estar en la serpiente, que no puede estar en la misma posición que otra comida y ue el juego se puede acabar

        while len(self.food_list) < self.n_food:

            new_food = self.generate_food()

            if new_food:
                self.food_list.add(new_food)
            else:
                break


    def reset(self):


        self.snake = [(1,0),(0,0),(0,0)]
        self.direction = (1, 0)
        self.food_list = set()
        self.update_food_list()
        self.is_game_over = False
        self.n_moviminetos = 0

    def __str__(self):
        dic_elementos = {
            'espacio' : ' ',
            'comida' : '*',
            'cabeza' : 'O',
            'cuerpo' : 'o'
        }

        string = ''

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if (i, j) in self.food_list:
                    string += dic_elementos['comida']
                elif (i, j) == self.snake[0]:
                    string += dic_elementos['cabeza']
                elif (i, j) in self.snake[1:]:
                    string += dic_elementos['cuerpo']
                else:
                    string += dic_elementos['espacio']

            string += '\n'

        return string

    def game_over(self):

        # Tiene que devolver toda la chicha
        self.is_game_over = True

        estadisticas = {
            "puntuacion" : len(self.snake),
            "n_moviminetos" : self.n_moviminetos
        }

        return estadisticas


    def update(self, move):

        self.n_moviminetos += 1

        # Suamos el movimiento que queremos hacer
        new_head = (self.snake[0][0] + move[0], self.snake[0][1] + move[1])

        # No podemos ir en la dirección contraria a la que vamos

        if len(self.snake) > 1 and (new_head[0] - self.snake[1][0], new_head[1] - self.snake[1][1]) == (-self.direction[0], -self.direction[1]):
            # Vamos en el mismo sentido que ibamos antes
            move = self.direction
            new_head = (self.snake[0][0] + move[0], self.snake[0][1] + move[1])



        # Gestionamos el avance y crecimiento de la serpiente

        if new_head in self.food_list:

            #Miraos si hemos ganado

            if len(self.snake) + 1 == self.shape[0] * self.shape[1]:
                print("SENOLLOP HA GANADO GRACIAS POR JUGAR")
                self.has_won = True
                self.game_over()

            self.ate_food = True
            self.snake = [new_head] + self.snake
            self.food_list.remove(new_head)
            self.update_food_list()
        else:
            self.ate_food = False
            self.snake = [new_head] + self.snake[:-1]

        # Gestionamos la dirección de la serpiente

        self.direction = move

        # Gestionamos la colisión con las paredes

        if new_head[0] < 0 or new_head[0] >= self.shape[0] or new_head[1] < 0 or new_head[1] >= self.shape[1]:
            self.game_over()

        # Gestionamos la colisión con la serpiente

        if new_head in self.snake[1:]:
            self.game_over()

    def state_matrix(self):
        state_matrix = np.zeros((self.shape[0], self.shape[1]))

        dic_elementos = {
            'espacio' : 0,
            'comida' :  3,
            'cabeza' :  2,
            'cuerpo' :  1
        }

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if (i, j) in self.food_list:
                    state_matrix[i, j] = dic_elementos['comida']
                elif (i, j) == self.snake[0]:
                    state_matrix[i, j] = dic_elementos['cabeza']
                elif (i, j) in self.snake[1:]:
                    state_matrix[i, j] = dic_elementos['cuerpo']
                else:
                    state_matrix[i, j] = dic_elementos['espacio']

        return state_matrix

    def state_matrix_cnn(self):
        # Número de canales: 4 (espacio libre, cuerpo, cabeza, comida)
        channels = 4
        state_matrix = np.zeros((channels, self.shape[0], self.shape[1]))

        # Definir índices de canales para cada elemento
        CHANNEL_EMPTY = 0
        CHANNEL_BODY = 1
        CHANNEL_HEAD = 2
        CHANNEL_FOOD = 3

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if (i, j) in self.food_list:
                    state_matrix[CHANNEL_FOOD, i, j] = 1
                if (i, j) == self.snake[0]:
                    state_matrix[CHANNEL_HEAD, i, j] = 1
                if (i, j) in self.snake[1:]:
                    state_matrix[CHANNEL_BODY, i, j] = 1
                if (i, j) not in self.food_list and (i, j) not in self.snake:
                    state_matrix[CHANNEL_EMPTY, i, j] = 1

        return state_matrix

class Snake_game:
    def __init__(self, size, n_food, agent):
        self.state = Game_state(n_food, size)
        self.agent = agent

    def play(self):
        while True:
            print(self.state)
            action = self.agent.get_action(self.state)
            self.state.update(action)   # Actualizamos el estado del juego
            if self.state.is_game_over:
                break

    def play_with_pygame(self):
        pygame.init()
        cell_size = 15
        screen = pygame.display.set_mode((self.state.shape[1] * cell_size, self.state.shape[0] * cell_size))
        clock = pygame.time.Clock()

        while not self.state.is_game_over:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return

            action = self.agent.get_action(self.state)
            self.state.update(action)

            screen.fill((0, 0, 0))

            pygame.draw.rect(screen, (230, 54, 241),(self.state.snake[0][1] * cell_size, self.state.snake[0][0] * cell_size, cell_size, cell_size))

            v = 0

            for segment in self.state.snake[1:]:
                v += 1
                #((241 - v) % 255, 173, (54 + v) % 255)
                pygame.draw.rect(screen, ((241 - v) % 255, 173, (54 + v) % 255), (segment[1] * cell_size, segment[0] * cell_size, cell_size, cell_size))
            for food in self.state.food_list:
                pygame.draw.rect(screen, (255, 0, 0), (food[1] * cell_size, food[0] * cell_size, cell_size, cell_size))

            pygame.display.flip()
            clock.tick(10)

        pygame.quit()

    def evaluar(self, n_partidas=100):
        puntuaciones = []
        movimientos = []
        for _ in tqdm(range(n_partidas)):
            self.state.reset()
            self.state.is_game_over = False  # Asegúrate de que el estado del juego se reinicie correctamente
            while not self.state.is_game_over:
                action = self.agent.get_action(self.state)
                self.state.update(action)
            puntuaciones.append(len(self.state.snake))
            movimientos.append(self.state.n_moviminetos)

        estadisticas = {
            "puntuacion_media": round(np.mean(puntuaciones), 2),
            "puntuacion_maxima": np.max(puntuaciones),
            "puntuacion_minima": np.min(puntuaciones),
            "movimientos_medios": round(np.mean(movimientos), 2),
            "movimientos_maximos": np.max(movimientos),
            "movimientos_minimos": np.min(movimientos),
            "movimientos por puntuacion": round(np.mean(movimientos) / np.mean(puntuaciones)),
            "proporcion_del_tablero_ocupada" : round(len(self.state.snake) / (self.state.shape[0] * self.state.shape[1]), 2)
        }

        return estadisticas

def enfrentar(a1, a2, n_partidas=100, size=(15, 15), n_food=5):

    def return_with_highlights(v1, v2, reverse=False):

        if reverse:
            if v1 < v2:
                return [f'((*{v1}*))', f'{v2}']
            else:
                return [f'{v1}', f'((*{v2}*))']

        else:
            if v1 > v2:
                return [f'((*{v1}*))', f'{v2}']
            else:
                return [f'{v1}', f'((*{v2}*))']

    # Imprime una comparacion entre 2 agentes
    game_1 = Snake_game(size, n_food, a1)
    game_2 = Snake_game(size, n_food, a2)

    stats_1 = game_1.evaluar(n_partidas)
    stats_2 = game_2.evaluar(n_partidas)

    print(f"{'Estadisticas':^30} | {'Agente 1':^15} | {'Agente 2':^15}")
    print(f"{'-' * 30} | {'-' * 15} | {'-' * 15}")

    estaditicas = stats_1.keys()

    for i, stat in enumerate(estaditicas):

        v1, v2 = return_with_highlights(stats_1[stat], stats_2[stat])
        if stat == 'movimientos por puntuacion':
            v1, v2 = return_with_highlights(stats_1[stat], stats_2[stat], reverse=True)

        print(f"{stat:<30} | {v1:^15} | {v2:^15}")

    print(f"{'-' * 30} | {'-' * 15} | {'-' * 15}")
