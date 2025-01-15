from abc import ABC, abstractmethod
from random import choice
from time import sleep
import numpy as np
from IPython.display import clear_output
from os import system
import pygame
from pygame.locals import QUIT
from tqdm import tqdm


def clear_console():
    clear_output(wait=True)
    system('cls')

class Game_state_MultiPlayer:

    ## Class to represent the state of the game in a Multiplayer game

    def __init__(self, n_food, n_snakes, shape = (30, 35)):
        self.shape = shape
        self.n_food = n_food
        self.n_snakes = n_snakes
        self.reset()

    def generate_food(self):

        posibles_posiciones = []

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if (i, j) not in self.food_list and (i, j) not in self.bodies:
                    posibles_posiciones.append((i, j))

        if posibles_posiciones:
            return choice(posibles_posiciones)
        else:
            return None


    def update_food_list(self):
        # Hay que tener en cuenta que la comida no puede estar en la serpiente, que no puede estar en la misma posición que otra comida y ue el juego se puede acabar

        while len(self.food_list) < self.n_food:

            new_food = self.generate_food()

            if new_food:
                self.food_list.add(new_food)
            else:
                break


    def reset(self):

        BEGIN_POSITIONS = [(0, 0),
                           (0, self.shape[1]),
                           (self.shape[0], self.shape[1]),
                           (self.shape[0], 0),
                           (0, round(self.shape[1] / 2)),
                           (round(self.shape[0] / 2), self.shape[1]),
                           (self.shape[0], round(self.shape[1] / 2)),
                           (round(self.shape[0] / 2), 0)
                           ]
        
        self.snakes = [[BEGIN_POSITIONS[i]] for i in range(self.n_snakes)]
        self.bodies = set([tupla for serp in self.snakes for tupla in serp])  # Contiene la info de TODOS los cuerpos de serpientes en un mismo bloque
        self.directions = [(0, 0) for i in range(self.n_snakes)]
        self.food_list = set()
        self.update_food_list()
        self.is_game_over = False

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
                elif (i, j) in self.bodies:
                    string += dic_elementos['cuerpo']
                else:
                    string += dic_elementos['espacio']

            string += '\n'

        return string

    def game_over(self):

        # Tiene que devolver toda la chicha
        self.is_game_over = True


    def update(self, moves):

        # Sumamos el movimiento que queremos hacer y sacamos todas las cabezas en orden
        new_heads = [(self.snakes[i][0][0] + move[0], self.snakes[i][0][1] + move[1]) for i, move in enumerate(moves)]

        # Gestionamos colision de cabezas (FUTURO IMPLEMENTAR)

        # Gestionamos la colisión con las paredes
        for i, new_head in enumerate(new_heads):
            if new_head[0] < 0 or new_head[0] >= self.shape[0] or new_head[1] < 0 or new_head[1] >= self.shape[1]:
                self.snakes.pop(i)
                new_heads.pop(i)
                
        # Gestionamos la colisión con la serpiente
            if new_head in self.bodies:
                self.snakes.pop(i)
                new_heads.pop(i)

        # Gestionamos fin del juego
        if len(self.snakes) == 1:
            self.game_over()

        # Gestionamos el avance y crecimiento de las serpientes
        for i, new_head in enumerate(new_heads):
            if new_head in self.food_list:
                self.snakes[i] = [new_head] + self.snakes[i]
                self.food_list.remove(new_head)
                self.update_food_list()
            else:
                self.snakes[i] = [new_head] + self.snakes[i][:-1]

        self.bodies = set([tupla for serp in self.snakes for tupla in serp])  # Actualizamos toda la Info de los cuerpos


    def state_matrix(self, snake_id):

        state_matrix = np.zeros(self.shape[0], self.shape[1])

        dic_elementos = {
            'espacio' : 0,
            'comida' :  3,
            'cabeza' :  2,
            'cuerpo' :  1,
            "cuerpo_ajeno" : 4
        }

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if (i, j) in self.food_list:
                    state_matrix[i, j] = dic_elementos['comida']
                elif (i, j) == self.snakes[snake_id][0]:
                    state_matrix[i, j] = dic_elementos['cabeza']
                elif (i, j) in self.snakes[snake_id][1:]:
                    state_matrix[i, j] = dic_elementos['cuerpo']
                elif (i, j) in self.bodies.intersection(set(self.snakes[snake_id])):  # Si esta en la matriz de cuerpos
                    state_matrix[i, j] = dic_elementos['cuerpo_ajeno']
                else:
                    state_matrix[i, j] = dic_elementos['espacio']

        return state_matrix


class Snake_game_MultiPlayer:
    def __init__(self, size, n_food, agents: iter):
        self.state = Game_state_MultiPlayer(n_food, len(agents), size)
        self.agents = list(agents)

    def play(self):
        while True:
            print(self.state)
            actions = [agent.get_action(self.state) for agent in self.agents]
            self.state.update(actions) 
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

            actions = [agent.get_action(self.state) for agent in self.agents]
            self.state.update(actions)

            screen.fill((0, 0, 0))
            
            v = 0  # una variación del color para diferenciar serpientes
            for snake in self.state.snakes:
                pygame.draw.rect(screen, ((230 + v) % 255, 54, 241),(snake[0][1] * cell_size, snake[0][0] * cell_size, cell_size, cell_size))

                for segment in snake[1:]:
                    pygame.draw.rect(screen, ((241 + v) % 255, 173, 54), (segment[1] * cell_size, segment[0] * cell_size, cell_size, cell_size))

                v += 15
                
            for food in self.state.food_list:
                pygame.draw.rect(screen, (255, 0, 0), (food[1] * cell_size, food[0] * cell_size, cell_size, cell_size))

            pygame.display.flip()
            clock.tick(15)

        pygame.quit()