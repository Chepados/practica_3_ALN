from abc import ABC, abstractmethod
from random import choice, random
from time import sleep
import os


class Game_state:
    ## Class to represent the state of the game

    def __init__(self, n_food, shape = (10, 10)):
        self.shape = shape
        self.n_food = n_food
        self.reset()
        self.is_game_over = False

    def generate_food(self):
        x = int(random() * self.shape[0])
        y = int(random() * self.shape[1])
        return (x, y)

    def generate_food_list(self):
        food_list = set()
        while len(food_list) < self.n_food:
            food_list.add(self.generate_food())
        return food_list

    def update_food_list(self):
        # Hay ue tener en cuenta que la comida no puede estar en la serpiente, que no puede estar en la misma posición que otra comida y ue el juego se puede acabar

        while len(self.food_list) < self.n_food:
            self.food_list.add(self.generate_food())

    def reset(self):


        self.snake = [(0, 0)]
        self.direction = (1, 0)
        self.food_list = self.generate_food_list()

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
        print('Game Over')
        print('Score: ', len(self.snake))
        self.is_game_over = True

    def update(self, move):
        # Suamos el movimiento que queremos hacer
        new_head = (self.snake[0][0] + move[0], self.snake[0][1] + move[1])


        # No podemos ir en la dirección contraria a la que vamos

        if len(self.snake) > 1 and (new_head[0] - self.snake[1][0], new_head[1] - self.snake[1][1]) == (-self.direction[0], -self.direction[1]):
            # Vamos en el mismo sentido que ibamos antes
            move = self.direction
            new_head = (self.snake[0][0] + move[0], self.snake[0][1] + move[1])



        # Gestionamos el avance y crecimiento de la serpiente

        if new_head in self.food_list:
            self.snake = [new_head] + self.snake
            self.food_list.remove(new_head)
            self.update_food_list()
        else:
            self.snake = [new_head] + self.snake[:-1]

        # Gestionamos la dirección de la serpiente

        self.direction = move

        # Gestionamos la colisión con las paredes

        if new_head[0] < 0 or new_head[0] >= self.shape[0] or new_head[1] < 0 or new_head[1] >= self.shape[1]:
            self.game_over()

        # Gestionamos la colisión con la serpiente

        if new_head in self.snake[1:]:
            self.game_over()





class Agent(ABC):
    ## Abstract class for the agent

    @abstractmethod
    def get_action(self, state):
        pass

class RandomAgent(Agent):
    def get_action(self, state):
        POSIBLE_ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return choice(POSIBLE_ACTIONS)

class ChasserAgent(Agent):
    ## Agent that tries to catch the food as soon as possible

    def get_action(self, state : Game_state):
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
            sleep(0.1)
            os.system('cls')





if __name__ == '__main__':
    agent = RandomAgent()
    game = Snake_game((15, 15), 5, agent)
    game.play()


