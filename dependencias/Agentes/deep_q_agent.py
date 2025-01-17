import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from IPython import display

# Seed Initialization for Reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    
class DeepQNetworkN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, learning_rate):

        super(DeepQNetworkN, self).__init__()

        self.input_dim = input_dim
        self.n_actions = n_actions

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.SmoothL1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.model(state)

class DQNAgentN:
    def __init__(self, gamma, epsilon, epsilon_min, epsilon_decay,  # Parámetros para el agente.
                learning_rate, input_dim, hidden_dim, n_actions,    # Parámetros para la red neuronal.
                batch_size, mem_size,                               # Parámetros para la memoria
                target_update_freq=100,                             # Parámetros para el target network.
                checkpoint_dir='checkpoints',                       # Directorio para guardar los checkpoints.
                state_function = None):
        
        self.state_function = state_function
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.action_space = [i for i in range(n_actions)]   # Representación en int de las acciones posibles. Es para la selección de épsilon greedy.
        
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        self.batch_size = batch_size
        self.mem_size = mem_size
        self.step = 0   # Para saber cuándo la memoria está llena y cuándo se puede empezar a entrenar el modelo con un tamaño de batch adecuado.
        self.memory = deque(maxlen=self.mem_size)
        
        '''Ahora vamos a inicializar las redes neuronales y la memoria'''
        self.q_eval = DeepQNetworkN(self.input_dim, self.hidden_dim,
                                     self.n_actions, self.learning_rate)
        self.q_target = DeepQNetworkN(self.input_dim, self.hidden_dim,
                                       self.n_actions, self.learning_rate)
        self.target_update_freq = target_update_freq
        self.update_target_network()
        
        # Chekpoint
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def update_target_network(self):
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def store_transition(self, state, action, reward, state_, done):
        self.memory.append((state, action, reward, state_, done))
        self.step += 1

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor([observation], dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        return action
    
    def get_action(self, state):
        state_vector = self.state_function(state)
        action_chosen = self.choose_action(state_vector)
        action_conv = {0:(0, 1), 1:(0, -1), 2:(1, 0), 3:(-1, 0)}
        return action_conv[action_chosen]
    
    def get_action_mem(self, action):
        action_conv = {(0, 1):0, (0, -1):1, (1, 0):2, (-1, 0):3}
        return action_conv[action]
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            
    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, states_, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float).to(self.q_eval.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.q_eval.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.q_eval.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.q_eval.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.q_eval.device)
        
        q_eval = self.q_eval(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            q_next = self.q_target(states_).max(1)[0]
        q_target = rewards + self.gamma * q_next * (1 - dones)
        
        loss = self.q_eval.loss(q_target, q_eval).to(self.q_eval.device)
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        if self.step % self.target_update_freq == 0:
            self.update_target_network()
            
    def save_checkpoint(self, episode, N, mean_score):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'dqn{N}N_ep{episode}_sc{mean_score:.2f}.pth')
        torch.save({
            'episode': episode,
            'model_state_dict': self.q_eval.state_dict(),
            'optimizer_state_dict': self.q_eval.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'mean_score': mean_score
        }, checkpoint_path)
        print(f"Checkpoint saved at Episode {episode} with Mean Score {mean_score:.2f}")
    
    def load_checkpoint(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.q_eval.load_state_dict(checkpoint['model_state_dict'])
            self.q_target.load_state_dict(self.q_eval.state_dict())
            self.q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Checkpoint loaded from Episode {checkpoint['episode']} with Mean Score {checkpoint['mean_score']:.2f}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
            
            
            
            
def state_function_8n(game_state):
    state_vector = np.zeros(8, dtype=float)

    head_x, head_y = game_state.snake[0]

    # Directions: Up, Right, Down, Left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # Immediate danger
    for idx, (dx, dy) in enumerate(directions):
        next_x, next_y = head_x + dx, head_y + dy
        # Check wall collision
        if next_x < 0 or next_x >= game_state.shape[0] or next_y < 0 or next_y >= game_state.shape[1]:
            state_vector[idx] = 1.0
        # Check self collision
        elif (next_x, next_y) in game_state.snake:
            state_vector[idx] = 1.0
        else:
            state_vector[idx] = 0.0

    # Food direction
    food_x, food_y = next(iter(game_state.food_list)) if game_state.food_list else (head_x, head_y)
    
    if food_x < head_x:
        state_vector[4] = 1.0  # Food Up
    elif food_x > head_x:
        state_vector[6] = 1.0  # Food Down
    else:
        state_vector[4] = 0.0
        state_vector[6] = 0.0

    if food_y < head_y:
        state_vector[7] = 1.0  # Food Left
    elif food_y > head_y:
        state_vector[5] = 1.0  # Food Right
    else:
        state_vector[5] = 0.0
        state_vector[7] = 0.0

    return state_vector

def state_function_15(game_state):
    state_vector = []

    # 1. Dangers: [front, right, left]
    head_x, head_y = game_state.snake[0]
    current_direction = game_state.direction

    # Definir movimientos según la dirección actual
    if current_direction == 'up':
        front = (-1, 0)
        right = (0, 1)
        left = (0, -1)
    elif current_direction == 'right':
        front = (0, 1)
        right = (1, 0)
        left = (-1, 0)
    elif current_direction == 'down':
        front = (1, 0)
        right = (0, -1)
        left = (0, 1)
    elif current_direction == 'left':
        front = (0, -1)
        right = (-1, 0)
        left = (1, 0)
    else:
        front = (0, 0)
        right = (0, 0)
        left = (0, 0)

    # Verificar peligros en front, right y left
    dangers = []
    for move in [front, right, left]:
        new_x = head_x + move[0]
        new_y = head_y + move[1]
        # Peligro si hay una pared o si la posición está ocupada por la serpiente
        if (new_x < 0 or new_x >= game_state.shape[0] or
            new_y < 0 or new_y >= game_state.shape[1] or
            (new_x, new_y) in game_state.snake):
            dangers.append(1)
        else:
            dangers.append(0)
    state_vector.extend(dangers)

    # 2. Direction: [up, right, down, left]
    directions = ['up', 'right', 'down', 'left']
    direction_vector = [1 if current_direction == dir else 0 for dir in directions]
    state_vector.extend(direction_vector)

    # 3. Food Location: [left, right, up, down]
    food_list = game_state.food_list
    if food_list:
        food_x, food_y = next(iter(food_list))
    else:
        food_x, food_y = head_x, head_y  # Si no hay comida, la posición de la cabeza

    food_left = 1 if food_y < head_y else 0
    food_right = 1 if food_y > head_y else 0
    food_up = 1 if food_x < head_x else 0
    food_down = 1 if food_x > head_x else 0
    food_direction = [food_left, food_right, food_up, food_down]
    state_vector.extend(food_direction)

    # 4. Accessible Spaces: [left, right, up, down]

    def count_BFS(start_pos):
        queue = deque([start_pos])
        visited = set([start_pos])
        while queue:
            pos = queue.popleft()
            for accion in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = (pos[0] + accion[0], pos[1] + accion[1])
                if (0 <= new_pos[0] < game_state.shape[0] and
                    0 <= new_pos[1] < game_state.shape[1] and
                    new_pos not in game_state.snake and
                    new_pos not in visited):
                    queue.append(new_pos)
                    visited.add(new_pos)
        return len(visited) / (game_state.shape[0] * game_state.shape[1])

    accessible_spaces = []
    movements = {'left': (0, -1), 'right': (0, 1), 'up': (-1, 0), 'down': (1, 0)}
    for direction in ['left', 'right', 'up', 'down']:
        move = movements[direction]
        new_x = head_x + move[0]
        new_y = head_y + move[1]
        new_pos = (new_x, new_y)
        if (0 <= new_x < game_state.shape[0] and
            0 <= new_y < game_state.shape[1] and
            new_pos not in game_state.snake):
            accessible = count_BFS(new_pos)
            accessible_spaces.append(accessible)
        else:
            accessible_spaces.append(0)
    state_vector.extend(accessible_spaces)

    return state_vector

def state_big_matrix(game_state):
    return game_state.state_matrix().flatten()


def plot_scores_function(scores, mean_scores, epsilons, id, neuronas):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.plot(epsilons, label='Epsilon')
    plt.legend()
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.savefig(f'media/imagenes/rl/plots/dqn{neuronas}N_ep{id}_sc{mean_scores}.png')
    plt.show()

def train_rl_agent(episodes, play_save_every, game, agent,state_function,
                   n_games, total_score, plot_epsilons, plot_scores, plot_mean_scores):
    
    for e in range(1, episodes+1):
        n_games += 1
        state = game.state
        state_vector = state_function(state)  # Use compact state
        done = False
        
        while not done:
            # Elegir acción
            action_game = agent.get_action(state)
            action_mem = agent.get_action_mem(action_game)
            
            # Realizar acción
            game.state.update(action_game)
            next_state = game.state
            
            # Recompensa
            if game.state.is_game_over:
                reward = -1
            elif game.state.ate_food == True:
                reward = 1
            else:
                reward = 0
                
            # Guardar transición
            next_state_vector = state_function(next_state)  # Use compact state
            agent.store_transition(state_vector, action_mem, reward, next_state_vector, int(game.state.is_game_over))
            state_vector = next_state_vector
            
            # Entrenar agente
            if len(agent.memory) >= agent.batch_size:
                agent.train()
                
            if game.state.is_game_over:
                # Crear estadísticas para plotearlas.
                score = len(game.state.snake)
                plot_scores.append(score)
                if n_games >= 50:
                    mean_score = np.sum(plot_scores[-50:]) / 50
                    plot_mean_scores.append(mean_score)
                else:
                    total_score += score
                    mean_score = total_score / n_games
                    plot_mean_scores.append(mean_score)
                plot_epsilons.append(agent.epsilon)
                agent.update_epsilon()
                print(f"Episode: {e}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.4}")
                
                
                # Save Checkpoint
                if e % play_save_every == 0:
                    agent.save_checkpoint(episode=e, mean_score=mean_score, N=agent.input_dim)
                    game.state.reset()
                    epsilon_actual = agent.epsilon
                    agent.epsilon = 0.0
                    game.play_with_pygame()
                    agent.epsilon = epsilon_actual
                    plot_scores_function(plot_scores, plot_mean_scores, plot_epsilons, id=e, neuronas=agent.input_dim)
                    
                game.state.reset()
                done = True
                