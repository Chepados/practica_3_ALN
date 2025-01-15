from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Seed Initialization for Reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    
class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, learning_rate):

        super(DeepQNetwork, self).__init__()

        self.input_dim = input_dim
        self.n_actions = n_actions

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.SmoothL1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.model(state)

class DQNAgent:
    def __init__(self, gamma, epsilon, epsilon_min, epsilon_decay,  # Parámetros para el agente.
                learning_rate, input_dim, n_actions,                # Parámetros para la red neuronal.
                batch_size, mem_size,                               # Parámetros para la memoria
                target_update_freq=100,                             # Parámetros para el target network.
                checkpoint_dir='checkpoints'                        # Directorio para guardar los checkpoints.
                ):

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.action_space = [i for i in range(n_actions)]   # Representación en int de las acciones posibles. Es para la selección de épsilon greedy.
        
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.n_actions = n_actions

        self.batch_size = batch_size
        self.mem_size = mem_size
        self.mem_cntr = 0   # Para saber cuándo la memoria está llena y cuándo se puede empezar a entrenar el modelo con un tamaño de batch adecuado.
        self.memory = deque(maxlen=self.mem_size)
        
        '''Ahora vamos a inicializar las redes neuronales y la memoria'''
        self.q_eval = DeepQNetwork(self.input_dim, self.n_actions, self.learning_rate)
        self.q_target = DeepQNetwork(self.input_dim, self.n_actions, self.learning_rate)
        self.target_update_freq = target_update_freq
        self.update_target_network()
        
        # Chekpoint
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def update_target_network(self):
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def store_transition(self, state, action, reward, state_, done):
        self.memory.append((state, action, reward, state_, done))
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor([observation], dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        return action
    
    def get_action(self, state):
        state_matrix = state.state_matrix().flatten()
        action_chosen = self.choose_action(state_matrix)
        action_conv = {0:(0, 1), 1:(0, -1), 2:(1, 0), 3:(-1, 0)}
        return action_conv[action_chosen]
    
    def get_action_mem(self, action):
        action_conv = {(0, 1):0, (0, -1):1, (1, 0):2, (-1, 0):3}
        return action_conv[action]
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            
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

        if self.mem_cntr % self.target_update_freq == 0:
            self.update_target_network()
    
    def save_checkpoint(self, episode, mean_score):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'dqn_agent_ep_{episode}_score_{mean_score:.2f}.pth')
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