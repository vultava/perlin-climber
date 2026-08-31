import torch                        # PyTorch core framework
import torch.nn as nn               # Neural network components
import torch.nn.functional as F     # Functions in torch
import torch.optim as optim
import random as rd
import numpy as np
from collections import deque

class QNetwork(nn.Module):          # Ihneretuje osnovni pytorch modul pomocu kojeg pytorch moze da interektuje sa nn

    def __init__(self, state_size: int = 6, action_size: int = 4):
        # state size: [relativ_x, relativ_y, h0, h1, h2 ,h3] -> 6
        # action size: [up, right, down, left] -> 4

        super(QNetwork, self).__init__()    # Pokrece pythorch kod iz nn.Module

        # Prvi fully connected sloj od state do internal nodes
        self.fc1 = nn.Linear(state_size, 64)
        # Drugi fully connected sloj od internal do internal nodes
        self.fc2 = nn.Linear(64, 64)
        # Treci fully connected sloj od internal do action, izlaz su Q-values
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Funkcija koja definise kako podaci teku kroz nn

        x = F.relu(self.fc1(state))     # Rectified Linear Unit activation function
        x = F.relu(self.fc2(x))

        return self.fc3(x)

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


class ReplayBuffer:

    # Funkcija koje formira baffer koji nakon dostizanja kapacite izbacuje najstariji clan, FIFO
    def __init__(self, cap: int = 10000):
        self.buffer = deque(maxlen=cap)

    # Na kraju baffera doda tuple
    def push(self, state, action, reward, next_state, done):
        # state -> ono sto agent vidi, get_state()
        # action -> smer kretanja koji ce odabrati nn
        # reward -> nagarada nakon koraka, iz step()
        # nest_state -> stanje kada se pomeri, iz step()
        # done -> da li je gotovo, iz step()
        self.buffer.append((state, action, reward, next_state, done))

    # Proizvoljan odabir prethodnih koraka iz buffera
    def sample(self, batch_size: int):

        # Bira random koraka
        batch = rd.sample(self.buffer, batch_size)

        # Pretvara iz liste tuplova u posebne liste za svaki deo
        states, actions, rewards, next_states, dones = zip(*batch)

        # Pretvaranje u tensore
        states_tens = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tens = torch.tensor(actions, dtype=torch.long)
        rewards_tens = torch.tensor(rewards, dtype=torch.float32)
        next_states_tens = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_tens = torch.tensor(dones, dtype=torch.float32)

        return states_tens, actions_tens, rewards_tens, next_states_tens, dones_tens

    def __len__(self) -> int:
        return len(self.buffer)

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


class DQNAgent:

    def __init__(self, state_size: int = 6, action_size: int = 4, lr: float = 1e-3, gamma: float = 0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # Glavni nn -> pravi nn koji se updejtuje pri svakom koraku
        self.policy_nn = QNetwork(state_size, action_size)

        # Target nn -> radi stabilizacija targeta za Q-vrednosti
        self.target_nn = QNetwork(state_size, action_size)
        self.target_nn.load_state_dict(self.policy_nn.state_dict())
        self.target_nn.eval()

        # Optimizator i funkcija gubitka (loss function)
        self.optimizer = optim.Adam(self.policy_nn.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Memory
        self.memory = ReplayBuffer()

    def select_action(self, state: np.ndarray, epsilon: float) -> int:

        # IZVODI NASUMICNE POKRETE, istrazuje
        if rd.random() < epsilon:
            return rd.randint(0, self.action_size-1)

        # TRENUTNO STANJE ULAZI U NN I POKRET JE ODREDJEN NA OSNOVU IZLAZA NN, najveci q-value
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_nn(state_tensor)
        return int(torch.argmax(q_values).item())

    def learn(self, batch_size: int = 64):

        # Saceka da se memory napuni
        if len(self.memory) < batch_size:
            return

        # Vadi ranodm iz memorije i unpekuje
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Testira q-vrednosti kroz glavni nn sa trenutnim stanjem
        curr_q_values = self.policy_nn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Testira q_vrednosti kroz target nn sa sledecim stanjem
        with torch.no_grad():
            max_next_q_values = self.target_nn(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Racuna razliku
        loss = self.criterion(curr_q_values, target_q_values)

        # Vrsi optimizaciju
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_nn(self):
        self.target_nn.load_state_dict(self.policy_nn.state_dict())
