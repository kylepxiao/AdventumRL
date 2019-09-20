import numpy as np
import random as rand
import torch
import torch.nn as nn
import os

class NeuralNet(nn.Module):
    def __init__(self, input_size=31, hidden_size=20, num_classes=2):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 23)
        self.activate1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=0.05)
        self.fc2 = nn.Linear(23, 15)
        self.activate2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p=0.05)
        self.fc3 = nn.Linear(15, num_classes)
        #self.activate3 = nn.Sigmoid()
        #self.dropout3 = nn.Dropout(p=0.05)
        #self.fc4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activate1(out)
        #out = self.dropout1(out)
        out = self.fc2(out)
        out = self.activate2(out)
        #out = self.dropout2(out)
        out = self.fc3(out)
        #out = self.activate3(out)
        #out = self.dropout3(out)
        #out = self.fc4(out)
        return out

class DeepQLearner(object):

    def __init__(self, \
        input_size = 4, \
        num_actions = 2, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.1, \
        radr = 1, \
        dyna = 10, \
        learning_rate = 1, \
        load_path = None, \
        save_path = None, \
        verbose = False):

        self.verbose = verbose
        self.input_size = input_size
        self.num_actions = num_actions
        self.s = [0] * input_size
        self.a = 0

        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.samples = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuralNet(input_size = input_size, num_classes=num_actions).to(device)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Load saved model
        self.save_path = save_path
        if load_path != None and os.path.exists(save_path):
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.eval()

        self.losses = []

    def author(self):
        return 'kxiao36'

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.model.eval()
        with torch.no_grad():
            self.s = s
            output = self.model(torch.Tensor([s]))
            output_Q, output_action = torch.max(output.data, 1)
            action = output_action[0].item()
            return action

    def query_dyna(self,s_prime,r):
        """
        @summary: Does NOT update the Q table and returns an action
        @param s_prime: The new state
        @param r: The reward
        @returns: The selected action
        """
        if self.rar > rand.random():
            self.model.eval()
            with torch.no_grad():
                self.samples.append([self.s, self.a, s_prime, r])
                next_output = self.model(torch.Tensor([s_prime]))
                next_output_Q, next_output_action = torch.max(next_output.data, 1)
                self.a = next_output_action[0].item()
        else:
            self.a = rand.randint(0, self.num_actions - 1)
        self.s = s_prime
        return self.a

    def clear_dyna(self):
        self.samples = []

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The reward
        @returns: The selected action
        """
        self.model.eval()
        with torch.no_grad():
            self.samples.append([self.s, self.a, s_prime, r])
            next_output = self.model(torch.Tensor([s_prime]))
            print(next_output)
            next_output_Q, next_output_action = torch.max(next_output.data, 1)
            next_action = next_output_action[0].item()
            expected_reward = r + self.gamma * next_output_Q[0].item()

        self.model.train()
        output = self.model(torch.Tensor([self.s]))
        output_Q, output_action = torch.max(output.data, 1)
        label = torch.Tensor()
        label.data = output.clone()
        label[0][self.a] = expected_reward
        loss = self.criterion(output, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

        self.s = s_prime
        self.a = next_action if self.rar > rand.random() else rand.randint(0, self.num_actions - 1)
        self.rar *= self.radr

        if self.verbose: print("s =", s_prime,"a =",self.a,"r =",r)
        return self.a

    def run_dyna(self):
        """
        @summary: Runs dyna on saved samples
        @returns: The loss for the epoch
        """
        s_list = []
        a_list = []
        s_prime_list = []
        r_list = []
        rand.shuffle(self.samples)
        for sample in self.samples:
            [s, a, s_prime, r] = sample
            s_list.append(s)
            a_list.append(a)
            s_prime_list.append(s_prime)
            r_list.append(r)

        self.model.eval()
        with torch.no_grad():
            next_output = self.model(torch.Tensor(s_prime_list))
            next_output_Q, next_output_action = torch.max(next_output.data, 1)
            expected_reward = torch.Tensor(r_list) + self.gamma * next_output_Q

        self.model.train()
        output = self.model(torch.Tensor(s_list))
        output_Q, output_action = torch.max(output.data, 1)
        label = torch.Tensor()
        label.data = output.clone()
        for i in range(len(label)):
            label[i, a_list[i]] = expected_reward[i]
        loss = self.criterion(output, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        return loss.item()

    def save(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.save_path)

    def plot_loss(self):
        import matplotlib.pyplot as plt
        prices_length = 10
        ravgs = [sum(self.losses[i:i+prices_length])/prices_length for i in range(len(self.losses)-prices_length+1)]
        plt.plot(ravgs)
        plt.show()
