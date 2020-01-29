import numpy as np
import random as rand
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import defaultdict
from math import floor
#from collections import deque

def FF_weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(-0.3, 0.3)
        m.bias.data.fill_(0)

def CNN_weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(-0.001, 0.001)
        m.bias.data.fill_(0)

class NeuralNet(nn.Module):
    def __init__(self, input_size=31, hidden_size=20, num_classes=2):
        super(NeuralNet, self).__init__()
        self.fc0 = nn.Linear(input_size, 256)
        self.activate0 = nn.LeakyReLU()
        self.fc1 = nn.Linear(256, 64)
        self.activate1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=0.05)
        self.fc2 = nn.Linear(64, num_classes)
        self.activate2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p=0.05)
        self.fc3 = nn.Linear(23, 15)
        self.activate3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(p=0.05)
        self.fc4 = nn.Linear(15, num_classes)

    def forward(self, x):
        out = self.fc0(x)
        out = self.activate0(out)
        out = self.fc1(out)
        out = self.activate1(out)
        #out = self.dropout1(out)
        out = self.fc2(out)
        #out = self.activate2(out)
        #out = self.dropout2(out)
        #out = self.fc3(out)
        #out = self.activate3(out)
        #out = self.dropout3(out)
        #out = self.fc4(out)
        return out

class CNN(nn.Module):
    def __init__(self, input_size=31, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 12, stride=2)
        self.pool1 = nn.MaxPool2d(8, 8)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(1152 + input_size, 384)
        self.activate1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=0.05)
        self.fc2 = nn.Linear(384, 64)
        self.activate2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p=0.05)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x, y):
        # Convolution
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool2(F.relu(self.conv2(out)))
        out = torch.flatten(out, 1)
        out = torch.cat((out, y), 1)
        # Feedforward
        out = self.fc1(out)
        out = self.activate1(out)
        out = self.fc2(out)
        out = self.activate2(out)
        out = self.fc3(out)
        return out

class DeepQLearner(object):

    def __init__(self, \
        input_size = 4, \
        num_actions = 2, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.2, \
        radr = 1, \
        dyna = 10, \
        learning_rate = 0.02, \
        batch_size = 32, \
        clip = 1,  \
        load_path = None, \
        save_path = None, \
        camera = False, \
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
        self.max_samples = 512
        self.clip = clip
        self.samples = []
        self.batch_size = batch_size
        self.camera = camera
        self.gamma_dropout = 0
        #self.state_actions = defaultdict(int)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not camera:
            self.model = NeuralNet(input_size = input_size, num_classes=num_actions).to(self.device)
            self.model.apply(FF_weights_init_uniform)
        else:
            self.model = CNN(input_size = input_size, num_classes=num_actions).to(self.device)
            self.model.apply(CNN_weights_init_uniform)
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
            if self.camera:
                output = self.model(torch.Tensor([s[0]]).to(self.device), torch.Tensor([s[1]]).to(self.device))
            else:
                output = self.model(torch.Tensor([s]).to(self.device))
            output_Q, output_action = torch.max(output.data, 1)
            action = output_action[0].item()
        if self.rar < rand.random():
            return action
        else:
            if self.verbose: print("RANDOM ACTION")
            return rand.randint(0, self.num_actions - 1)
            #least_common = min([(self.state_actions[(tuple(self.s), a)], self.s, a) for a in range(self.num_actions)])
            #return least_common[2]

    def query_dyna(self,s_prime,r):
        """
        @summary: Does NOT update the Q table and returns an action
        @param s_prime: The new state
        @param r: The reward
        @returns: The selected action
        """
        if self.rar < rand.random():
            self.model.eval()
            with torch.no_grad():
                if self.camera or [self.s, self.a, s_prime, r] not in self.samples:
                    self.samples.append([self.s, self.a, s_prime, r])
                #self.state_actions[(tuple(self.s), self.a)] += 1
                while len(self.samples) > self.max_samples:
                    self.samples.pop(0)
                next_output = self.model(torch.Tensor([s_prime]).to(self.device))
                next_output_Q, next_output_action = torch.max(next_output.data, 1)
                self.a = next_output_action[0].item()
        else:
            self.a = rand.randint(0, self.num_actions - 1)
        self.s = s_prime
        return self.a

    def clear_dyna(self):
        self.samples = []
        #self.state_actions = defaultdict(int)

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The reward
        @returns: The selected action
        """
        newSample = [self.s, self.a, s_prime, r]
        if self.camera or newSample not in self.samples:
            self.samples.append(newSample)
            r += 5 if not self.camera else 0
        while len(self.samples) > self.max_samples:
            self.samples.pop(0)
        self.model.eval()
        with torch.no_grad():
            if self.camera:
                next_output = self.model(torch.Tensor([s_prime[0]]).to(self.device), torch.Tensor([s_prime[1]]).to(self.device))
            else:
                next_output = self.model(torch.Tensor([s_prime]).to(self.device))
            if self.verbose: print(next_output)
            next_output_Q, next_output_action = torch.max(next_output.data, 1)
            next_action = next_output_action[0].item()
            if self.gamma_dropout > rand.random():
                expected_reward = torch.Tensor([r])
            else:
                expected_reward = r + self.gamma * next_output_Q[0].item()
            if self.verbose: print(expected_reward)

        self.model.train()
        if self.camera:
            output = self.model(torch.Tensor([self.s[0]]).to(self.device), torch.Tensor([self.s[1]]).to(self.device))
        else:
            output = self.model(torch.Tensor([self.s]).to(self.device))
        output_Q, output_action = torch.max(output.data, 1)
        label = torch.Tensor()
        label.data = output.clone()
        label[0][self.a] = expected_reward
        loss = self.criterion(output, label)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        self.losses.append(loss.item())

        if self.rar < rand.random():
            self.a = next_action
        else:
            if self.verbose: print("RANDOM ACTION")
            self.a = rand.randint(0, self.num_actions - 1)
            #least_common = min([(self.state_actions[(tuple(self.s), a)], self.s, a) for a in range(self.num_actions)])
            #self.a = least_common[2]
            #self.state_actions[(tuple(self.s), self.a)] += 1
        self.rar *= self.radr

        #if newSample not in self.samples:
        #    self.samples.append(newSample)
        #while len(self.samples) > self.max_samples:
        #    self.samples.pop()

        self.s = s_prime
        #if self.verbose: print("s =", s_prime,"a =",self.a,"r =",r)
        return self.a

    def run_dyna(self):
        """
        @summary: Runs dyna on saved samples
        @returns: The loss for the epoch
        """
        if len(self.samples) == 0:
            return

        s_list = []
        a_list = []
        s_prime_list = []
        r_list = []
        if self.camera:
            s_state_list = []
            s_prime_state_list = []
        #rand.shuffle(self.samples)
        curr_samples = rand.sample(self.samples, len(self.samples))
        for sample in curr_samples:
            [s, a, s_prime, r] = sample
            if self.camera:
                s_list.append(s[0])
                a_list.append(a)
                s_prime_list.append(s_prime[0])
                r_list.append(r)
                s_state_list.append(s[1])
                s_prime_state_list.append(s_prime[1])
            else:
                s_list.append(s)
                a_list.append(a)
                s_prime_list.append(s_prime)
                r_list.append(r)
        #sample_set = {}
        #for sample in reversed(self.samples):
        #    sample_set[str(sample)] = sample
        #self.samples = list(sample_set.values())

        self.samples.sort(key = lambda x: x[3])

        permutation = torch.randperm(len(s_prime_list))
        minibatch_losses = []
        s_tensor = torch.Tensor(s_list).to(self.device)
        s_prime_tensor = torch.Tensor(s_prime_list).to(self.device)
        r_tensor = torch.Tensor(r_list).to(self.device)
        if self.camera:
            s_state_tensor =torch.Tensor(s_state_list).to(self.device)
            s_prime_state_tensor =torch.Tensor(s_prime_state_list).to(self.device)
        for i in range(0, len(s_prime_list), self.batch_size):
            indices = permutation[i:i+self.batch_size]
            self.model.eval()
            with torch.no_grad():
                if self.gamma_dropout > rand.random():
                    expected_reward = r_tensor[indices]
                else:
                    if self.camera:
                        next_output = self.model(s_prime_tensor[indices], s_prime_state_tensor[indices])
                    else:
                        next_output = self.model(s_prime_tensor[indices])
                    next_output_Q, next_output_action = torch.max(next_output.data, 1)
                    expected_reward = r_tensor[indices] + self.gamma * next_output_Q

            self.model.train()
            if self.camera:
                output = self.model(s_tensor[indices], s_state_tensor[indices])
            else:
                output = self.model(s_tensor[indices])
            output_Q, output_action = torch.max(output.data, 1)

            label = torch.Tensor()
            label.data = output.clone()
            for i in range(len(label)):
                k = indices[i].item()
                label[i, a_list[k]] = expected_reward[i]

            loss = self.criterion(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            minibatch_losses.append(loss.item())

        avg_loss = sum(minibatch_losses) / len(minibatch_losses)
        self.losses.append(avg_loss)
        return avg_loss

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
