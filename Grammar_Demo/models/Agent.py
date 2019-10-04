import abc 
from abc import ABC, abstractmethod 

#Abstract agent class for the grammar API
class Agent(ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass
    
    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def act(self):
        pass

    @abc.abstractmethod
    def getActionSpace(self):
        pass

    @abc.abstractmethod
    def getObservations(self, world_state):
        pass
    