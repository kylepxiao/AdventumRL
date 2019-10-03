import abc 
from abc import ABC, abstractmethod 

#Abstract agent class for the grammar api
class Agent(ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    
    