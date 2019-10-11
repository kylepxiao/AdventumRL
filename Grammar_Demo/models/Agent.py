import abc 
from abc import ABC, abstractmethod 
from builtins import object

#Abstract agent class for the grammar API
class Agent(ABC):

    @abc.abstractmethod
    def __init__(self):
        """
        @summary: Initialize the agent
        """
        pass

    @abc.abstractmethod
    def run(self):
        """
        @summary: Run the agent on the world
        """
        pass
    
    @abc.abstractmethod
    def train(self):
        """
        @summary: Train the agent on the observations
        """
        pass

    @abc.abstractmethod
    def act(self, world_state, current_r):
        """
        @summary: Have the agent act on the current world state and reward
        @param world_state: The current state the agent will act upon
        @param current_r: The current reward of the agent
        @returns: The updated reward of the agent
        """
        pass

    @abc.abstractmethod
    def getActionSpace(self): #Get LogicalActions and 'regular' actions
        """
        @summary: Get all possible actions for the agent
        @returns: Possible agent actions and LogicalActions
        """
        pass

    @abc.abstractmethod
    def getObservations(self):
        """
        @summary: Get the agent's current observations
        @returns: Current observations
        """
        pass

    @abc.abstractmethod
    def addObservations(self, observation):
        """
        @summary: Add observations to the agent
        @param observation: A quadruple of state, action, newState, and reward for the agent to remember
        """
        pass 

    @abc.abstractmethod
    def queryActions(self, world_state, current_r):
        """
        @summary: Place the agent in a new state and recieve the successive action it takes
        @param world_state: The current state the agent will act upon
        @param current_r: The current reward of the agent
        @returns: The action taken by the agent and the new reward
        """
        pass

    @abc.abstractmethod
    def setState(self, world_state): #Put the agent in a new state (use another quadruple as above), but dont get a new action from it 
        """
        @summary: Place the agent in a new state
        @param world_state: The current state the agent will act upon
        """
        pass            