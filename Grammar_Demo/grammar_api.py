from __future__ import print_function
from future import standard_library
standard_library.install_aliases()

from builtins import range
from builtins import object
from textworld.logic import Action, Rule, Placeholder, Predicate, Proposition, Signature, State, Variable
from MalmoLogicState import *
from constants import *
from parse_grammar import GrammarParser
from models.Agent import Agent
from models.TabQAgent import TabQAgent
from models.DQNAgent import DQNAgent
import MalmoPython
import json
import logging
import os
import random
import sys
import time
import argparse


"""if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)"""

class grammar_logic:
    #Rules (https://textworld.readthedocs.io/en/latest/textworld.logic.html) also work/Predicates
    def __init__(self, file=None):
        self.parser = GrammarParser(file=file)
        self.logicalActions = self.parser.getActions()
        self.triggers = self.parser.getTriggers()
        self.goals = self.parser.getGoal()
        self.rules = []
        self.predicates = []

    #Intermediate methods that will eventually be replaced by a parser (maybe integrated into agents/its own file?)
    def addRule(self, name, precondition, postcondition):
        self.rules.append(Rule(name, precondition, postcondition)) #Given a scenario (5 keys, 5 doors), create a rule to allow door unlocking and then create actions for all pairs it applies to
        self.logicalActions.append(LogicalAction(name, precondition, postcondition, None, None)) #Pretty sure this is wrong, but we want to add an action that corresponds to the added rule
    def addAction(self, name, precondition, postcondition, command, reward):
        self.logicalActions.append(LogicalAction(name, precondition, postcondition, command, reward))
    def addPredicate(self, name, parameters):
        self.predicates.append(Predicate(name, parameters)) #If class A is related to class B, return true (Any Key next to Any Door)


class grammar_mission:
    # Also add support to allow users to run their own mission through here
    def __init__(self, mission_file=None, quest_file=None, agent=None, grammar_file=None):
        self.mission_file = mission_file or './grammar_demo.xml'
        self.quest_file = quest_file or './quest_entities.xml'
        self.grammar_logic = grammar_logic(grammar_file) if grammar_file is not None else grammar_logic(file="quest_grammar.json")
        self.agent = agent
    def getMission(self):
        return self.mission_file
    def setMission(self, mission):
        self.mission_file = mission
    def getQuest(self):
        return self.quest_file
    def setQuest(self, quest):
        self.quest_file = quest
    def setAgent(self, agent):
        self.agent = agent

    def getRewards(self):
        return self.agent.getWorldState().rewards
    def getWorldState(self):
        return self.agent.getWorldState()
    def getWorldStateObservations(self):
        return json.loads(self.agent.getWorldState().observations[-1].text)
        # You'll need to extract the observations from the JSON yourself, includes entity
        # I'm unsure if -1 is the best way to index, I'm deferring to the official malmo example (https://github.com/microsoft/malmo/pull/192/files)
    def sendCommand(self, action):
        self.agent.sendCommand(action)

    def setGrammar(self, newGrammar):
        self.grammar_logic = newGrammar
    def getGrammar(self):
        return self.grammar_logic
    def addGrammarLogicalAction(self, action):
        self.grammar_logic.logicalActions.append(action)
    def addGrammarTrigger(self, trigger):
        self.grammar_logic.triggers.add(trigger)

    def run_mission(self): # Running the mission (taken from grammar_demo.py)
        # -- set up the mission -- #
        with open(self.mission_file, 'r') as f:
            print("Loading mission from %s" % self.mission_file)
            mission_xml = f.read()
            my_mission = MalmoPython.MissionSpec(mission_xml, True)
        # add 20% holes for interest
        """for x in range(1,4):
            for z in range(1,13):
                if random.random()<0.1:
                    my_mission.drawBlock( x,45,z,"lava")"""

        max_retries = 3

        if self.agent.host.receivedArgument("test"):
            num_repeats = 1
        else:
            num_repeats = 150

        cumulative_rewards = []
        for i in range(num_repeats):
            print()
            print('Repeat %d of %d' % ( i+1, num_repeats ))

            my_mission_record = MalmoPython.MissionRecordSpec()

            for retry in range(max_retries):
                try:
                    self.agent.host.startMission( my_mission, my_mission_record )
                    break
                except RuntimeError as e:
                    if retry == max_retries - 1:
                        print("Error starting mission:",e)
                        exit(1)
                    else:
                        time.sleep(2.5)

            print("Waiting for the mission to start", end=' ')
            world_state = self.agent.host.getWorldState()
            while not world_state.has_mission_begun:
                print(".", end="")
                time.sleep(0.1)
                world_state = self.agent.host.getWorldState()
                for error in world_state.errors:
                    print("Error:",error.text)
            print()

            # -- run the agent in the world -- #
            cumulative_reward = self.agent.run()
            print('Cumulative reward: %d' % cumulative_reward)
            cumulative_rewards += [ cumulative_reward ]

            # -- clean up -- #
            time.sleep(0.5) # (let the Mod reset)

        print("Done.")

        print()
        print("Cumulative rewards for all %d runs:" % num_repeats)
        print(cumulative_rewards)
        return

#if __name__ == "__main__":
parser = argparse.ArgumentParser(description='Run missions in Malmo')
parser.add_argument("--mission_file", help='choose which mission file to run', default='./grammar_demo.xml')
parser.add_argument("--quest_file", help='choose file to specify quest entities', default='./quest_entities.xml')
parser.add_argument("--grammar_file", help='choose file to specify logical grammar', default="./quest_grammar.json")
parser.add_argument("--agent", help='choose which agent to run (TabQAgent, DQNAgent)', default="TabQAgent")
args = parser.parse_args()

if __name__ == "__main__":
    mission = grammar_mission(mission_file=args.mission_file, quest_file=args.quest_file, grammar_file=args.grammar_file)
    if args.agent == 'TabQAgent':
        mission.setAgent(TabQAgent(mission.getGrammar(), mission.getMission(), mission.getQuest()))
    elif args.agent == 'DQNAgent':
        mission.setAgent(DQNAgent(mission.getGrammar(), mission.getMission(), mission.getQuest()))
    else:
        print("unrecognized agent")
    mission.run_mission()
