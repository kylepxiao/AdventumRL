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
from models.CameraDQNAgent import CameraDQNAgent
import MalmoPython
import json
import logging
import os
import random
import sys
import time
import argparse

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import tkinter as tk
    import functools
    print = functools.partial(print, flush=True)

'''
@summary: Parses the grammar logic and stores grammar information for a mission
'''
class GrammarLogic:
    # Rules (https://textworld.readthedocs.io/en/latest/textworld.logic.html) also work/Predicates
    def __init__(self, file=None):
        self.parser = GrammarParser(file=file)
        self.variables = self.parser.getVariables()
        self.defaultFacts = self.parser.getDefaultFacts()
        self.logicalActions = self.parser.getActions()
        self.triggers = self.parser.getTriggers()
        self.goal = self.parser.getGoal()
        self.rules = []
        self.predicates = []

    # #Intermediate methods that will eventually be replaced by a parser (maybe integrated into agents/its own file?)
    # def addRule(self, name, precondition, postcondition):
    #     self.rules.append(Rule(name, precondition, postcondition)) #Given a scenario (5 keys, 5 doors), create a rule to allow door unlocking and then create actions for all pairs it applies to
    #     self.logicalActions.append(LogicalAction(name, precondition, postcondition, None, None)) #Pretty sure this is wrong, but we want to add an action that corresponds to the added rule
    # def addAction(self, name, precondition, postcondition, command, reward):
    #     self.logicalActions.append(LogicalAction(name, precondition, postcondition, command, reward))
    # def addPredicate(self, name, parameters):
    #     self.predicates.append(Predicate(name, parameters)) #If class A is related to class B, return true (Any Key next to Any Door)

'''
@summary: Holds all information for the currently executing mission, calls everything else
'''
class GrammarMission:
    def __init__(self, mission_file='./grammar_demo.xml', quest_file='./quest_entities.xml', grammar_file="./quest_grammar.json", agent=None, log="off"):
        self.mission_file = mission_file
        self.quest_file = quest_file
        self.grammar_file = grammar_file
        self.grammar_logic = GrammarLogic(grammar_file)
        self.agent = agent
        self.log = False if log is 'off' else True

    '''
    @summary: Parses the quest and mission files to create the initial world state
    @returns: The initial world state
    '''
    def getInitialWorldState(self):
        state_list = set(self.grammar_logic.defaultFacts)
        # define globa  l objects and propositions
        entities = {}
        objectVars = {}
        objectProps = {}
        props = set()
        boundaries = set()

        if self.quest_file is not None:
            # parse XML
            questTree = ET.parse(self.quest_file)
            objectTags = [("Grid", "boundary")]
            ns = namespace(questTree.getroot())
            objectTargets = dict((ns + tag[0], tag[1]) for tag in objectTags)

            for item in questTree.iter():
                if item.tag in objectTargets.keys():
                    item_class = objectTargets[item.tag]
                    currentVar = Variable(item.attrib['name'], item_class)
                    if item_class == "boundary":
                        for child in item.iter():
                            if child.tag[len(ns):] == "min":
                                pmin = (child.attrib['x'], child.attrib['y'], child.attrib['z'])
                            elif child.tag[len(ns):] == "max":
                                pmax = (child.attrib['x'], child.attrib['y'], child.attrib['z'])
                        currentVar = Boundary(item.attrib['name'], pmin, pmax)
                        entities[item.attrib['name']] = currentVar
                        boundaries.add(item.attrib['name'])
                        #props.add(Proposition("in", [currentVar, worldVar]))

        if self.mission_file is not None:
            # parse XML
            missionTree = ET.parse(self.mission_file)
            objectTags = [("DrawBlock", "block"), ("DrawItem", "item")]
            ns = namespace(missionTree.getroot())
            objectTargets = dict((ns + tag[0], tag[1]) for tag in objectTags)

            for item in missionTree.iter():
                if item.tag in objectTargets.keys():
                    item_class = objectTargets[item.tag]
                    currentVar = Variable(item.attrib['type'], item_class)
                    if item_class == "item":
                        currentVar = Item(item.attrib['type'], item.attrib['x'], item.attrib['y'], item.attrib['z'])
                        entities[item.attrib['type']] = currentVar
                        for boundary in boundaries:
                            if entities[boundary].contains(currentVar):
                                props.add(Proposition("in", [currentVar, entities[boundary]]))
                elif item.tag == ns + "AgentStart":
                    for child in item.iter():
                        if child.tag[len(ns):] == "Placement":
                            currentVar = Actor("player", child.attrib['x'], child.attrib['y'], child.attrib['z'])
                            entities['player'] = currentVar

            state_list = state_list.union( props )
        return MalmoLogicState(facts=state_list,
            actions=self.grammar_logic.logicalActions,
            triggers=self.grammar_logic.triggers,
            entities=entities,
            boundaries=boundaries,
            goal=self.grammar_logic.goal)


    '''
    @summary: Gets the current mission being run
    @returns: The current mission file
    '''
    def getMission(self):
        return self.mission_file


    '''
    @summary: Sets the current mission to a different one
    @param mission: The new mission to be run
    '''
    def setMission(self, mission):
        self.mission_file = mission


    '''
    @summary: Gets the current quest being run
    @returns: The current quest file
    '''
    def getQuest(self):
        return self.quest_file


    '''
    @summary: Sets the current mission to a different one
    @param quest: The new quest to be followed
    '''
    def setQuest(self, quest):
        self.quest_file = quest

    '''
    @summary: Sets the current agent to a different one
    @param agent: The new agent to run the mission with
    '''
    def setAgent(self, agent: Agent):
        self.agent = agent(
            LogicalAgentHost(
                initialState = self.getInitialWorldState(),
                actions = self.grammar_logic.logicalActions,
                goal = self.grammar_logic.goal,
                triggers = self.grammar_logic.triggers
            )
        )

    '''
    @summary: Gets the current reward from the agent
    @returns: Current rewards
    '''
    def getRewards(self):
        return self.agent.getWorldState().rewards

    '''
    @summary: Gets the current world state
    @returns: Current world state
    '''
    def getWorldState(self):
        return self.agent.getWorldState()

    '''
    @summary: Gets the current observations about the world
    @returns: World state observations
    '''
    def getWorldStateObservations(self):
        return json.loads(self.agent.getWorldState().observations[-1].text)

    '''
    @summary: Send an action to the agent
    @param action: An action for the agent to execute
    '''
    def sendCommand(self, action):
        self.agent.sendCommand(action)

    '''
    @summary: Change the current grammar logic
    @param grammar: New grammar logic for the mission
    '''
    def setGrammar(self, grammar):
        self.grammar_logic = grammar

    '''
    @summary: Gets the current grammar logic
    @returns: Current grammar logic
    '''
    def getGrammar(self):
        return self.grammar_logic


    '''
    @summary: Runs the current mission
    '''
    def run_mission(self, num_repeats=100000): # Running the mission (taken from grammar_demo.py)
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

        checkpoint_iter = 100

        #if self.agent.host.receivedArgument("test"):
        #    num_repeats = 1
        #else:
        #    num_repeats = 150

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
            # cumulative_rewards += [ cumulative_reward ]

            if self.log and (i % checkpoint_iter == 0):
                self.agent.logOutput()

            # -- clean up -- #
            time.sleep(0.5) # (let the Mod reset)

        print("Done.")

        print()
        #print("Cumulative rewards for all %d runs:" % num_repeats)
        #print(cumulative_rewards)
        return

parser = argparse.ArgumentParser(description='Run missions in Malmo')
parser.add_argument("--mission_file", help='choose which mission file to run', default='./grammar_demo.xml')
parser.add_argument("--quest_file", help='choose file to specify quest entities', default='./quest_entities.xml')
parser.add_argument("--grammar_file", help='choose file to specify logical grammar', default="./quest_grammar.json")
parser.add_argument("--agent", help='choose which agent to run (TabQAgent, DQNAgent)', default="TabQAgent")
parser.add_argument("--log", help='whether to record logs for a mission', default="off")
args = parser.parse_args()

if __name__ == "__main__":
    mission = GrammarMission(mission_file=args.mission_file, quest_file=args.quest_file, grammar_file=args.grammar_file, log=args.log)
    if args.agent == 'TabQAgent':
        mission.setAgent(TabQAgent)
    elif args.agent == 'DQNAgent':
        mission.setAgent(DQNAgent)
    elif args.agent == 'CameraDQNAgent':
        mission.setAgent(CameraDQNAgent)
    else:
        print("unrecognized agent")
    mission.run_mission()
