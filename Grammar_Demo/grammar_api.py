from __future__ import print_function
from future import standard_library
standard_library.install_aliases()

from builtins import range
from builtins import object
from textworld.logic import Action, Rule, Placeholder, Predicate, Proposition, Signature, State, Variable
from MalmoLogicState import *
from constants import *
from models import *
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
else:
    import tkinter as tk

class grammar_logic:
    #Rules (https://textworld.readthedocs.io/en/latest/textworld.logic.html) also work/Predicates
    def __init__(self, logicalActions=None, triggers=None, goals = None):
        grabPrecondition = [Proposition("notreached", [boundary1Var]), Proposition("in", [playerVar, boundary1Var])]
        grabPostcondition = [Proposition("reached", [boundary1Var]), Proposition("in", [playerVar, boundary1Var])]
        grabAction = LogicalAction("grab", grabPrecondition, grabPostcondition, "reward 50")

        lockPrecondition = [Proposition("in", [playerVar, boundary2Var]), Proposition("in", [itemVars["diamond"], inventoryVar]), Proposition("locked", [doorVar])]
        lockPostcondition = [Proposition("unlocked", [doorVar])]
        unlockAction = LogicalAction("unlock", lockPrecondition, lockPostcondition, "discardCurrentItem", 75)

        goalPrecondition = [Proposition("unlocked", [doorVar]), Proposition("in", [playerVar, boundary3Var])]
        goalPostcondition = [Proposition("in", [playerVar, boundary3Var])]
        goalAction = LogicalAction("goal", goalPrecondition, goalPostcondition, "quit", 200)
        goal = [(Proposition("in", [itemVars['diamond'], boundary1Var]), True), (Proposition("in", [itemVars['diamond'], inventoryVar]), True)]
        
        self.logicalActions = logicalActions or [grabAction, unlockAction, goalAction]
        self.triggers = triggers or [Proposition("notreached", [boundary1Var]), Proposition("locked", [doorVar])]
        self.goals = goals or goal

class grammar_mission:
    # Also add support to allow users to run their own mission through here
    def __init__(self, mission_file=None, quest_file=None, agent=None, grammar_logic=None):
        if sys.version_info[0] == 2:
            sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
        else:
            import functools
            print = functools.partial(print, flush=True)
        
        self.mission_file = mission_file or './grammar_demo.xml'
        self.quest_file = quest_file or './quest_entities.xml'
        self.grammar_logic = grammar_logic or grammar_logic()
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
        try:
            self.agent.host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:',e)
            print(self.agent.host.getUsage())
            exit(1)
        if self.agent.host.receivedArgument("help"):
            print(self.agent.host.getUsage())
            exit(0)

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

# parser = argparse.ArgumentParser(description='Run missions in Malmo')
# parser.add_argument("mission", help='choose which mission to run')
# args = parser.parse_args()

# if (args.mission == 'grammar_demo'):
#     mission = grammar_mission
#     mission.run_mission(mission)