from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Grammar Demo

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998

from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
from textworld.logic import Action, Rule, Placeholder, Predicate, Proposition, Signature, State, Variable
from MalmoLogicState import *
from constants import *
import MalmoPython
import json
import logging
import os
import random
import sys
import time
if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk

#rewardPrecondition = [Proposition("in", [playerVar, boundary1Var])]
#rewardPostcondition = [Proposition("in", [playerVar, boundary1Var])]
#rewardRule = LogicalAction("reward", rewardPrecondition, rewardPostcondition, "win")

#unlock = Rule.parse("unlock :: $in(item, boundary) & locked(unlockable) -> in(agent, boundary) & unlocked(unlockable)")
#room1_key = {Placeholder.parse('item') : itemVars['diamond'], Placeholder.parse('boundary') : boundary2Var}

#unlock = Rule.parse("unlock :: $in(agent, world) & locked(unlockable) -> in(agent, world)")

grabPrecondition = [Proposition("notreached", [boundary1Var]), Proposition("in", [playerVar, boundary1Var])]
grabPostcondition = [Proposition("reached", [boundary1Var]), Proposition("in", [playerVar, boundary1Var])]
grabAction = LogicalAction("grab", grabPrecondition, grabPostcondition, "reward 50")


lockPrecondition = [Proposition("in", [playerVar, boundary2Var]), Proposition("in", [itemVars["diamond"], inventoryVar]), Proposition("locked", [doorVar])]
lockPostcondition = [Proposition("unlocked", [doorVar])]
unlockAction = LogicalAction("unlock", lockPrecondition, lockPostcondition, "discardCurrentItem")

#goal = Rule.parse("goal :: $in(agent, boundary) & unlocked(unlockable) -> in(agent, boundary) & locked(unlockable)")
#goal_map = {Placeholder.parse('agent') : playerVar, Placeholder.parse('boundary') : boundary3Var}

goalPrecondition = [Proposition("unlocked", [doorVar]), Proposition("in", [playerVar, boundary3Var])]
goalPostcondition = [Proposition("in", [playerVar, boundary3Var])]
goalAction = LogicalAction("goal", goalPrecondition, goalPostcondition, "win")

#asdfPrecondition = [Proposition("in", [itemVars['diamond'], worldVar])]
#asdfPostcondition = []
#asdfAction = LogicalAction("asdf", asdfPrecondition, asdfPostcondition, "asdf")

logicalActions = [grabAction, unlockAction, goalAction]

triggers = [Proposition("notreached", [boundary1Var]), Proposition("locked", [doorVar])]

class TabQAgent(object):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self, mission_file=None, quest_file=None):
        self.epsilon = 0.1 # chance of taking a random action instead of the best

        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.move_actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
        self.q_table = {}
        self.logical_q_table = {}
        self.canvas = None
        self.root = None
        goal = [(Proposition("in", [itemVars['diamond'], boundary1Var]), True),
            (Proposition("in", [itemVars['diamond'], inventoryVar]), True)]
        self.host = LogicalAgentHost(mission_file, quest_file, logicalActions, goal, triggers)
        self.alpha = 0.5
        self.gamma = 0.9

    def updateQTable( self, reward, current_state ):
        """Change q_table to reflect what we have learnt."""

        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]

        # TODO: what should the new action value be?
        maxqprime = max(self.q_table[current_state])
        new_q = old_q + self.alpha * (reward + self.gamma * maxqprime - old_q)

        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q

    def updateQTableFromTerminatingState( self, reward ):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""

        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]

        # TODO: what should the new action value be?
        new_q = old_q + self.alpha * (reward - old_q)

        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q

    def act(self, world_state, current_r ):
        """take 1 action in response to the current world state"""

        self.host.updateLogicState(world_state)
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        print(self.host.state.getStateKey())
        #print(self.host.getLogicalActions())
        #print([action for action in self.host.state.all_applicable_actions([unlock], room1_key)])

        #current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        current_s = self.host.state.getStateKey()
        logicalActions = self.host.state.getApplicableActions()
        actions = self.move_actions + logicalActions
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
        if current_s not in self.q_table:
            self.q_table[current_s] = ([0] * len(actions))

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable( current_r, current_s )

        self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) )

        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(actions) - 1)
            self.logger.info("Random action: %s" % actions[a])
        else:
            m = max(self.q_table[current_s])
            self.logger.debug("Current values: %s" % ",".join(str(x) for x in self.q_table[current_s]))
            l = list()
            for x in range(0, len(actions)):
                if self.q_table[current_s][x] == m:
                    l.append(x)
            y = random.randint(0, len(l)-1)
            a = l[y]
            self.logger.info("Taking q action: %s" % actions[a])

        # try to send the selected action, only update prev_s if this succeeds
        try:
            self.host.sendCommand(actions[a], is_logical = a >= len(self.move_actions))
            self.prev_s = current_s
            self.prev_a = a
            """if self.host.checkGoal():
                self.host.sendCommand("discardCurrentItem")
                print("\n\n-----------BOUNDING BOX DETECTED-----------")
                print(self.host.state.goal)
                print("\n")
            else:
                self.host.sendCommand(actions[a])
                self.prev_s = current_s
                self.prev_a = a"""

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

        return current_r

    def run(self):
        """run the agent on the world"""

        total_reward = 0

        self.prev_s = None
        self.prev_a = None

        is_first_action = True

        # main loop:
        world_state = self.host.getWorldState()
        while world_state.is_mission_running:

            current_r = 0

            if is_first_action:
                self.host.resetState()
                # wait until have received a valid observation
                while True:
                    time.sleep(0.1)
                    world_state = self.host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    current_r += self.host.rewardValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(0.1)
                    world_state = self.host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    current_r += self.host.rewardValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = self.host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    current_r += self.host.rewardValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, current_r)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState( current_r )

        self.drawQ()

        return total_reward

    def drawQ( self, curr_x=None, curr_y=None ):
        scale = 40
        world_x = 6
        world_y = 14
        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
            self.canvas.grid()
            self.root.update()
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        curr_radius = 0.2
        action_positions = [ ( 0.5, action_inset ), ( 0.5, 1-action_inset ), ( action_inset, 0.5 ), ( 1-action_inset, 0.5 ) ]
        # (NSWE to match action order)
        min_value = -20
        max_value = 20
        suffixes = ["000:0", "000:1"]
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d|" % (x,y)
                self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")
                for action in range(4):
                    #if not s in self.q_table:
                    #    continue
                    values = []
                    for suf in suffixes:
                        if s + suf in self.q_table:
                            values.append(self.q_table[s + suf][action])
                    if len(values) == 0:
                        continue
                    value = float(sum(values)) / len(values)
                    color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
                    color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                    color_string = '#%02x%02x%02x' % (255-color, color, 0)
                    self.canvas.create_oval( (x + action_positions[action][0] - action_radius ) *scale,
                                             (y + action_positions[action][1] - action_radius ) *scale,
                                             (x + action_positions[action][0] + action_radius ) *scale,
                                             (y + action_positions[action][1] + action_radius ) *scale,
                                             outline=color_string, fill=color_string )
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale,
                                     (curr_y + 0.5 - curr_radius ) * scale,
                                     (curr_x + 0.5 + curr_radius ) * scale,
                                     (curr_y + 0.5 + curr_radius ) * scale,
                                     outline="#fff", fill="#fff" )
        self.root.update()

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

mission_file = './grammar_demo.xml'
quest_file = './quest_entities.xml'
agent = TabQAgent(mission_file, quest_file)
try:
    agent.host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent.host.getUsage())
    exit(1)
if agent.host.receivedArgument("help"):
    print(agent.host.getUsage())
    exit(0)

# -- set up the mission -- #
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
# add 20% holes for interest
"""for x in range(1,4):
    for z in range(1,13):
        if random.random()<0.1:
            my_mission.drawBlock( x,45,z,"lava")"""

max_retries = 3

if agent.host.receivedArgument("test"):
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
            agent.host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2.5)

    print("Waiting for the mission to start", end=' ')
    world_state = agent.host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent.host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    print()

    # -- run the agent in the world -- #
    cumulative_reward = agent.run()
    print('Cumulative reward: %d' % cumulative_reward)
    cumulative_rewards += [ cumulative_reward ]

    # -- clean up -- #
    time.sleep(0.5) # (let the Mod reset)

print("Done.")

print()
print("Cumulative rewards for all %d runs:" % num_repeats)
print(cumulative_rewards)
