from __future__ import print_function
from __future__ import division

# Basic test of multi-agent mission concept.
# To use:
# 1: Start two mods (make sure the ClientPool is set up to point to them, see below.)
# 2: Start agent one - eg "python multimaze.py"
# 3: Start agent two - eg "python multimaze --role 1"
# They should find each other and begin running missions.


#from snake import Snake
from game import Game
from numpy import *

from agent import Agent
from memory import Memory, ExperienceReplay

from PIL import Image
from builtins import range
from past.utils import old_div
import os
os.environ["MALMO_XSD_PATH"] = '''/home/kylexiao/MalmoPlatform/Schemas'''
try:
    import MalmoPython
    import malmoutils
except ImportError:
    import malmo.MalmoPython as MalmoPython
    import malmo.malmoutils as malmoutils

# import malmo.MalmoPython as MalmoPython
# import malmo.malmoutils as malmoutils

import random
import sys
import time
import json
import logging
if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk
#import malmoutils

#pip install future
#pip install malmoenv
#conda.exe update -n base -c defaults conda
#conda.exe install --channel https://conda.anaconda.org/crowdAI malmo
#


#cuda 10
#sudo pip uninstall protobuf
#tensorflow-gpu v 1.10.0
#cudnn 7.5
#sudo pip install -U protobuf

timeBetweenActions = 0.1
oracle_type = "SYNTHETIC"
oracle_advice = "move 1"
advice_wait_time = 10 #in seconds
agent_host = None

def processFrameMaze(frame):
    print('hi')
    # try:
        # im = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) )
        # #image.save( 'TEST.png' )
        # indices = np.dstack(np.indices(im.shape[:2]))
        # data = np.concatenate((im, indices), axis=-1)
        # return data
    # except:
    #     print('not enough frame data')

class MalmoGameControls(Game):
    def __init__(self, grid_size=64):
        self.reset()
        self.actions = {0:'move 1', 1:'turn 1', 2:'turn -1', 3:'turn -1'} #action 3 will happen 2 times ("turn around") per line 120
        self.location = [0,0,0]
        self.stepsTaken = 0
        self.score = 0
        self.gotToEnd = False
        self.lastFrame = np.zeros((64,64))
        self.subgoalA = False
        self.subgoalB = False

    @property
    def name(self):
        return "Malmo"
    
    @property
    def nb_actions(self):
        return 4
    
    def reset(self):
        global agent_host
        self.score = 0
        self.stepsTaken = 0
        self.lastFrame = np.zeros((64,64))
        self.gotToEnd = False
        self.subgoalA = False
        self.subgoalB = False
        self.location = [0,0,0]
        print("RESET")
        time.sleep(1.0)
        #print(agent_host)
        tp_command = "chat /tp 7.5 12.0 3.5"
        agent_host.sendCommand( tp_command )
        time.sleep(1.0)
        print("-----------------------TELEPORTED----------------------")
        #time.sleep(5.0)

    def play(self, action):
        global agent_host
        actionToSend = self.actions[action]
        if ( action < 3 ):
            agent_host.sendCommand(actionToSend)
        else:
            agent_host.sendCommand(actionToSend)
            time.sleep(0.1)
            agent_host.sendCommand(actionToSend)
        self.stepsTaken = self.stepsTaken + 1
        #print(self.stepsTaken)
        #world_state = agent_host.peekWorldState()
        #time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.video_frames) > 0:
            frame = world_state.video_frames[-1]
            temp = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) ).convert('L')
            self.lastFrame = array(temp)
        if len(world_state.observations) > 0:
            obs_text = world_state.observations[-1].text
            obs = json.loads(obs_text) # most recent observation

            if not u'XPos' in obs or not u'ZPos' in obs:
                x = 0
            else:
                self.location = [int(obs[u'XPos']),int(obs[u'ZPos']),int(obs[u'Yaw'])]
            #current_s = "%d:%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']), int(obs[u'Yaw']))

            if int(obs[u'XPos']) >= 15 and int(obs[u'ZPos']) >= 15:
                print('END!')
                self.score = self.score + 10000
                self.gotToEnd = True
                time.sleep(3)
            elif self.subgoalA == False and int(obs[u'ZPos']) > 7:
                self.score = self.score + 1500
                self.subgoalA = True
            elif self.subgoalB == False and int(obs[u'ZPos']) > 13:
                self.score = self.score + 1500
                self.subgoalB = True
            else:
                self.score = self.score - 0.5
        else:
            time.sleep(0.1)
            print("World State Skip")
            world_state = agent_host.getWorldState()
            if len(world_state.video_frames) > 0:
                frame = world_state.video_frames[-1]
                temp = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) ).convert('L')
                self.lastFrame = array(temp)
            if len(world_state.observations) > 0:
                obs_text = world_state.observations[-1].text
                obs = json.loads(obs_text) # most recent observation

                if not u'XPos' in obs or not u'ZPos' in obs:
                    x = 0
                else:
                    self.location = [int(obs[u'XPos']),int(obs[u'ZPos']),int(obs[u'Yaw'])]
                #current_s = "%d:%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']), int(obs[u'Yaw']))

                if int(obs[u'XPos']) >= 15 and int(obs[u'ZPos']) >= 15:
                    print('END!')
                    self.score = self.score + 10000
                    self.gotToEnd = True
                    time.sleep(3)
                elif self.subgoalA == False and int(obs[u'ZPos']) > 7:
                    self.score = self.score + 1500
                    self.subgoalA = True
                elif self.subgoalB == False and int(obs[u'ZPos']) > 13:
                    self.score = self.score + 1500
                    self.subgoalB = True
                else:
                    self.score = self.score - 0.5
    def get_grid(self):
        return self.location

    def get_state(self):
        # world_state = agent_host.peekWorldState()
        # frame = world_state.video_frames[0]
        # temp = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) )
        # temp=temp.convert('1')
        # #x=temp.shape[0]
        # #y=temp.shape[1]*temp.shape[2]

        # #temp.resize((x,y)) # a 2D array
        # A = array(temp)
        # print(A.shape)
        return self.lastFrame

    def get_score(self):
        return self.score

    def is_over(self):
        return ((self.stepsTaken >= 1000) or (self.gotToEnd == True))

    def is_won(self):
        return self.gotToEnd

    def get_frame(self):
        return self.get_state()

    def draw(self):
        return self.get_state()

    def get_possible_actions(self):
        return range(self.nb_actions)

# class TabQAgent(object):
#     """Tabular Q-learning agent for discrete state/action spaces."""

#     def __init__(self):
#         self.epsilon = 0.5 # chance of taking a random action instead of the best
#         self.advice_epsilon = 0.4 # ARBITER PARAMETERS - 10% chance to ask for advice
#         self.logger = logging.getLogger(__name__)
#         if False: # True if you want to see more information
#             self.logger.setLevel(logging.DEBUG)
#         else:
#             self.logger.setLevel(logging.INFO)
#         self.logger.handlers = []
#         self.logger.addHandler(logging.StreamHandler(sys.stdout))

#         #self.actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
#         self.actions = ["move 1","turn -1", "turn 1"]
#         self.q_table = {}
#         self.o_table = {} #oracle table best actions, x pos, z pos, yaw (looking direction)
#         self.canvas = None
#         self.root = None

#     def updateQTable( self, reward, current_state ):
#         """Change q_table to reflect what we have learnt."""
        
#         # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
#         old_q = self.q_table[self.prev_s][self.prev_a]
  
#         sum = 0
#         for x in range(0, len(self.actions)):
#             sum += self.q_table[current_state][x]
#         new_q = sum / 4
          
        
#         # assign the new action value to the Q-table
#         self.q_table[self.prev_s][self.prev_a] = new_q
        
#     def updateQTableFromTerminatingState( self, reward ):
#         """Change q_table to reflect what we have learnt, after reaching a terminal state."""
        
#         # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
#         old_q = self.q_table[self.prev_s][self.prev_a]
        
#         # TODO: what should the new action value be?
#         new_q = reward

        
#         # assign the new action value to the Q-table
#         self.q_table[self.prev_s][self.prev_a] = new_q
        
#     def act(self, world_state, agent_host, current_r, oracle_type ):
#         """take 1 action in response to the current world state"""
#         global oracle_advice
#         global advice_wait_time
#         obs_text = world_state.observations[0].text
#         obs = json.loads(obs_text) # most recent observation
#         self.logger.info(list(obs.keys()))
#         if not u'XPos' in obs or not u'ZPos' in obs:
#             self.logger.error("Incomplete observation received: %s" % obs_text)
#             return 0
#         current_s = "%d:%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']), int(obs[u'Yaw']))
#         self.logger.info("State: %s (x = %.2f, z = %.2f, yaw = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos']), float(obs[u'Yaw'])))
#         if current_s not in self.q_table:
#             self.q_table[current_s] = ([0] * len(self.actions))

#         # update Q values
#         if self.prev_s is not None and self.prev_a is not None:
#             self.updateQTable( current_r, current_s )

#         #self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) )

#         # ARBITER PARAMETERS
#         # select the next action
#         rnd = random.random()
#         if rnd < self.epsilon and rnd > self.advice_epsilon:
#             a = random.randint(0, len(self.actions) - 1)
#             self.logger.info("Random action: %s" % self.actions[a])
#             # try to send the selected action, only update prev_s if this succeeds
#             try:
#                 agent_host.sendCommand(self.actions[a])
#                 self.prev_s = current_s
#                 self.prev_a = a

#             except RuntimeError as e:
#                 self.logger.error("Failed to send command: %s" % e)
#         elif rnd < self.epsilon and rnd < self.advice_epsilon:
#             #a = random.randint(0, len(self.actions) - 1)
#             if oracle_type == "HUMAN":
#                 agent_host.sendCommand("chat Should I turn right, turn left or move forward?")
#                 time.sleep(advice_wait_time)
#                 chat_world_state = agent_host.peekWorldState()
#                 #ms = chat_world_state.observations[-1].text
#                 for ob in chat_world_state.observations:
#                     ms = ob.text
#                     o = json.loads(ms)
#                     chat = o.get(u'Chat', "")
#                     #chat = o[u'Chat']
#                     #print(chat)
#                     # clist = []
#                     # for command in chat:
#                     #     clist.append(command)
#                     for c in chat:
#                         print(c)
#                         print("____________________")
#                         parts = c.split("> ")
#                         if len(parts) > 1:
#                             print(parts[0])
#                             if parts[0] == "<Agent":
#                                 continue
#                             elif parts[0] == "<Player":
#                                 print(parts[1])
#                                 if parts[1] == "move forward":
#                                     oracle_advice = "move 1"
#                                 if parts[1] == "turn left":
#                                     oracle_advice = "turn -1"
#                                 if parts[1] == "turn right":
#                                     oracle_advice = "turn 1"

#                                 agent_host.sendCommand(oracle_advice)
#                                 if oracle_advice == "move 1":
#                                     a = 0
#                                 if oracle_advice == "turn -1":
#                                     a = 1    
#                                 if oracle_advice == "turn 1":
#                                     a = 2    
#                                 self.prev_s = current_s
#                                 self.prev_a = a
#                                 return current_r
#                             else:
#                                 print("No player response... taking random action")
#                                 a = random.randint(0, len(self.actions) - 1)
#                                 self.logger.info("Random action: %s" % self.actions[a])
#                                 # try to send the selected action, only update prev_s if this succeeds
#                                 try:
#                                     agent_host.sendCommand(self.actions[a])
#                                     self.prev_s = current_s
#                                     self.prev_a = a
#                                     return current_r

#                                 except RuntimeError as e:
#                                     self.logger.error("Failed to send command: %s" % e)

#             msg = world_state.observations[-1].text
#             obs = json.loads(msg)
#             current_x = int(obs[u'XPos'])
#             current_z = int(obs[u'ZPos'])
#             yaw = int(obs[u'Yaw'])
#             if current_z < 6 and current_x < 12:
#                 if yaw == 270:
#                     oracle_advice = "move 1"
#                 else:
#                     oracle_advice = "turn 1"
#             elif current_z < 6 and current_x >= 12:
#                 if yaw == 270:
#                     oracle_advice = "turn 1"
#                 elif yaw == 0:
#                     oracle_advice = "move 1"
#                 else:
#                     oracle_advice = "turn -1"
#             elif current_z == 6:
#                 if yaw == 0:
#                     oracle_advice = "move 1"
#                 else:
#                     oracle_advice = "turn 1"
#             elif current_z > 6 and current_z < 12 and current_x > 6:
#                 if yaw == 90:
#                     oracle_advice = "move 1"
#                 else:
#                     oracle_advice = "turn 1"
#             elif current_z < 12 and current_x <= 6:
#                 if yaw == 90:
#                     oracle_advice = "turn -1"
#                 elif yaw == 0:
#                     oracle_advice = "move 1"
#                 else:
#                     oracle_advice = "turn 1"
#             elif current_z == 12:
#                 if yaw == 0:
#                     oracle_advice = "move 1"
#                 else:
#                     oracle_advice = "turn 1"
#             elif current_z < 20:
#                 if yaw == 0:
#                     oracle_advice = "move 1"
#                 else:
#                     oracle_advice = "turn 1"
#             elif current_z == 20:
#                 if yaw == 270:
#                     oracle_advice = "move 1"
#                 else:
#                     oracle_advice = "turn 1"
#             self.logger.info("Arbiter action: %s" % oracle_advice)
#             # try to send the selected action, only update prev_s if this succeeds
#             try:
#                 agent_host.sendCommand(oracle_advice)
#                 if oracle_advice == "move 1":
#                     a = 0
#                 if oracle_advice == "turn -1":
#                     a = 1    
#                 if oracle_advice == "turn 1":
#                     a = 2    
#                 self.prev_s = current_s
#                 self.prev_a = a

#             except RuntimeError as e:
#                 self.logger.error("Failed to send command: %s" % e)
#         else:
#             m = max(self.q_table[current_s])
#             #self.logger.info("Current values: %s" % ",".join(str(x) for x in self.q_table[current_s]))
#             l = list()
#             for x in range(0, len(self.actions)):
#                 if self.q_table[current_s][x] == m:
#                     l.append(x)
#             y = random.randint(0, len(l)-1)
#             a = l[y]
#             self.logger.info("Taking q action: %s" % self.actions[a])
#             # try to send the selected action, only update prev_s if this succeeds
#             try:
#                 agent_host.sendCommand(self.actions[a])
#                 self.prev_s = current_s
#                 self.prev_a = a

#             except RuntimeError as e:
#                 self.logger.error("Failed to send command: %s" % e)



#         return current_r


#     def run(self, agent_host, timeBetweenActions, oracle_type):
#         """run the agent on the world"""

#         total_reward = 0
        
#         self.prev_s = None
#         self.prev_a = None
        
#         is_first_action = True
        
#         # main loop:
#         world_state = agent_host.getWorldState()
#         while world_state.is_mission_running:

#             current_r = 0
            
#             if is_first_action:
#                 # wait until have received a valid observation
#                 while True:
#                     time.sleep(timeBetweenActions)
#                     world_state = agent_host.getWorldState()
#                     for error in world_state.errors:
#                         self.logger.error("Error: %s" % error.text)
#                     for reward in world_state.rewards:
#                         current_r += reward.getValue()
#                     if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
#                         total_reward += self.act(world_state, agent_host, current_r, oracle_type)
#                         break
#                     if not world_state.is_mission_running:
#                         break
#                 is_first_action = False
#             else:
#                 # wait for non-zero reward
#                 while world_state.is_mission_running and current_r == 0:
#                     time.sleep(timeBetweenActions)
#                     processFrameMaze(world_state.video_frames[-1])
#                     world_state = agent_host.getWorldState()
#                     for error in world_state.errors:
#                         self.logger.error("Error: %s" % error.text)
#                     for reward in world_state.rewards:
#                         current_r += reward.getValue()
#                         print(current_r)
#                 # allow time to stabilise after action
#                 while True:
#                     time.sleep(timeBetweenActions)
#                     world_state = agent_host.getWorldState()
#                     for error in world_state.errors:
#                         self.logger.error("Error: %s" % error.text)
#                     for reward in world_state.rewards:
#                         current_r += reward.getValue()
#                     if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
#                         total_reward += self.act(world_state, agent_host, current_r, oracle_type)
#                         print(total_reward)
#                         break
#                     if not world_state.is_mission_running:
#                         break

#         # process final reward
#         self.logger.debug("Final reward: %d" % current_r)
#         total_reward += current_r

#         # update Q values
#         if self.prev_s is not None and self.prev_a is not None:
#             self.updateQTableFromTerminatingState( current_r )
            
#         #self.drawQ()
    
#         return total_reward
        
#     def drawQ( self, curr_x=None, curr_y=None ):
#         scale = 20
#         world_x = 21
#         world_y = 21
#         if self.canvas is None or self.root is None:
#             self.root = tk.Tk()
#             self.root.wm_title("Q-table")
#             self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
#             self.canvas.grid()
#             self.root.update()
#         self.canvas.delete("all")
#         action_inset = 0.1
#         action_radius = 0.1
#         curr_radius = 0.2
#         action_positions = [ ( 0.5, action_inset ), ( 0.5, 1-action_inset ), ( action_inset, 0.5 ), ( 1-action_inset, 0.5 ) ]
#         # (NSWE to match action order)
#         min_value = -100
#         max_value = 100
#         for x in range(world_x):
#             for y in range(world_y):
#                 s = "%d:%d" % (x,y)
#                 self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")
#                 for action in range(3):
#                     if not s in self.q_table:
#                         continue
#                     value = self.q_table[s][action]
#                     color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
#                     color = max( min( color, 255 ), 0 ) # ensure within [0,255]
#                     color_string = '#%02x%02x%02x' % (255-color, color, 0)
#                     self.canvas.create_oval( (x + action_positions[action][0] - action_radius ) *scale,
#                                              (y + action_positions[action][1] - action_radius ) *scale,
#                                              (x + action_positions[action][0] + action_radius ) *scale,
#                                              (y + action_positions[action][1] + action_radius ) *scale, 
#                                              outline=color_string, fill=color_string )
#         if curr_x is not None and curr_y is not None:
#             self.canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale, 
#                                      (curr_y + 0.5 - curr_radius ) * scale, 
#                                      (curr_x + 0.5 + curr_radius ) * scale, 
#                                      (curr_y + 0.5 + curr_radius ) * scale, 
#                                      outline="#fff", fill="#fff" )
#         self.root.update()

malmoutils.fix_print()

MalmoPython.setLogging("", MalmoPython.LoggingSeverityLevel.LOG_OFF)

items = {'red_flower':'flower',
         'apple':'apple',
         'iron_sword':'sword',
         'iron_pickaxe':'pickaxe',
         'diamond_sword':'sword'
         }
obj_id = list(items.keys())[random.randint(0, len(items)-1)]

def genExperimentID( episode ):
    return "MMExp#" + str(episode)

#http://www.minecraft101.net/superflat/ can be used to generate the generatorString for the flatworld
def GetMissionXML( current_seed, xorg, yorg, zorg ):
    return '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary></Summary>
        </About>

        <ServerSection>
                <ServerInitialConditions>
                    <Time>
                        <StartTime>6000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                    <AllowSpawning>false</AllowSpawning>
              </ServerInitialConditions>
            <ServerHandlers>

                <FlatWorldGenerator seed="1" generatorString="3;1*minecraft:bedrock,7*minecraft:dirt,1*minecraft:grass;4;decoration"/>
                  <DrawingDecorator>
                    <!-- coordinates for cuboid are inclusive -->
                    <DrawCuboid x1="-5" y1="10" z1="-5" x2="27" y2="13" z2="27" type="bedrock" /> <!-- limits of our arena, order of drawing matters! -->
                    <DrawCuboid x1="0" y1="11" z1="0" x2="19" y2="17" z2="19" type="air" /> <!-- limits of our arena -->
                    <DrawCuboid x1="-4" y1="13" z1="-4" x2="26" y2="28" z2="26" type="air" /> <!-- limits of our arena -->
                    <DrawCuboid x1="0" y1="11" z1="6" x2="11" y2="13" z2="6" type="mossy_cobblestone" /> <!-- Hedge A -->
                    <DrawCuboid x1="8" y1="11" z1="13" x2="19" y2="13" z2="13" type="mossy_cobblestone" /> <!-- Hedge B -->
                    <DrawCuboid x1="0" y1="11" z1="13" x2="3" y2="13" z2="13" type="mossy_cobblestone" /> <!-- Hedge C -->
                    <DrawCuboid x1="15" y1="11" z1="14" x2="19" y2="14" z2="19" type="brick_block" /> <!-- Red House -->
                    <DrawCuboid x1="15" y1="11" z1="16" x2="18" y2="12" z2="17" type="air" /> <!-- Red House Interior -->
                    <DrawCuboid x1="16" y1="11" z1="0" x2="19" y2="14" z2="5" type="gold_block" /> <!-- Yellow House -->
                    <DrawCuboid x1="16" y1="11" z1="2" x2="18" y2="12" z2="3" type="air" /> <!-- Yellow House Inside/Door -->
                    <DrawCuboid x1="0" y1="11" z1="0" x2="4" y2="14" z2="5" type="lapis_block" /> <!-- Blue House -->
                    <DrawCuboid x1="1" y1="11" z1="2" x2="4" y2="12" z2="3" type="air" /> <!-- Blue House Inside/Door -->
                    <DrawCuboid x1="0" y1="11" z1="14" x2="3" y2="14" z2="19" type="emerald_block" /> <!-- Green House -->
                    <DrawCuboid x1="1" y1="11" z1="16" x2="3" y2="12" z2="17" type="air" /> <!-- Green House Inside/Door -->
                    <DrawBlock x="19" y="10" z="20" type="redstone_block" /> <!-- Target -->
                    <DrawBlock x="19" y="10" z="21" type="redstone_block" /> <!-- Target -->
                    <DrawBlock x="19" y="10" z="19" type="redstone_block" /> <!-- Target -->
                    <DrawBlock x="21" y="10" z="19" type="redstone_block" /> <!-- Target -->
                    <DrawBlock x="20" y="10" z="19" type="redstone_block" /> <!-- Target -->
                    <DrawSphere x="10" y="12" z="119" radius="40" type="lapis_block" /> <!-- Blue Mountain -->
                    <DrawSphere x="10" y="12" z="-89" radius="40" type="gold_block" /> <!-- Gold Mountain -->
                    <DrawSphere x="-99" y="12" z="10" radius="40" type="brick_block" /> <!-- Red Mountain -->
                    <DrawSphere x="119" y="12" z="10" radius="40" type="emerald_block" /> <!-- Green Mountain -->
                  </DrawingDecorator>
                <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>Player</Name>
            <AgentStart>
                <Placement x="-1" y="14.0" z="-1" yaw="1"/>
                <Inventory>
                    <InventoryObject type="golden_helmet" slot="39" quantity="1"/>
                    <InventoryObject type="golden_chestplate" slot="38" quantity="1"/>
                    <InventoryObject type="golden_leggings" slot="37" quantity="1"/>
                    <InventoryObject type="golden_boots" slot="36" quantity="1"/>
                  </Inventory>
            </AgentStart>
            <AgentHandlers>
                <ObservationFromFullStats/>
                <DiscreteMovementCommands>
                  <ModifierList type="deny-list">
                    <command>attack</command>
                  </ModifierList>
                </DiscreteMovementCommands>
                <VideoProducer>
                    <Width>400</Width>
                    <Height>300</Height>
                </VideoProducer>
            </AgentHandlers>
        </AgentSection>
        
        <AgentSection mode="Survival">
            <Name>Agent1</Name>
            <AgentStart>
                <Placement x="5.5" y="12.0" z="3.5"/>
            </AgentStart>
            <AgentHandlers>
                <ObservationFromFullStats/>
                <ObservationFromChat/>
                <DiscreteMovementCommands/>
                <AbsoluteMovementCommands/>
                <ChatCommands />
                <VideoProducer viewpoint="0" want_depth="false">
                    <Width>64</Width>
                    <Height>64</Height>
                  </VideoProducer>
            </AgentHandlers>
        </AgentSection>

        <AgentSection mode="Survival">
            <Name>Villager</Name>
            <AgentStart>
                <Placement x="27.5" y="22.0" z="27.5" yaw="270"/>
            </AgentStart>
            <AgentHandlers>
                <ObservationFromFullStats/>
                <LuminanceProducer>
                    <Width>20</Width>
                    <Height>10</Height>
                </LuminanceProducer>
            </AgentHandlers>
        </AgentSection>
  </Mission>'''

agent_host = MalmoPython.AgentHost()
agent_host.addOptionalIntArgument( "behavior,b", "If behavior is set to 1, the agent will act more slowly/randomly and occasionally ask for text advice", 0)
agent_host.addOptionalIntArgument( "role,r", "For multi-agent missions, the role of this agent instance", 0)
agent_host.addOptionalIntArgument( "advice,a", "Advice Type. 0 is never asks, 1 is a percentage based oracle, 2 is percentage based oracle with newtonian", 0)
agent_host.addOptionalIntArgument( "friction,f", "Advice Friction: how many actions to follow advice for before recomputing. Default is 3", 3)

malmoutils.parse_command_line(agent_host)

role = agent_host.getIntArgument("role")
print("Will run as role",role)


behavior = agent_host.getIntArgument("behavior")
if behavior==1:
    print("Running in human-input mode...")
    timeBetweenActions = 2.0
    oracle_type = "HUMAN"

if agent_host.receivedArgument("test"):
    if role == 0:
        forward_args = " --test --role 1"
        if agent_host.receivedArgument('record_video'):
            forward_args += " --record_video"
        recordingsDirectory = agent_host.getStringArgument('recording_dir')
        if recordingsDirectory:
            forward_args += " --recording_dir " + recordingsDirectory
        print("For test purposes, launching self with [{}] now.".format(forward_args))
        import subprocess
        subprocess.Popen(sys.executable + " " + __file__ + forward_args, shell=True)
    num_episodes = 5
else:
    num_episodes = 1

#agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.KEEP_ALL_OBSERVATIONS)
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

# Create a client pool here - this assumes two local mods with default ports,
# but you could have two mods on different machines, and specify their IP address here.
client_pool = MalmoPython.ClientPool()
client_pool.add( MalmoPython.ClientInfo( "127.0.0.1", 10000 ) )
client_pool.add( MalmoPython.ClientInfo( "127.0.0.1", 10001 ) )
client_pool.add( MalmoPython.ClientInfo( "127.0.0.1", 10002 ) )

chat_frequency = 30 # if we send chat messages too frequently the agent will be disconnected for spamming
num_steps_since_last_chat = 0
cumulative_rewards = []
for iRepeat in range(num_episodes):

    xorg = (iRepeat % 64) * 32
    zorg = ((old_div(iRepeat, 64)) % 64) * 32
    yorg = 200 + ((old_div(iRepeat, (64*64))) % 64) * 8

    print("Mission " + str(iRepeat) + " --- starting at " + str(xorg) + ", " + str(yorg) + ", " + str(zorg))
    
    validate = True
    my_mission = MalmoPython.MissionSpec(GetMissionXML(iRepeat, xorg, yorg, zorg), validate)

    my_mission_record = malmoutils.get_default_recording_object(agent_host, "episode_{}_role_{}".format(iRepeat + 1, role))
    unique_experiment_id = genExperimentID(iRepeat) # used to disambiguate multiple running copies of the same mission
 
    max_retries = 3
    retry = 0
    while True:
        try:
            print("Calling startMission...")
            agent_host.startMission( my_mission, client_pool, my_mission_record, role, unique_experiment_id )
            #agent_host.startMission( my_mission, client_pool )
            break
        except MalmoPython.MissionException as e:
            errorCode = e.details.errorCode
            if errorCode == MalmoPython.MissionErrorCode.MISSION_SERVER_WARMING_UP:
                print("Server not online yet - will keep waiting as long as needed.")
                time.sleep(1)
            elif errorCode in [MalmoPython.MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE,
                               MalmoPython.MissionErrorCode.MISSION_SERVER_NOT_FOUND]:
                retry += 1
                if retry == max_retries:
                    print("Error starting mission:", e)
                    exit(1)
                print("Resources not found - will wait and retry a limited number of times.")
                time.sleep(5)
            else:
                print("Blocking error:", e.message)
                exit(1)

    print("Waiting for the mission to start", end=' ')
    start_time = time.time()
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            for err in world_state.errors:
                print(err)
            exit(1)
        if time.time() - start_time > 120:
            print("Mission failed to begin within two minutes - did you forget to start the other agent?")
            exit(1)
    print()
    print("Mission has begun.")
    if (role == 1):
        # agent = TabQAgent()
        # cumulative_reward = agent.run(agent_host, timeBetweenActions, oracle_type)
        # print('Cumulative reward: %d' % cumulative_reward)
        # cumulative_rewards += [ cumulative_reward ]
        #fn = "9nyc-250-1000-epr8-heat-adam.csv"
        #fn = "400-rl-nopool.csv"
        fn = "3-normal.csv"
        with open(fn,'a') as f:
          f.write('session_id,advice_type,meta_advice_type,time,epoch,frames,score,win_perc,epsilon,loss,steps,advice_count,non_clamped_ac,model_actions'+'\n')
          f.flush()
          f.close()

        # fn2 = "heat.csv"
        # with open(fn2,'a') as f2:
        #   f2.write('advice_type,x,z,t'+'\n')
        #   f2.flush()
        #   f2.close()

        from keras.models import Sequential
        from keras.layers import *
        from keras.optimizers import *
        from keras import losses
        from keras import backend as K
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        from keras.utils import plot_model
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                            # (nothing gets printed in Jupyter, only if you run it standalone)
        #sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras
        print(K.tensorflow_backend._get_available_gpus())
        K.set_image_dim_ordering('th')

        grid_size = 64
        vision_size = 64
        nb_frames = 50 #frame memory
        nb_actions = 4

        # gamma = decay rate of past observations
        total_sessions = 4 # 15 sessions at 80 epochs would take about 10 hours to run
        for i in range(1,total_sessions):
            model = Sequential()
            model.add(Conv2D(64, (3, 3), input_shape=(nb_frames, vision_size, vision_size)))
            model.add(BatchNormalization())
            model.add(Activation("relu"))

            model.add(Conv2D(64, (3, 3)))
            model.add(BatchNormalization())
            model.add(Activation("relu"))

            #model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(32, (3, 3)))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
            #model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(16, (3, 3)))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
            #model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(8, (3, 3)))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
            #model.add(Dropout(0.15))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(nb_actions))
            model.compile(loss=losses.mean_squared_logarithmic_error, optimizer='adam')
            plot_model(model, to_file='model.png', show_shapes=True)
            time.sleep(1)
            sim = MalmoGameControls(grid_size)

            agent = Agent(model=model, memory_size=4000, nb_frames=nb_frames)
            agent.train(sim, batch_size=8, nb_epoch=250, gamma=0.95, epsilon_rate=0.99, epsilon=[0.99, .01], total_sessions=total_sessions, session_id=i, checkpoint=25)
            #agent.play(sim)
        print("Finished Running")

    # main loop for players/non DL agents:
    while world_state.is_mission_running:
        time.sleep(30.0)
        world_state = agent_host.getWorldState()
        if world_state.is_mission_running:
            if (role == 0):
                print("Free running")
            elif (role == 2):
                print("Doing nothing")
                #agent_host.sendCommand("chat Find me some item X")
                # msg = world_state.observations[-1].text
                # ob = json.loads(msg)
                # if u'LineOfSight' in ob:
                #     los = ob[u'LineOfSight']
                #     print(los)
                # print(ob) #spawn random item when touching blue blocks below quest giver, reward for picking up item, reward for dropping item
        if len(world_state.errors) > 0:
            for err in world_state.errors:
                print(err)
            
    print("Mission has stopped.")
    print()
