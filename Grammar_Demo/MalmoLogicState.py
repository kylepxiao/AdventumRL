import json
import MalmoPython
from textworld.logic import State, Variable, Proposition, Action, Rule
from xml.etree import ElementTree as ET
from parse_mission_xml import namespace
from constants import *
from math import sin, cos, exp

class MalmoLogicState(State):
    def __init__(self, facts=None, actions=[], triggers=[], entities={}, boundaries={}, goal=None):
        super().__init__(facts=facts)
        self.actions = actions
        self.triggers = triggers
        self.entities = entities
        self.boundaries = boundaries
        self.goal = goal
        self.world_bounds = entities['world_bounds']
        self.yaw = 0
        self.pitch = 0

        #TODO: Allow user to specify global set of variables and relations a priori
        self.varNames = [v.name for v in self.variables]
        self.relations = list(set([fact.name for fact in self.facts]))

    def copy(self):
        return MalmoLogicState(facts=self.facts, actions=self.actions, triggers=self.triggers, entities=self.entities, boundaries=self.boundaries, goal=self.goal)

    def all_applicable_actions(self, rules=None, mapping=None):
        if rules is not None:
            return super().all_applicable_actions(rules, mapping)
        else:
            return super().all_applicable_actions(self.actions, mapping)

    def updateLogicState(self, world_state):
        if len(world_state.observations) < 1:
            return

        observation = json.loads(world_state.observations[-1].text)

        # Clear all inventory propositions in world state
        oldInventoryProps = set()
        for fact in self.facts:
            if fact.names[-1] == "inventory":
                oldInventoryProps.add(fact)
        #TODO remove inventory items from bounding boxes
        self.remove_facts(oldInventoryProps)

        # Positional update
        self.yaw = observation['Yaw']
        self.pitch = observation['Pitch']
        self.entities['player'].setPosition(observation['XPos'], observation['YPos'], observation['ZPos'])

        # Check if player in boundaries
        for boundary in self.boundaries:
            if self.entities[boundary].contains(self.entities['player']):
                self.add_fact( Proposition("in", [self.entities['player'], self.entities[boundary]]) )
            else:
                self.remove_fact( Proposition("in", [self.entities['player'], self.entities[boundary]]) )

        # Check if player next to item
        for boundary in self.entities:
            if self.entities[boundary].by(self.entities['player']):
                self.add_fact( Proposition("by", [self.entities['player'], self.entities[boundary]]) )
            else:
                self.remove_fact( Proposition("by", [self.entities['player'], self.entities[boundary]]) )

        # Add all inventory propositions to world state
        curInventoryVar = Variable("current", "inventory")
        if 'inventory' in observation:
            inventory = observation['inventory']
            inventoryVar = Variable("inventory", "inventory")
            for i, item in enumerate(inventory):
                key = item['type']
                self.add_fact( Proposition("in", [self.entities[key], inventoryVar]) )
                self.entities[key].setPosition(observation['XPos'], observation['YPos'], observation['ZPos'])
                if i == 0:
                    self.add_fact( Proposition("in", [self.entities[key], curInventoryVar]) )

                # Check if items in inventory leave boundaries
                for boundary in self.boundaries:
                    if self.entities[boundary].contains(self.entities[key]):
                        self.add_fact( Proposition("in", [self.entities[key], self.entities[boundary]]) )
                    else:
                        self.remove_fact( Proposition("in", [self.entities[key], self.entities[boundary]]) )

    def getStateKey(self, entity='player'):
        actionflags = [1 if self.is_applicable(action) else 0 for action in self.actions]
        actionstr = ''.join(map(str, actionflags))
        triggerflags = [1 if self.is_fact(trigger) else 0 for trigger in self.triggers]
        triggerstr = ''.join(map(str, triggerflags))
        return "%d:%d|%s" % (int(self.entities[entity].x), int(self.entities[entity].z), actionstr + ":" + triggerstr)

    def getStateEmbedding(self, entity='player'):
        if self.world_bounds is None:
            actionflags = [1 if self.is_applicable(action) else 0 for action in self.actions]
            triggerflags = [1 if self.is_fact(trigger) else 0 for trigger in self.triggers]
            xValue = 1 / (1 + exp(float(self.entities[entity].x)))
            zValue = 1 / (1 + exp(float(self.entities[entity].z)))
            return [xValue, zValue] + actionflags + triggerflags
        else:
            (x1, y1, z1), (x2, y2, z2) = self.world_bounds.roundPosition()
            actionflags = [1 if self.is_applicable(action) else 0 for action in self.actions]
            triggerflags = [1 if self.is_fact(trigger) else 0 for trigger in self.triggers]
            xValue = [1 if round(self.entities[entity].x) == i else 0 for i in range(x1, x2+1, 1)]
            zValue = [1 if round(self.entities[entity].z) == i else 0 for i in range(z1, z2+1, 1)]
            return xValue + zValue + actionflags + triggerflags

    def getRelationalKnowledgeGraph(self):
        graph = []
        for fact in self.facts:
            if len(fact.names) == 1:
                graph.append([fact.names[0], fact.name, fact.names[0]])
            else:
                graph += [[fact.names[p1], fact.name, fact.names[p2]] for p1 in range(len(fact.names)) for p2 in range(p1+1,len(fact.names))]

        # convert to ids
        for i in range(len(graph)):
            if graph[i][0] in self.varNames:
                graph[i][0] = self.varNames.index(graph[i][0])
            else:
                self.varNames.append(graph[i][0])
                graph[i][0] = len(self.varNames) - 1

            if graph[i][1] in self.relations:
                graph[i][1] = self.relations.index(graph[i][1])
            else:
                self.relations.append(graph[i][1])
                graph[i][1] = len(self.relations) - 1

            if graph[i][2] in self.varNames:
                graph[i][2] = self.varNames.index(graph[i][2])
            else:
                self.varNames.append(graph[i][2])
                graph[i][2] = len(self.varNames) - 1

        return graph

    def getUndirectedKnowledgeGraph(self):
        graph = []
        for fact in self.facts:
            if len(fact.names) == 1:
                graph.append([fact.names[0], fact.names[0]])
            else:
                graph += [[fact.names[p1], fact.names[p2]] for p1 in range(len(fact.names)) for p2 in range(p1+1,len(fact.names))]

        # convert to ids
        for i in range(len(graph)):
            if graph[i][0] in self.varNames:
                graph[i][0] = self.varNames.index(graph[i][0])
            else:
                self.varNames.append(graph[i][0])
                graph[i][0] = len(self.varNames) - 1

            if graph[i][1] in self.varNames:
                graph[i][1] = self.varNames.index(graph[i][1])
            else:
                self.varNames.append(graph[i][1])
                graph[i][1] = len(self.varNames) - 1

        return graph


    def getApplicableActions(self):
        return [action for action in self.actions if self.is_applicable(action)]

    def checkGoal(self):
        for subgoal in self.goal:
            prop = subgoal[0]
            val = subgoal[1]
            if self.is_fact(prop) != val:
                return False
        return True

    def goalHeuristic(self):
        total = 0
        for subgoal in self.goal:
            prop = subgoal[0]
            val = subgoal[1]
            if self.is_fact(prop) == val:
                total += 1
        return float(total) / len(self.goal)

    def currentInventoryItem(self):
        for fact in self.facts:
            if fact.names[-1] == 'current':
                return fact
        return None



class LogicalAgentHost(MalmoPython.AgentHost):
    def __init__(self, initialState = None, actions=[], goal=None, triggers=[]):
        super().__init__()
        self.state = initialState
        self.state.goal = goal
        self.state.actions = actions
        self.state.triggers = triggers
        self.state.world_bounds = self.state.entities['world_bounds']
        self.reward = 0
        self.initialState = self.state.copy()
        self.pastHeuristic = self.initialState.goalHeuristic()
        self.heuristicReward = 15

    def updateLogicState(self, world_state):
        self.state.updateLogicState(world_state)

    def checkGoal(self):
        return self.state.checkGoal()

    def getLogicalActions(self):
        return [action for action in self.state.all_applicable_actions()]

    def rewardValue(self):
        if self.reward != 0:
            print("-----\nLOGICAL STATE CHANGE REWARD\n-----")
        """if self.state.goalHeuristic() > self.pastHeuristic:
            self.pastHeuristic = self.state.goalHeuristic()
            self.reward += self.heuristicReward
            print("-----\nHEURISTIC REWARD\n-----")"""
        reward = self.reward
        self.reward = 0
        return reward

    def resetState(self):
        self.state = self.initialState.copy()

    def sendCommand(self, command, is_logical=False):
        if is_logical:
            self.state.apply(command)
            self.reward += command.reward
        command = str(command)
        segments = command.split()
        instruction = segments[0]
        if instruction == "movenorth":
            self.state.entities['player'].moveNorth(segments[1])
        elif instruction == "movesouth":
            self.state.entities['player'].moveSouth(segments[1])
        elif instruction == "moveeast":
            self.state.entities['player'].moveEast(segments[1])
        elif instruction == "movewest":
            self.state.entities['player'].moveWest(segments[1])
        elif instruction == "discardCurrentItem":
            currentProp = self.state.currentInventoryItem()
            dx = cos(self.state.yaw)*cos(self.state.pitch)
            dy = sin(self.state.yaw)*cos(self.state.pitch)
            dz = sin(self.state.pitch)
            self.state.entities[currentProp.names[0]].addPosition(dx, dy, dz)
        elif instruction == "reward":
            self.reward += float(segments[1])
            return
        super().sendCommand(command)

class LogicalAction(Action):
    def __init__(self, name, preconditions, postconditions, command, reward=0):
        super().__init__(name, preconditions, postconditions)
        self.command = command
        self.reward = reward

    def __str__(self):
        return self.command
