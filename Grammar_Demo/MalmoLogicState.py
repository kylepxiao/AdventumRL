import json
import MalmoPython
from textworld.logic import State, Variable, Proposition, Action, Rule
from xml.etree import ElementTree as ET
from parse_mission_xml import namespace
from constants import *
from math import sin, cos

class MalmoLogicState(State):
    def __init__(self, facts=None, actions=[], triggers=[], entities={}, boundaries={}, goal=None):
        super().__init__(facts=facts)
        self.actions = actions
        self.triggers = triggers
        self.entities = entities
        self.boundaries = boundaries
        self.goal = goal
        self.yaw = 0
        self.pitch = 0;

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

        # Add all inventory propositions to world state
        inventoryVar = Variable("inventory", "inventory")
        inventory = observation['inventory']
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

def getInitialWorldState(mission_file=None, quest_file=None):
    state_list = defaultFacts
    # define global objects and propositions
    entities = {}
    objectVars = {}
    objectProps = {}
    props = set()
    boundaries = set()

    if quest_file is not None:
        # parse XML
        questTree = ET.parse(quest_file)
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
                    props.add(Proposition("in", [currentVar, worldVar]))

    if mission_file is not None:
        # parse XML
        missionTree = ET.parse(mission_file)
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
                    props.add(Proposition("in", [currentVar, worldVar]))
                    for boundary in boundaries:
                        if entities[boundary].contains(currentVar):
                            props.add(Proposition("in", [currentVar, entities[boundary]]))
            elif item.tag == ns + "AgentStart":
                for child in item.iter():
                    if child.tag[len(ns):] == "Placement":
                        currentVar = Agent("player", child.attrib['x'], child.attrib['y'], child.attrib['z'])
                        entities['player'] = currentVar
                        props.add(Proposition("in", [currentVar, worldVar]))
                        props.add(Proposition("has", [playerVar, inventoryVar]))

        state_list = state_list.union( props )

    return MalmoLogicState(facts=state_list, entities=entities, boundaries=boundaries)

class LogicalAgentHost(MalmoPython.AgentHost):
    def __init__(self, mission_file=None, quest_file=None, actions=[], goal=None, triggers=[]):
        super().__init__()
        self.state = MalmoLogicState() if mission_file is None else getInitialWorldState(mission_file, quest_file)
        self.state.goal = goal
        self.state.actions = actions
        self.state.triggers = triggers
        self.reward = 0
        self.initialState = self.state.copy()

    def updateLogicState(self, world_state):
        self.state.updateLogicState(world_state)

    def checkGoal(self):
        return self.state.checkGoal()

    def getLogicalActions(self):
        return [action for action in self.state.all_applicable_actions()]

    def rewardValue(self):
        if self.reward != 0:
            print("LOGICAL STATE CHANGE REWARD")
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
