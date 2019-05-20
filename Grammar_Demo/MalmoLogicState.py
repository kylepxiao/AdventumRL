import json
import MalmoPython
from textworld.logic import State, Variable, Proposition
from xml.etree import ElementTree as ET
from parse_mission_xml import namespace
from constants import *

class MalmoLogicState(State):
    def __init__(self, facts=None, entities={}, boundaries={}, goal=None):
        super().__init__(facts=facts)
        self.entities = entities
        self.boundaries = boundaries
        self.goal = goal

    def updateLogicState(self, world_state):
        if len(world_state.observations) < 1:
            return

        observation = json.loads(world_state.observations[-1].text)

        # Clear all inventory propositions in world state
        oldInventoryProps = set()
        for fact in self.facts:
            if fact.names[-1] == "inventory":
                oldInventoryProps.add(fact)
        self.remove_facts(oldInventoryProps)

        # Add all inventory propositions to world state
        inventoryVar = Variable("inventory", "inventory")
        inventory = observation['inventory']
        for item in inventory:
            key = item['type']
            self.add_fact( Proposition("in", [self.entities[key], inventoryVar]) )
            self.entities[key].setPosition(observation['XPos'], observation['YPos'], observation['ZPos'])

            # Check if items in inventory leave boundaries
            for boundary in self.boundaries:
                if self.entities[boundary].contains(self.entities[key]):
                    self.add_fact( Proposition("in", [self.entities[key], self.entities[boundary]]) )
                else:
                    self.remove_fact( Proposition("in", [self.entities[key], self.entities[boundary]]) )

    def checkGoal(self):
        return self.is_fact(self.goal)

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
                # start each object with the proposition that it is in the current world
                # objectProps[item.attrib['type']] = Proposition("in", [currentVar, worldVar])
                # objectVars[item.attrib['type']]= currentVar

        state_list = state_list.union( props )

    return MalmoLogicState(state_list, entities, boundaries)

class LogicalAgentHost(MalmoPython.AgentHost):
    def __init__(self, mission_file=None, quest_file=None, goal=None):
        super().__init__()
        self.state = MalmoLogicState() if mission_file is None else getInitialWorldState(mission_file, quest_file)
        self.state.goal = goal

    def updateLogicState(self, world_state):
        self.state.updateLogicState(world_state)

    def sendCommand(self, command):
        super().sendCommand(command)
