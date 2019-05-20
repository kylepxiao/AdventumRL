from textworld.logic import Variable, Proposition
import MalmoPython

worldVar = Variable("world", "world")
playerVar = Variable("player", "agent")
inventoryVar = Variable("inventory", "inventory")

itemVars = {
    "apple" : Variable("apple", "item"),
    "diamond" : Variable("diamond", "item")
}

defaultFacts = { Proposition("in", [var, worldVar]) for var in itemVars.values() }.union({
    Proposition("at", [playerVar, worldVar]),
    Proposition("in", [inventoryVar, worldVar]),
    })

class Boundary(Variable):
    def __init__(self, name, pmin, pmax):
        super().__init__(name, type="boundary")
        self.x1 = float(pmin[0])
        self.y1 = float(pmin[1])
        self.z1 = float(pmin[2])
        self.x2 = float(pmax[0])
        self.y2 = float(pmax[1])
        self.z2 = float(pmax[2])

    def position(self):
        return ((self.x1, self.y1, self.z1), (self.x2, self.y2, self.z2))

    def setPosition(self, x1, y1, z1, x2, y2, z2):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.z1 = float(z1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.z2 = float(z2)

    def contains(self, entity):
        if self.x1 <= entity.x and entity.x <= self.x2:
            if self.y1 <= entity.y and entity.y <= self.y2:
                if self.z1 <= entity.z and entity.z <= self.z2:
                    return True
        return False

class Item(Variable):
    def __init__(self, name, x, y, z):
        super().__init__(name, type="item")
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def setPosition(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def position(self):
        return (self.x, self.y, self.z)

class Agent(Variable):
    def __init__(self, name, x, y, z):
        super().__init__(name, type="agent")
        self.x = x
        self.y = y
        self.z = z
