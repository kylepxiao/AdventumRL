from textworld.logic import Variable, Proposition
import MalmoPython

worldVar = Variable("world", "world")
playerVar = Variable("player", "agent")
inventoryVar = Variable("inventory", "inventory")
curInventoryVar = Variable("current", "inventory")
doorVar = Variable("door", "unlockable")
boundary1Var = Variable("boundary1", "boundary")
boundary2Var = Variable("boundary2", "boundary")
boundary3Var = Variable("boundary3", "boundary")

itemVars = {
    "apple" : Variable("apple", "item"),
    "diamond" : Variable("diamond", "item")
}

defaultFacts = { Proposition("in", [var, worldVar]) for var in itemVars.values() }.union({
    Proposition("at", [playerVar, worldVar]),
    Proposition("in", [inventoryVar, worldVar]),
    Proposition("locked", [doorVar]),
    Proposition("notreached", [boundary1Var]),
    Proposition("by", [playerVar, doorVar])
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

    def by(self, entity, thresh=20):
        if self.x1 <= entity.x and entity.x <= self.x2:
            if self.y1 <= entity.y and entity.y <= self.y2:
                if self.z1 <= entity.z and entity.z <= self.z2:
                    return False
        if self.x1 <= entity.x + thresh and entity.x <= self.x2 + thresh:
            if self.y1 <= entity.y + thresh and entity.y <= self.y2 + thresh:
                if self.z1 <= entity.z + thresh and entity.z <= self.z2 + thresh:
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

    def addPosition(self, dx, dy, dz):
        self.x += dx
        self.y += dy
        self.z += dz

    def position(self):
        return (self.x, self.y, self.z)

    def by(self, entity, thresh=20):
        if self.x <= entity.x + thresh and entity.x <= self.x + thresh:
            if self.y <= entity.y + thresh and entity.y <= self.y + thresh:
                if self.z <= entity.z + thresh and entity.z <= self.z + thresh:
                    return True
        return False

class Agent(Variable):
    def __init__(self, name, x, y, z):
        super().__init__(name, type="agent")
        self.x = int(float(x))
        self.y = int(float(y))
        self.z = int(float(z))

    def position(self):
        return (self.x, self.y, self.z)

    def setPosition(self, x, y, z):
        self.x = int(float(x))
        self.y = int(float(y))
        self.z = int(float(z))

    def addPosition(self, dx, dy, dz):
        self.x += dx
        self.y += dy
        self.z += dz

    def moveNorth(self, d=1):
        self.z += int(float(d))

    def moveSouth(self, d=1):
        self.z -= int(float(d))

    def moveEast(self, d=1):
        self.x += int(float(d))

    def moveWest(self, d=1):
        self.x -= int(float(d))

    def by(self, entity, thresh=20):
        if self.x <= entity.x + thresh and entity.x <= self.x + thresh:
            if self.y <= entity.y + thresh and entity.y <= self.y + thresh:
                if self.z <= entity.z + thresh and entity.z <= self.z + thresh:
                    return True
        return False
