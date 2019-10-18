import json
import itertools
from textworld.logic import Variable, Proposition, Predicate, Rule
from MalmoLogicState import *

class GrammarParser:
    def __init__(self, file=None):
        with open(file) as f:
            self.data = json.load(f)

    def __parseVariables(self, r):
        v = []
        for type in r.keys():
            for name in r[type]:
                v.append(Variable(name, type))
        return v

    def __parseFacts(self, r):
        f = []
        for p in r:
            temp = []
            for v in p["vars"]:
                temp.append(self.__parseVariables(v))
            for cbn in itertools.product(*temp):
                if "negate" in p.keys():
                    f.append((Proposition(p["name"], list(cbn)), not p["negate"]))
                else:
                    f.append(Proposition(p["name"], list(cbn)))
        return f

    def __parseActions(self, r):
        a = []
        for act in r:
            pre = self.__parseFacts(act["precondition"])
            post = self.__parseFacts(act["postcondition"])
            a.append(LogicalAction(act["name"], pre, post, act["command"], act["reward"]))
        return a

    def __parsePredicates(self, r):
        f = []
        for p in r:
            temp = []
            for v in p["vars"]:
                temp.append(self.__parseVariables(v))
            for cbn in itertools.product(*temp):
                if "negate" in p.keys():
                    f.append((Predicate(p["name"], list(cbn)), not p["negate"]))
                else:
                    f.append(Predicate(p["name"], list(cbn)))
        return f

    def __parseRules(self, r):
        a = []
        for act in r:
            pre = self.__parsePredicates(act["precondition"])
            post = self.__parsePredicates(act["postcondition"])
            a.append(Rule(act["name"], pre, post))
        return a

    def getVariables(self):
        return self.__parseVariables(self.data["Variables"])

    def getItems(self):
        return self.__parseVariables(self.data["Items"])

    def getDefaultFacts(self):
        return self.__parseFacts(self.data["DefaultFacts"])

    def getActions(self):
        return self.__parseActions(self.data["Actions"])

    def getTriggers(self):
        return self.__parseFacts(self.data["Triggers"])

    def getGoal(self):
        return self.__parseFacts(self.data["Goal"])

    def getPredicates(self):
        return self.__parsePredicates(self.data["Predicates"])

    def getRules(self):
        return self.__parseRules(self.data["Rules"])

if __name__ == "__main__":
    parser = GrammarParser("quest_grammar.json")
    print(parser.getRules())
