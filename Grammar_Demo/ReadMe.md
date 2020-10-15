# AdventumRL 

To get started, view grammar_demo folder

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
   1. [Installation](#installation)
   2. [Running Missions](#running-missions)
3. [Additional Information](#additional-information)
   1. [Agents](#agents)
   2. [Grammar](#grammar)
   3. [Mission & Quest Files](#mission--quest-files)
   4. [Grammar Schema](#grammar-schema)
   5. [Quest Schema](#quest-schema)

## Overview

AdventumRL is an API framework that allows for complex, mission-based reinforcement learning in Minecraft. It builds upon of [Malmo](https://github.com/microsoft/malmo), a reinforcement learning platform that hooks into Minecraft (Java Edition). AdventumRL implements support for propositional logic in a high-level configuration space via [Textworld grammar](https://github.com/microsoft/TextWorld), and includes sample agents that make use of this grammar in an example cliff-walking mission.

## Getting Started

### Installation

For Mac or Linux:
1. Download the latest pre-built release of [Malmo]( https://github.com/Microsoft/malmo/releases )
2. Install the Malmo dependencies for your OS: [Linux](https://github.com/microsoft/malmo/blob/master/doc/install_linux.md), [Mac](https://github.com/microsoft/malmo/blob/master/doc/install_macosx.md) (Make sure to install the optional python modules as well)
3. Download the latest release of [AdventumRL ](https://github.gatech.edu/kxiao36/Malmo_Grammar) and place the files into the same folder as the Malmo release from before
4. Install the following pip modules: Textworld, Nose, PyTorch, and TKinter
   - Linux
     - `sudo pip3 install nose torch textworld==1.1.0 matplotlib`
     - `sudo apt-get install python3-tk`
5. In one terminal, go to the `Minecraft` folder and launch the client
   - `sudo ./launchClient.sh`
6. In a second terminal, go to the `Grammar_Demo` folder and run the `grammar_api.py`  file
   - `python3 grammar_api.py`
   - For more information about running missions, look at the Running Missions section


For Windows:
1. Download the latest pre-built release of [Malmo]( https://github.com/Microsoft/malmo/releases ) and download the Windows dependencies [here](https://github.com/microsoft/malmo/blob/master/doc/install_windows.md)
    a. When adding MALMO_XSD_PATH to environment variables, make sure to add them to your PATH
2. Download the latest release of [AdventumRL](https://github.gatech.edu/kxiao36/Malmo_Grammar) and place the files into the same folder as the Malmo release from before
3. Download and setup [WSL] (https://www.microsoft.com/en-us/p/ubuntu-2004-lts/9n6svws3rx71?activetab=pivot:overviewtab) (Please download Ubuntu version 18.04 or above)
4. Install the following pip modules: Textworld, Nose, PyTorch, and TKinter
     - `sudo pip3 install nose torch future pillow textworld==1.1.0 matplotlib`
     - `sudo apt-get install python3-tk`
     - `sudo update-ca-certificates -f`
5. Setup [MobaXTerm] (https://nickjanetakis.com/blog/using-wsl-and-mobaxterm-to-create-a-linux-dev-environment-on-windows)
     - Make sure to add the display port on wsl based on your version of wsl
6. Open Powershell and go to the `Minecraft` folder to launch the client 
     - `sudo ./launchClient.sh`
7. In MobaXTerm, go to the `Grammar_Demo` folder and run the `grammar_api.py`  file
     - `python3 grammar_api.py`
### Running Missions

When running a mission, you must specify a mission file, quest file, grammar file, and an agent. By default, the grammar_api will use the default files and TabQAgent for a sample cliff-walking exercise. 

## Additional Information

### Agents

Currently, AdventumRL includes two prebuilt agents, a TabQAgent (`TabQAgent.py`) and a DqnAgent (`DQNAgent.py` ), which can be run on missions. These agents can be modified, and additional agents can be created by referencing files in the `Grammar_Demo/models` folder. The `Agent.py` superclass provides guidelines for methods a new agent might potentially need. 

The DqnAgent is designed to work on both CPU and GPU, but the trained models that come with AdventumRL are set to work with a GPU specifically. 

### Grammar

AdventumRL supports the following logical relations. Additional grammatical constructs can be defined by the player in the quest_grammar.json file. Triggers, facts observable by the general state space, can also be defined. 

- **in** - Whether entity A's coordinates are contained within entity B's coordinates (*Coord(A)*  ⊂  *Coord(B)*)
- **at** - Whether entity A's coordinates overlap with entity B's coordinates (*Coord(A)*  ∩  *Coord(B)* != 0)
- **by** - Whether entity A's coordinates are are close enough to entity B's coordinates so entity A can interact with entity B in the Minecraft world (∃ δ <  ε s.t. ((*Coord(A)* *+ δ*) ∩ *Coord(B)* != 0)
- **unlocked** - A theme specific attribute relation for unlockable items
- **inhand** - Whether an entity is currently selected by the player and is usable in the world 

### Mission & Quest Files

The quest file, `quest_entities.xml` can be used to define physical entities in the Minecraft world, including constructs like bounding boxes, which can be particularly helpful when creating a new mission. 

The mission file utilizes Malmo's specifications, for which more detail can be found in their [official documentation]( https://microsoft.github.io/malmo/0.17.0/Schemas/Mission.html). 

### Grammar Schema

```json
{
    "Variables": {
        "Type of the variable": ["Name of the variables under type"],
    },
    "Items": {
        "Type of the item": ["Name of the items under type"]
    },
    "DefaultFacts": [
        {
            "name": "Name of the fact",
            "vars": [
                {"Type of the variable": ["Name of the variable"]},
                {"Can have arbitrary number of variable types in relation": ["Arbitrary number of variable names"]}
            ]
        }
    ],
    "Actions": [
        {
            "name": "Name of action",
            "precondition": [
                {
                    "name": "Name of the fact",
                    "vars": [
                        {"Type of the variable": ["Name of the variable"]},
                    ]
                },
                {
                    "name": "Can have arbitrary number of facts",
                    "vars": [
                        {"Type of the variable": ["Name of the variable"]},
                    ]
                }
            ],
            "postcondition": [
                {
                    "name": "Name of the fact",
                    "vars": [
                        {"Type of the variable": ["Name of the variable"]},
                    ]
                },
                {
                    "name": "Can have arbitrary number of facts",
                    "vars": [
                        {"Type of the variable": ["Name of the variable"]},
                    ]
                }
            ],
            "command": "Corresponding malmo command",
            "reward": 0
        }
    ],
    "Goal": [
        {
            "name": "Name of the fact",
            "vars": [
                {"Type of the variable": ["Name of the variable"]},
            ],
            "negate": False
        },
        {
            "name": "Can have arbitrary number of facts",
            "vars": [
                {"Type of the variable": ["Name of the variable"]},
            ]
        }
    ],
    "Triggers": [
        {
            "name": "Name of the fact",
            "vars": [
                {"Type of the variable": ["Name of the variable"]},
            ],
            "negate": False
        },
        {
            "name": "Can have arbitrary number of facts",
            "vars": [
                {"Type of the variable": ["Name of the variable"]},
            ]
        }
    ],
    "Predicates": [
        {
            "name": "Name of predicate",
            "vars": [
                {"Type of the variable": ["Name of the variable"]},
                {"Can have arbitrary number of variable types in relation": ["Arbitrary number of variable names"]}
            ]
        }
    ],
    "Rules": [
        "name": "Name of rule",
            "precondition": [
                {
                    "name": "Name of the predicate",
                    "vars": [
                        {"Type of the variable": ["Name of the variable"]},
                    ]
                },
                {
                    "name": "Can have arbitrary number of predicates",
                    "vars": [
                        {"Type of the variable": ["Name of the variable"]},
                    ]
                }
            ],
            "postcondition": [
                {
                    "name": "Name of the predicate",
                    "vars": [
                        {"Type of the variable": ["Name of the variable"]},
                    ]
                },
                {
                    "name": "Can have arbitrary number of predicates",
                    "vars": [
                        {"Type of the variable": ["Name of the variable"]},
                    ]
                }
            ],
            "command": "Corresponding malmo command",
            "reward": 0
        }
    ]
}
```

### Quest Schema
```xml
<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Quest xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <Grid name="Entity Name Here">
      <min x="min_x" y="min_y" z="min_z"/>
      <max x="max_x" y="max_y" z="max_z"/>
    </Grid>
    ...
</Quest>
```