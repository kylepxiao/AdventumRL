# Malmo + Grammar

[TOC]

## Overview

Malmo + Grammar is an extension to extend the scope of [Malmo](https://github.com/microsoft/malmo), a reinforcement learning platform that hooks into Minecraft (Java Edition). 

## Getting Started

### Installation

1. Download the latest pre-built release of [Malmo]( https://github.com/Microsoft/malmo/releases )
2. Install the Malmo dependencies for your OS: [Windows](https://github.com/microsoft/malmo/blob/master/doc/install_windows.md), [Linux](https://github.com/microsoft/malmo/blob/master/doc/install_linux.md), [Mac](https://github.com/microsoft/malmo/blob/master/doc/install_macosx.md) (Make sure to install the optional python modules as well)
3. Download the latest release of [Malmo+Grammar](https://github.gatech.edu/kxiao36/Malmo_Grammar) and place the files into the same folder as the Malmo release from before
4. Install the following pip modules: Textworld, Nose, PyTorch, and TKinter
   - Linux
     - `sudo pip3 install nose textworld torch`
     - `sudo apt-get install python3-tk`
5. In one terminal, go to the `Minecraft` folder and launch the client
   - `sudo ./launchClient.sh`
6. In a second terminal, go to the `Grammar_Demo` folder and run the `grammar_api.py`  file
   - `python3 grammar_api.py`
   - For more information about running missions, look at the Running Missions section

### Running Missions

When running a mission, you must specify a mission file, quest file, grammar file, and an agent. By default, the grammar_api will use the default files and TabQAgent for a sample cliff-walking exercise. 

## Additional Information

### Grammar

Malmo + Grammar supports the use of [Textworld grammar](https://textworld.readthedocs.io/en/latest/textworld.generator.grammar.html) to add additional considerations and information to the mission. 

### Mission & Quest Files