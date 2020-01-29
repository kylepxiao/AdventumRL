import matplotlib.pyplot as plt
import csv

x = []
y = []
axes = plt.gca()
axes.set_ylim([-200,350])

with open('logs/CameraDQNAgent-20200128-200911.txt','r') as csvfile:
    plots = csvfile.readlines()
    for i, row in enumerate(plots):
        x.append(i)
        y.append(float(row))

plt.plot(x[:5000],y[:5000], label='Rewards')
plt.xlabel('Iterations')
plt.ylabel('Cumulative Rewards Per Run')
plt.title('Camera-based CNN + DQN Training on Sample Mission with Triggers')
plt.legend()
plt.show()

x = []
y = []
axes = plt.gca()
axes.set_ylim([-200,350])

with open('logs/CameraDQNAgent-20200127-143808.txt','r') as csvfile:
    plots = csvfile.readlines()
    for i, row in enumerate(plots):
        x.append(i)
        y.append(float(row))

plt.plot(x[:5000],y[:5000], label='Rewards')
plt.xlabel('Iterations')
plt.ylabel('Cumulative Rewards Per Run')
plt.title('Camera-based CNN + DQN Training on Sample Mission without Triggers')
plt.legend()
plt.show()

x = []
y = []
axes = plt.gca()
axes.set_ylim([-200,350])

with open('logs/DQNAgent-20200118-132104.txt','r') as csvfile:
    plots = csvfile.readlines()
    for i, row in enumerate(plots):
        x.append(i)
        y.append(float(row))

plt.plot(x[:5000],y[:5000], label='Rewards')
plt.xlabel('Iterations')
plt.ylabel('Cumulative Rewards Per Run')
plt.title('Shallow DQN Training on Sample Mission without Triggers')
plt.legend()
plt.show()

x = []
y = []
axes = plt.gca()
axes.set_ylim([-200,350])

with open('logs/DQNAgent-20200117-162507.txt','r') as csvfile:
    plots = csvfile.readlines()
    for i, row in enumerate(plots):
        x.append(i)
        y.append(float(row))

plt.plot(x[:5000],y[:5000], label='Rewards')
plt.xlabel('Iterations')
plt.ylabel('Cumulative Rewards Per Run')
plt.title('DQN Training on Sample Mission without Triggers')
plt.legend()
plt.show()

x = []
y = []
axes = plt.gca()
axes.set_ylim([-200,350])

with open('logs/DQNAgent-20200116-182014.txt','r') as csvfile:
    plots = csvfile.readlines()
    for i, row in enumerate(plots):
        x.append(i)
        y.append(float(row))

plt.plot(x[:5000],y[14000:19000], label='Rewards')
plt.xlabel('Iterations')
plt.ylabel('Cumulative Rewards Per Run')
plt.title('DQN Training on Sample Mission using Propositions/Triggers')
plt.legend()
plt.show()

x = []
y = []
axes = plt.gca()
axes.set_ylim([-200,350])

with open('logs/TabQAgent-20200117-160413.txt','r') as csvfile:
    plots = csvfile.readlines()
    for i, row in enumerate(plots):
        x.append(i)
        y.append(float(row))

plt.plot(x[:200],y[:200], label='Rewards')
plt.xlabel('Iterations')
plt.ylabel('Cumulative Rewards Per Run')
plt.title('TabQ training on Sample Mission using Propositions/Triggers')
plt.legend()
plt.show()

x = []
y = []
axes = plt.gca()
axes.set_ylim([-200,350])

with open('logs/TabQAgent-20200123-172338.txt','r') as csvfile:
    plots = csvfile.readlines()
    for i, row in enumerate(plots):
        x.append(i)
        y.append(float(row))

plt.plot(x[:200],y[:200], label='Rewards')
plt.xlabel('Iterations')
plt.ylabel('Cumulative Rewards Per Run')
plt.title('TabQ training on Sample Mission without Triggers')
plt.legend()
plt.show()
