import random
from itertools import product
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import BaseScheduler
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer


class ConwayAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.die = 0

    def step(self):
        pass


class ConveyModel(Model):
    """A model with some number of agents."""

    def __init__(self, N, width, height):
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = BaseScheduler(self)

        # Create agents
        for i in range(self.num_agents):
            a = ConwayAgent(i, self)
            # Add the agent to a random grid cell
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            while(len(self.grid.get_cell_list_contents((x, y)))):
                x = random.randrange(self.grid.width)
                y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            self.schedule.add(a)
        self.i = i

        self.datacollector = DataCollector(
            agent_reporters={"State": lambda a: a.die})

    def step(self):
        self.datacollector.collect(self)
        new_agents = []
        for (x, y) in product(range(self.grid.width), range(self.grid.height)):
            ns = self.grid.iter_neighbors((x, y), True)
            neighbors = 0
            for n in ns:
                if(n):
                    neighbors += 1
            if(self.grid[x][y]):  # live cell
                if(neighbors < 2):  # underpopulation
                    list(self.grid[x][y])[0].die = 1
                elif(neighbors > 3):  # overpopulation
                    list(self.grid[x][y])[0].die = 1
            else:  # dead cell
                if(neighbors == 3):
                    new_agents.append((x, y))
        for (x, y) in product(range(self.grid.width), range(self.grid.height)):
            if self.grid[x][y]:
                a = list(self.grid[x][y])[0]
                if a.die:
                    self.grid.remove_agent(a)
                    self.schedule.remove(a)
        for na in new_agents:
            self.i += 1
            a = ConwayAgent(self.i, self)
            self.grid.place_agent(a, na)
            self.schedule.add(a)


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "red",
                 "r": 0.5}
    return portrayal


grid = CanvasGrid(agent_portrayal, 100, 100, 500, 500)
server = ModularServer(ConveyModel,
                       [grid],
                       "Convey Model",
                       {"N": 2000, "width": 100, "height": 100})

server.port = 8521  # The default
server.launch()
