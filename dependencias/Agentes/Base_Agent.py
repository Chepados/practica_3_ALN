from abc import ABC, abstractmethod

class Agent(ABC):
    ## Abstract class for the agent

    @abstractmethod
    def get_action(self, state):
        pass