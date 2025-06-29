class Agent:
    def __init__(self, position, agent_id):
        self.position = position
        self.energy = 0
        self.cooldown = 0
        self.agent_id = agent_id
        self.rest_turns = 0

    def decide_action(self, world):
        raise NotImplementedError("Subclasses must implement this")

    def __str__(self):
        return f"{self.__class__.__name__}#{self.agent_id} at {self.position}"