from abc import abstractmethod


class ContextualBandit:
    def __init__(self, context_dimension):
        self.context_dimension = context_dimension

    @abstractmethod
    def predict(self, context):
        """
        Predict next action given observation in terms of context.
        :param context: context vector which defines observation
        :return: next action
        """
        pass

    @abstractmethod
    def update(self, action, context, reward):
        """

        :param action: performed action
        :param context: context vector which defines observation
        :param reward: reward given action
        """
        pass
