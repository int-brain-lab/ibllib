import abc


class OneAbstract(abc.ABC):

    @abc.abstractmethod
    def load(self, eid, **kwargs):
        return

    @abc.abstractmethod
    def list(self, **kwargs):
        return

    @abc.abstractmethod
    def search(self, **kwargs):
        return
