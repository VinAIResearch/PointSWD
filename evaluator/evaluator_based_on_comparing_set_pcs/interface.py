import abc

import six


@six.add_metaclass(abc.ABCMeta)
class Evaluator:  # (metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "evaluate") and callable(subclass.evaluate)
