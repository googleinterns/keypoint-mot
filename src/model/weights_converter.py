import logging
from typing import NamedTuple


class WeightHandlerReturn(NamedTuple):
    """
    Return data class for WeightHandler.
    processed - True if the weight pair was successfully handled
    matched_source - True if the source weight was processed
    matched_target - True if the target weight was processed
    """
    processed: bool
    matched_source: bool
    matched_target: bool


class WeightHandler:
    """
    Generic class that deals with processing of a specific pair of weights and chains the operations.
    For subclassing, _process_weights method must be implemented.
    """

    def __init__(self, logger: logging.Logger = None):
        self.next_handler = None
        self.logger = logger

    def then(self, next_handler):
        """next_handler: WeightHandler - the next operation to be called if the current handler can not resolve the
        matching"""
        self.next_handler = next_handler
        return next_handler

    def set_chain_logger(self, logger: logging.Logger):
        """logger: logging.Logger - the logger to be set on all the chained operations"""
        self.logger = logger
        if self.next_handler is not None:
            self.next_handler.set_chain_logger(logger)

    def convert_weight(self, source_weight, target_weight) -> WeightHandlerReturn:
        """
        Tries to assign the source_weight to target_weight (i.e. target_weight = source_weight).
        If the assignment is not possible, the next handler in chain is called and the result is returned.

        Args:
            source_weight: Any
            target_weight: Any
        Returns:
            WeightHandlerReturn
        """

        result = self._process_weights(source_weight, target_weight)
        if not result.processed and self.next_handler is not None:
            return self.next_handler.convert_weight(source_weight, target_weight)
        return result

    def __call__(self, source_weight, target_weight):
        return self.convert_weight(source_weight, target_weight)

    def _process_weights(self, source_weight, target_weight) -> WeightHandlerReturn:
        """Should provide the actual processing logic for the given weight pair"""
        raise NotImplementedError
