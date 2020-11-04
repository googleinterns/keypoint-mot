import logging
from unittest import TestCase

from model import weights_converter


class TestWeightHandler(TestCase):

    def test_then(self):
        current_handler = weights_converter.WeightHandler()
        next_handler = weights_converter.WeightHandler()
        result_handler = current_handler.then(next_handler)

        self.assertIs(next_handler, result_handler)
        self.assertIs(current_handler.next_handler, next_handler)

    def test_set_chain_logger(self):
        handlers = [weights_converter.WeightHandler() for i in range(5)]
        for i in range(0, len(handlers) - 1):
            handlers[i].then(handlers[i + 1])

        logger = logging.getLogger(__name__)
        handlers[0].set_chain_logger(logger=logger)

        for handler in handlers:
            self.assertIs(handler.logger, logger)

    def test_convert_weight(self):
        handlers = [weights_converter.WeightHandler() for i in range(5)]
        not_processed_return = weights_converter.WeightHandlerReturn(processed=False, matched_source=False,
                                                                     matched_target=False)
        processed_return = weights_converter.WeightHandlerReturn(processed=True, matched_source=True,
                                                                 matched_target=False)

        for i in range(0, len(handlers) - 1):
            handlers[i].then(handlers[i + 1])
            handlers[i]._process_weights = lambda a, b: not_processed_return

        handlers[-1]._process_weights = lambda a, b: processed_return

        result = handlers[0].convert_weight(None, None)
        self.assertEqual(result, processed_return)
