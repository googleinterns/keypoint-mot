import logging
from typing import Any, List, NamedTuple, Tuple

import numpy as np
import tensorflow as tf
import torch

from model import base_model, dla


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


class PytorchToTensorflowHandlers:
    @staticmethod
    def _preprocess_tensorflow(tensorflow_weight):
        return tensorflow_weight, tensorflow_weight.name, tuple(tensorflow_weight.shape)

    @staticmethod
    def _preprocess_pytorch(pytorch_weight):
        return pytorch_weight[1], pytorch_weight[0], tuple(pytorch_weight[1].shape)

    class SameShapeHandler(WeightHandler):
        """Assigns source_weight to target_weight if the weights have same shape."""

        def _process_weights(self, source_weight, target_weight) -> WeightHandlerReturn:
            weight_tf, name_tf, shape_tf = PytorchToTensorflowHandlers._preprocess_tensorflow(target_weight)
            weight_py, name_py, shape_py = PytorchToTensorflowHandlers._preprocess_pytorch(source_weight)

            if shape_tf == shape_py:
                weight_tf.assign(weight_py.numpy())
                if self.logger:
                    self.logger.info(f'{name_tf} was assigned successfully from {name_py}')
                return WeightHandlerReturn(processed=True, matched_source=True, matched_target=True)

            return WeightHandlerReturn(processed=False, matched_source=False, matched_target=False)

    class ConvolutionHandler(WeightHandler):
        """Assigns source_weight to target_weight if the weights are 2D convolution weights."""

        def _process_weights(self, source_weight, target_weight) -> WeightHandlerReturn:
            weight_tf, name_tf, shape_tf = PytorchToTensorflowHandlers._preprocess_tensorflow(target_weight)
            weight_py, name_py, shape_py = PytorchToTensorflowHandlers._preprocess_pytorch(source_weight)

            if len(shape_tf) == len(shape_py) == 4 and shape_tf == (shape_py[2], shape_py[3], shape_py[1], shape_py[0]):
                weight_tf.assign(weight_py.numpy().transpose(2, 3, 1, 0))
                if self.logger:
                    self.logger.info(f'{name_tf} was assigned successfully from {name_py}, transposed')
                return WeightHandlerReturn(processed=True, matched_source=True, matched_target=True)

            return WeightHandlerReturn(processed=False, matched_source=False, matched_target=False)

    class DepthwiseTransposedConvolutionHandler(WeightHandler):
        """
        Assigns source_weight to target_weight if the source_weight is a depthwise transposed convolution and
        target_weight is a regular transposed convolution.
        """

        def _process_weights(self, source_weight, target_weight) -> WeightHandlerReturn:
            weight_tf, name_tf, shape_tf = PytorchToTensorflowHandlers._preprocess_tensorflow(target_weight)
            weight_py, name_py, shape_py = PytorchToTensorflowHandlers._preprocess_pytorch(source_weight)

            if len(shape_tf) == len(shape_py) == 4:
                equal_height_width = (shape_tf[:2] == shape_py[2:])
                equal_out_channels = (shape_tf[3] == shape_py[0])
                depthwise_conv_py = (shape_py[1] == 1)
                if 'conv2d_transpose' in name_tf and equal_height_width and equal_out_channels and depthwise_conv_py:
                    gen_weights = np.zeros(shape_tf, dtype=np.float32)
                    py_transposed = weight_py.numpy().transpose(2, 3, 1, 0)
                    for ch in range(shape_tf[2]):
                        gen_weights[:, :, ch, ch] = py_transposed[:, :, 0, ch]
                    weight_tf.assign(tf.convert_to_tensor(gen_weights))

                    if self.logger:
                        self.logger.info(f'{name_tf} was assigned successfully from {name_py}, depthwise convolution')
                    return WeightHandlerReturn(processed=True, matched_source=True, matched_target=True)

            return WeightHandlerReturn(processed=False, matched_source=False, matched_target=False)

    class BatchNormalizationSkipExtraHandler(WeightHandler):
        """Skips the 'num_batches_tracked' weight from pytorch BatchNorm layers."""

        def _process_weights(self, source_weight, target_weight) -> WeightHandlerReturn:
            _, name_py, _ = PytorchToTensorflowHandlers._preprocess_pytorch(source_weight)

            if 'num_batches_tracked' in name_py:
                if self.logger:
                    self.logger.info(f'skipped {name_py}')
                return WeightHandlerReturn(processed=True, matched_source=True, matched_target=False)

            return WeightHandlerReturn(processed=False, matched_source=False, matched_target=False)


class WeightsConverter:
    """
    Deals with conversion of the source model's weights to the target model's weights.
       Args:
           source_weights: List[Any] - the weights that need to be transferred
           target_weights: List[Any] - the weights which need to receive the transfer
           weight_handler: WeighthHandler - handles a specific weight pair conversion
           silent_fail: bool - if True, when a pair of weights could not be handled, no exception is thrown.
    """

    def __init__(self, source_weights: List[Any], target_weights: List[Any], weight_handler: WeightHandler,
                 silent_fail: bool = False):
        self.source_weights = source_weights
        self.target_weights = target_weights
        self.weight_handler = weight_handler
        self.silent_fail = silent_fail

    def do_conversion(self):
        """Iterates over the weight lists and does the conversion."""
        source_idx = 0
        target_idx = 0

        while target_idx < len(self.target_weights):
            source_weight = self.source_weights[source_idx]
            target_weight = self.target_weights[target_idx]
            handle_result = self.weight_handler(source_weight, target_weight)
            if handle_result.processed:
                source_idx += handle_result.matched_source
                target_idx += handle_result.matched_target
            else:
                if not self.silent_fail:
                    raise Warning(f'{target_weight} could not be matched with {source_weight}')

    def __call__(self):
        self.do_conversion()


class DLASegConverter(WeightsConverter):
    """Class used for loading pytorch weights in tensorflow for DLASeg."""

    @staticmethod
    def get_DLASeg_weights_handler() -> WeightHandler:
        """Returns the handler used in the conversion."""
        handler = PytorchToTensorflowHandlers.SameShapeHandler()
        handler.then(PytorchToTensorflowHandlers.ConvolutionHandler()).then(
                PytorchToTensorflowHandlers.DepthwiseTransposedConvolutionHandler()).then(
                PytorchToTensorflowHandlers.BatchNormalizationSkipExtraHandler())
        return handler

    @staticmethod
    def get_weights_list_pytorch(filename: str) -> List[Tuple[str, torch.Tensor]]:
        """Loads the .pth weights from the given filename, on cpu."""
        weights = torch.load(filename, map_location=torch.device('cpu'))
        weights = weights['state_dict']
        weights = list(weights.items())
        return weights

    @staticmethod
    def get_weights_list_tensorflow(model: base_model.BaseModel, batch_size: int, input_height: int,
                                    input_width: int) -> List[tf.Variable]:
        """Given a base_model.BaseModel model and the input shape, it constructs and returns the list with model's
        weights."""
        img = tf.zeros((batch_size, 3, input_height, input_width), tf.float32)
        pre_img = tf.zeros_like(img)
        pre_hm = tf.zeros((batch_size, 1, input_height, input_width), tf.float32)
        model(img, pre_img, pre_hm)  # construct the weights
        return model.weights

    @staticmethod
    def get_logger(logging_level=logging.WARNING) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging_level)
        logger.addHandler(logging.StreamHandler())
        return logger

    def __init__(self, pytorch_pth_path: str, model: dla.DLASeg, batch_size: int, input_height: int, input_width: int):
        """
        Args:
            pytorch_pth_path: str - path to .pth file holding pytorch weights (result of a .state_dict() call)
            model: dla.DLASeg - tensorflow model instance
            batch_size, input_height, input_width: int - input characteristics for the tensorflow model
        """
        pytorch_weights = self.get_weights_list_pytorch(pytorch_pth_path)
        tensorflow_weights = self.get_weights_list_tensorflow(model, batch_size, input_height, input_width)
        handler = self.get_DLASeg_weights_handler()
        handler.set_chain_logger(self.get_logger(logging.INFO))
        super().__init__(source_weights=pytorch_weights, target_weights=tensorflow_weights, weight_handler=handler,
                         silent_fail=False)
