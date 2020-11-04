import logging
from unittest import TestCase

import numpy as np
import tensorflow as tf
import torch

from model import weights_converter

torch.set_grad_enabled(False)


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


class TestPytorchToTensorflowHandlers(TestCase):
    def test__preprocess_tensorflow(self):
        name = 'a_random_name'
        shape = (1, 1)
        var = tf.Variable(name=name, shape=shape, initial_value=[[0]])
        res_tensor, res_name, res_shape = weights_converter.PytorchToTensorflowHandlers._preprocess_tensorflow(var)
        self.assertIs(var, res_tensor)
        self.assertEqual(f'{name}:0', res_name)
        self.assertEqual(shape, res_shape)

    def test__preprocess_pytorch(self):
        layer = torch.nn.Linear(1, 1, bias=False)
        var = list(layer.state_dict().items())[0]
        res_tensor, res_name, res_shape = weights_converter.PytorchToTensorflowHandlers._preprocess_pytorch(var)
        self.assertIs(var[1], res_tensor)
        self.assertIs(res_name, 'weight')
        self.assertEqual(res_shape, (1, 1))


class TestSameShapeHandler(TestCase):
    def test(self):
        inp = np.array([[1, 2, 3]], dtype=np.float32)
        model_tensorflow = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.001)
        model_tensorflow.trainable = False
        model_tensorflow(inp)

        model_pytorch = torch.nn.BatchNorm1d(3, momentum=0.1, eps=0.001)
        model_pytorch.train(False)
        model_pytorch.weight.uniform_()

        converter = weights_converter.PytorchToTensorflowHandlers.SameShapeHandler()
        res_weight = converter(('', model_pytorch.state_dict()['weight']), model_tensorflow.weights[0])
        res_bias = converter(('', model_pytorch.state_dict()['bias']), model_tensorflow.weights[1])

        np.testing.assert_allclose(model_tensorflow(inp).numpy(), model_pytorch(torch.tensor(inp)).numpy(), rtol=1e-5)
        expected_res = weights_converter.WeightHandlerReturn(processed=True, matched_source=True, matched_target=True)
        self.assertEqual(res_weight, expected_res)
        self.assertEqual(res_bias, expected_res)


class TestConvolutionHandler(TestCase):
    def test(self):
        in_channels = 3
        kernel_size = 3
        out_channels = 5
        inp = np.ones((1, in_channels, 15, 15), dtype=np.float32)

        model_tensorflow = tf.keras.layers.Conv2D(out_channels, kernel_size, use_bias=False,
                                                  data_format='channels_first')
        model_tensorflow(inp)

        model_pytorch = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        model_pytorch.weight.uniform_()

        converter = weights_converter.PytorchToTensorflowHandlers.ConvolutionHandler()
        res = converter(('', model_pytorch.state_dict()['weight']), model_tensorflow.weights[0])

        np.testing.assert_allclose(model_tensorflow(inp).numpy(), model_pytorch(torch.tensor(inp)).numpy(), rtol=1e-5)
        expected_res = weights_converter.WeightHandlerReturn(processed=True, matched_source=True, matched_target=True)
        self.assertEqual(res, expected_res)


class TestDepthwiseTransposedConvolutionHandler(TestCase):
    def test(self):
        in_channels = 3
        kernel_size = 3
        out_channels = in_channels
        inp = np.ones((1, in_channels, 15, 15), dtype=np.float32)

        model_tensorflow = tf.keras.layers.Conv2DTranspose(out_channels, kernel_size, use_bias=False,
                                                           data_format='channels_first')
        model_tensorflow(inp)

        model_pytorch = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, bias=False,
                                                 groups=out_channels)
        model_pytorch.weight.uniform_()

        converter = weights_converter.PytorchToTensorflowHandlers.DepthwiseTransposedConvolutionHandler()
        res = converter(('', model_pytorch.state_dict()['weight']), model_tensorflow.weights[0])

        np.testing.assert_allclose(model_tensorflow(inp).numpy(), model_pytorch(torch.tensor(inp)).numpy(), rtol=1e-5)
        expected_res = weights_converter.WeightHandlerReturn(processed=True, matched_source=True, matched_target=True)
        self.assertEqual(res, expected_res)


class TestBatchNormalizationSkipExtraHandler(TestCase):
    def test(self):
        model_pytorch = torch.nn.BatchNorm1d(3, momentum=0.1, eps=0.001)

        converter = weights_converter.PytorchToTensorflowHandlers.BatchNormalizationSkipExtraHandler()
        res = converter(list(model_pytorch.state_dict().items())[-1], None)

        expected_res = weights_converter.WeightHandlerReturn(processed=True, matched_source=True, matched_target=False)
        self.assertEqual(res, expected_res)
