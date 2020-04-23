# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from unittest import SkipTest

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as np
from jax import test_util as jtu
from jax import core
from jax import lax
from jax import random
from jax import tree_util
from jax.api import (pmap, soft_pmap, jit, vmap, jvp, vjp, grad, make_jaxpr,
                     linearize, device_put)
from jax.lib import xla_bridge as xb
from jax.lib import xla_client
from jax.util import prod
from jax.interpreters import pxla
from jax.interpreters import xla
from jax.interpreters.sharded_jit import sharded_jit, set_sharding
from jax.interpreters.sharded_jit import PartitionSpec as P

from jax.config import config
config.parse_flags_with_absl()


class ShardedJitTest(jtu.JaxTestCase):

  def testBasic(self):
    @partial(sharded_jit, partitions=((P(2, 1), P(2, 1)), None))
    def f(x, y):
      return x + y

    shape = (8, 8)
    x = onp.arange(onp.prod(shape), dtype=np.float32).reshape(shape)
    actual = f(x, x + 1)
    expected = x + (x + 1)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, pxla.ShardedDeviceArray)
    self.assertLen(actual.device_buffers, 2)
    self.assertAllClose(actual.device_buffers[0].to_py(), expected,
                        check_dtypes=False)

  def testNested(self):
    shape = (4, 4)

    @partial(sharded_jit, partitions=(P(4,1), None))
    def f(x):
      y = x + 1
      z = sharded_jit(lambda x, y: lax.dot(x, y),
                      partitions=(P(2,2), P(2,2)))(y, y)
      return z + 1

    x = onp.arange(onp.prod(shape), dtype=np.float32).reshape(shape)
    expected = lax.dot(x + 1, x + 1) + 1
    actual = f(x)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, pxla.ShardedDeviceArray)
    self.assertLen(actual.device_buffers, 4)
    self.assertAllClose(actual.device_buffers[0].to_py(), expected,
                        check_dtypes=False)

    hlo = jax.xla_computation(f)(x)
    self.assertIn("sharding={devices=[4,1]0,1,2,3}", hlo.as_hlo_text())

  def testGradOfNested(self):
    @partial(sharded_jit, partitions=(P(4,1), None))
    def f(x):
      y = x + 1
      p, vjp_f = vjp(lambda z: np.sin(set_sharding(z, P(2,2))), y)
      return vjp_f(p)

    def expected_f(x):
      y = x + 1
      p, vjp_f = vjp(lambda z: np.sin(z), y)
      return vjp_f(p)

    shape = (4, 4)
    x = onp.arange(onp.prod(shape), dtype=np.float32).reshape(shape)
    actual = f(x)
    expected = expected_f(x)
    self.assertAllClose(actual, expected, check_dtypes=False)


class PmapOfShardedJitTest(jtu.JaxTestCase):

  def _test(self, f, in_partitions, out_partitions, dtype=np.float32):
    shape = (1, 4, 4)
    num_shards = shape[0] * onp.prod(in_partitions[0])
    if num_shards > xb.local_device_count():
      raise SkipTest("requires %d devices" % num_shards)

    x = onp.arange(onp.prod(shape, dtype=dtype)).reshape(shape)
    y = x + 1
    result = pmap(
        sharded_jit(f, partitions=(in_partitions, out_partitions)))(x, y)
    expected = pmap(f)(x, y)
    self.assertAllClose(result, expected, check_dtypes=False)

    flat_result = tree_util.tree_flatten(result)[0]
    for r in flat_result:
      self.assertTrue(isinstance(r, pxla.ShardedDeviceArray))
      self.assertEqual(len(r.device_buffers), num_shards)

  @parameterized.named_parameters({
      "testcase_name":
          "_in_parts={}_out_parts={}".format(in_partitions,
                                             out_partitions).replace(" ", ""),
      "in_partitions":
          in_partitions,
      "out_partitions":
          out_partitions
  } for in_partitions in [
      (P(2, 1), P(2, 1)),
      (P(2, 1), P(1, 2)),  # TODO(skye): fix layout issue
      (P(2, 2), P(2, 2)),
      (P(4, 1), P(2, 2)),
  ] for out_partitions in [in_partitions[0], None])
  def testBasic(self, in_partitions, out_partitions):

    def f(x, y):
      return lax.dot(x, y)

    self._test(f, in_partitions, out_partitions)

  @parameterized.named_parameters({
      "testcase_name":
          "_in_parts={}_out_parts={}".format(in_partitions,
                                             out_partitions).replace(" ", ""),
      "in_partitions":
          in_partitions,
      "out_partitions":
          out_partitions
  } for in_partitions in [
      (P(2, 1), P(2, 1)),
      (P(2, 1), P(1, 2)),  # TODO(skye): fix layout issue
      (P(4, 1), P(2, 2))
  ] for out_partitions in [(in_partitions[1], in_partitions[0],
                            None), (None, None, None)])
  def testMultipleOutputs(self, in_partitions, out_partitions):

    def f(x, y):
      a = lax.dot(x, y)
      return a, a + 1, 3

    with self.assertRaises(NotImplementedError):
      self._test(f, in_partitions, out_partitions)

  @parameterized.named_parameters({
      "testcase_name":
          "_in_parts={}_out_parts={}".format(in_partitions,
                                             out_partitions).replace(" ", ""),
      "in_partitions":
          in_partitions,
      "out_partitions":
          out_partitions
  } for in_partitions in [
      (P(2, 1), P(2, 1)),
      (P(2, 1), P(1, 2)),  # TODO(skye): fix layout issue
      (P(4, 1), P(2, 2))
  ] for out_partitions in [in_partitions[0], None])
  def testArrayConstants(self, in_partitions, out_partitions):

    def f(x, y):
      a = lax.dot(x, y)
      b = a + np.ones(a.shape)
      c = b + np.ones(a.shape[0])
      return c

    self._test(f, in_partitions, out_partitions)

  def testPyTreeArgs(self):
    if xb.local_device_count() < 4:
      raise SkipTest("requires 4 devices")

    def f(a, b, c):
      a1, a2 = a
      c1, (c2, c3) = c
      return a1 + a2 + b + c1 + c2 + c3

    def _make_arg(*shape):
      return onp.arange(onp.prod(shape)).reshape(shape)

    a = (_make_arg(2, 4, 4), _make_arg(2))
    b = _make_arg(2, 4, 4)
    c = (_make_arg(2), (_make_arg(2, 4, 4), _make_arg(2, 4, 4)))

    in_parts = (None, P(2, 1), (None, P(2, 1)))
    out_parts = P(2, 1)

    result = pmap(sharded_jit(f, partitions=(in_parts, out_parts)))(a, b, c)
    expected = pmap(f)(a, b, c)

    self.assertAllClose(result, expected, check_dtypes=False)
    self.assertTrue(isinstance(result, pxla.ShardedDeviceArray))
    self.assertEqual(len(result.device_buffers), 4)

  def testPyTreeOutputs(self):
    if xb.local_device_count() < 4:
      raise SkipTest("requires 4 devices")

    def f(x):
      return x + 1, ((x + 2, x + 3), x + 4)

    shape = (2, 4, 4)
    x = onp.arange(onp.prod(shape)).reshape(shape)
    in_parts = (P(2, 1),)
    out_parts = (P(2, 1), ((P(1, 2), None), P(2, 1)))

    result = pmap(sharded_jit(f, partitions=(in_parts, out_parts)))(x)
    expected = pmap(f)(x)

    self.assertAllClose(result, expected, check_dtypes=False)

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testManyArgs(self):
    if xb.local_device_count() < 4:
      raise SkipTest("requires 4 devices")

    num_args = 200

    def f(*args):
      return np.sum(args)

    shape = (2, 4, 4)
    args = [onp.arange(onp.prod(shape)).reshape(shape)] * num_args
    in_partitions = (P(2, 1),) * num_args
    out_partitions = None
    result = pmap(sharded_jit(f, partitions=(in_partitions,
                                             out_partitions)))(*args)
    expected = pmap(f)(*args)

    self.assertAllClose(result, expected, check_dtypes=False)
    self.assertTrue(isinstance(result, pxla.ShardedDeviceArray))
    self.assertEqual(len(result.device_buffers), 4)

  def testInfeed(self):
    if xb.local_device_count() < 2:
      raise SkipTest("requires 2 devices")

    shape = (jax.device_count() // 2, 4, 4)
    infeed_shape = jax.ShapedArray(shape[1:], np.float32)
    # None for token
    # TODO(skye): maybe replicate the token automatically
    infeed_parts = (P(2, 1), None)

    def f(x):
      token = lax.create_token(x)
      (y,), token = lax.infeed(token, (infeed_shape,), sharding=infeed_parts)
      return lax.dot(x, y)

    x = onp.arange(onp.prod(shape), dtype=np.float32).reshape(shape)

    sharded_infeed_shape = list(infeed_shape.shape)
    # sharded_infeed_shape[0] //= 2
    y = onp.arange(
        onp.prod(sharded_infeed_shape),
        dtype=np.float32).reshape(sharded_infeed_shape)

    print("infeed shape: ", y.shape)
    for device in jax.devices():
      xla_client.transfer_to_infeed((y,), device)
    result = pmap(sharded_jit(f, partitions=(None, P(2, 1))))(x)
    print("printing result...", flush=True)
    print(result, flush=True)

  def testInfeedOnlySharding(self):
    if xb.local_device_count() < 2:
      raise SkipTest("requires 2 devices")

    shape = (jax.device_count() // 2, 4, 4)
    infeed_shape = jax.ShapedArray(shape[1:], np.float32)
    # None for token
    # TODO(skye): maybe replicate the token automatically
    infeed_parts = (P(2, 1), None)

    def f(x):
      token = lax.create_token(x)
      (y,), token = lax.infeed(token, (infeed_shape,), sharding=infeed_parts)
      return lax.dot(x, y)

    x = onp.arange(onp.prod(shape), dtype=np.float32).reshape(shape)

    sharded_infeed_shape = list(infeed_shape.shape)
    # sharded_infeed_shape[0] //= 2
    y = onp.arange(
        onp.prod(sharded_infeed_shape),
        dtype=np.float32).reshape(sharded_infeed_shape)

    print("infeed shape: ", y.shape)
    for device in jax.devices():
      xla_client.transfer_to_infeed((y,), device)
    result = pmap(sharded_jit(f, partitions=(None, P(2, 1))))(x)
    print("printing result...", flush=True)
    print(result, flush=True)

  def testNestedShardedJit(self):
    shape = (2, 4, 4)

    @partial(sharded_jit, partitions=(P(4,1), None))
    def f(x):
      y = x + 1
      z = sharded_jit(lambda x, y: lax.dot(x, y),
                      partitions=(P(2,2), P(2,2)))(y, y)
      return z + 1

    x = onp.arange(onp.prod(shape), dtype=np.float32).reshape(shape)
    expected_f = lambda x: lax.dot(x + 1, x + 1) + 1
    expected = np.stack((expected_f(x[0]), expected_f(x[1])))
    actual = pmap(f)(x)
    self.assertAllClose(actual, expected, check_dtypes=False)

  def testNestedShardedJitMismatchedPartitions(self):
    shape = (2, 4, 4)

    @partial(sharded_jit, partitions=(P(2,1), None))
    def f(x):
      y = x + 1
      z = sharded_jit(lambda x, y: lax.dot(x, y),
                      partitions=(P(2,2), P(2,2)))(y, y)
      return z + 1

    x = onp.arange(onp.prod(shape), dtype=np.float32).reshape(shape)
    with self.assertRaisesRegex(RuntimeError,
                                "Invalid sharding for instruction"):
      pmap(f)(x)


if __name__ == "__main__":
  absltest.main()
