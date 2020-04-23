# Copyright 2020 Google LLC
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

from functools import partial
from typing import Callable, Dict, Optional, Sequence, Type

from absl import logging
import numpy as onp

from .. import core
from ..abstract_arrays import ShapedArray, ConcreteArray, array_types
from . import partial_eval as pe
# TODO(skye): separate pmap into it's own module?
from . import ad
from . import pxla
from . import xla
from .. import linear_util as lu
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from ..api_util import flatten_fun
from ..tree_util import tree_flatten, tree_unflatten, tree_multimap, _replace_nones
from ..util import extend_name_stack, wrap_name, safe_zip

xops = xc._xla.ops

"""WIP sharded_jit

TODO:
* Fix mismatched layout issue (see commented-out test cases in spmd_test.py)
* Apply partitions to pytrees
* Testing + support beyond pmap of sharded_jit
* Return constants from sharded_jit
"""


def _map(f, *xs):
  return tuple(map(f, *xs))


### arg handling


def _spatial_partitioned_args(devices, assignments, partitions, args):
  nargs = len(args)
  nrep, npar = assignments.shape
  # buffers = [[[None] * nargs for _ in range(npar)] for _ in range(nrep)] # TODO
  buffers = [[None] * nargs for _ in range(nrep * npar)]
  for a, (arg, partition) in enumerate(safe_zip(args, partitions)):
    bufs = _partition_array(arg, devices, assignments,
                                             partition)
    for r in range(nrep):
      for p in range(npar):
        # buffers[r][p][a] = bufs[r][p]  # TODO update C++
        buffers[r * npar + p][a] = bufs[r][p]
  return buffers


partition_arg_handlers = {}


def _partition_array(x, devices, assignments, partition):
  nrep, npar = assignments.shape
  assert nrep == 1  # TODO generalize beyond single-replica
  shards = [x]
  for i, parts in enumerate(partition):
    shards = _flatten(onp.split(s, parts, i) for s in shards)
  bufs = [[None] * npar for _ in range(nrep)]
  for (r, p), device in onp.ndenumerate(assignments):
    bufs[r][p] = xla.device_put(shards[p], devices[device])
  return bufs


def _flatten(lst):
  return [elt for sublst in lst for elt in sublst]


for _t in array_types:
  partition_arg_handlers[_t] = _partition_array

### result handling

class ResultToPopulate(object): pass
result_to_populate = ResultToPopulate()

def _pvals_to_results_handler(nrep, npart, partitions, out_pvals):
  nouts = len(out_pvals)
  handlers = [_pval_to_result_handler(npart, parts, out_pval)
              for parts, out_pval in safe_zip(partitions, out_pvals)]

  def handler(out_bufs):
    assert nrep * npart == len(out_bufs)
    buffers = [[result_to_populate] * nrep * npart for _ in range(nouts)]
    for r, tuple_buf in enumerate(out_bufs):
      for i, buf in enumerate(tuple_buf):
        buffers[i][r] = buf
    assert not any(buf is result_to_populate for bufs in buffers
                   for buf in bufs)
    return [h(bufs) for h, bufs in zip(handlers, buffers)]

  return handler


def _pval_to_result_handler(npart, parts, pval):
  pv, const = pval
  if pv is None:
    raise NotImplementedError  # TODO handle constant outputs
  else:
    if pv is not core.abstract_unit:
      spec = _partitioned_sharding_spec(npart, parts, pv)
      indices = pxla.spec_to_indices(pv.shape, spec)
    else:
      spec = indices = None
    return pxla.aval_to_result_handler(spec, indices, pv)


def _aval_to_result_handler(partition, aval):
  return result_handlers[type(aval)](partition, aval)


result_handlers: Dict[Type[core.AbstractValue], Callable] = {}


def _array_result_handler(partition, aval):

  def handler(bufs):
    bufs, = bufs  # TODO generalize beyond single replica
    shards = [buf.to_py() for buf in bufs]  # TODO device persistence
    partition = (1,) # TODO (wangtao): revisit this hack.
    for i, parts in enumerate(partition):
      shards = [onp.concatenate(cs, axis=i) for cs in _chunk(shards, parts)]
    result = shards
    return result

  return handler


def _chunk(lst, sz):
  assert not len(lst) % sz
  return [lst[i:i + sz] for i in range(0, len(lst), sz)]


result_handlers[ShapedArray] = _array_result_handler
result_handlers[ConcreteArray] = _array_result_handler

### computation building


@lu.cache
def _sharded_callable(fun, num_partitions, partitions, out_parts_thunk, name,
                      *abstract_args):
  if xb.get_backend().platform != "tpu":
    logging.warning("sharded_jit only works on TPU")

  nrep = 1
  in_pvals = [pe.PartialVal.unknown(aval) for aval in abstract_args]
  jaxpr, out_pvals, consts = pe.trace_to_jaxpr(fun, in_pvals, instantiate=False, bottom=True)

  if not jaxpr.eqns and all(outvar.aval is core.abstract_unit
                            for outvar in jaxpr.outvars):
    return lambda *_: [
        const if pv is None else core.unit for pv, const in out_pvals
    ]

  out_parts = out_parts_thunk()

  c = xb.make_computation_builder("spjit_{}".format(fun.__name__))
  xla_consts = _map(partial(xb.constant, c), consts)
  xla_args = _xla_sharded_args(c, abstract_args, partitions[0])
  axis_env = xla.AxisEnv(nrep, (), ())
  out_nodes = xla.jaxpr_subcomp(
      c, jaxpr, None, axis_env, xla_consts (),
      extend_name_stack(wrap_name(name, "sharded_jit")), *xla_args)
  out_tuple = xb.with_sharding(c, out_parts, xops.Tuple, c, out_nodes)
  built = c.Build(out_tuple)

  # logging.error("========= hlo start")
  # logging.error(built.as_hlo_text())
  # logging.error("========= hlo end")

  devices = xb.local_devices()[:num_partitions]
  assert len(devices) == num_partitions  # TODO generalize beyond single-replica
  device_assignment = onp.array([[d.id for d in devices]])
  device_assignment = onp.reshape(device_assignment, (-1, num_partitions))
  # device_assignment = None  # TODO(skye): replace with default device assignment?

  backend = xb.get_backend()
  compiled = backend.compile(
      built,
      compile_options=xb.get_compile_options(nrep, num_partitions, device_assignment))

  # logging.error("========= hlo optimized start")
  # logging.error(compiled.hlo_modules()[0].to_string())
  # logging.error("========= hlo optimized end")

  input_specs = [
      _partitioned_sharding_spec(num_partitions, parts, aval)
      for parts, aval in zip(partitions[0], abstract_args)]
  input_indices = [pxla.spec_to_indices(aval.shape, spec)
                   if spec is not None else None
                   for aval, spec in zip(abstract_args, input_specs)]

  handle_args = partial(pxla.shard_args, compiled.local_devices(),
                        input_indices)

  handle_args = partial(_spatial_partitioned_args, compiled.local_devices(),
                        device_assignment, partitions[0])
  handle_outs = _pvals_to_results_handler(nrep, num_partitions, out_parts,
                                          out_pvals)
  return partial(_execute_spatially_partitioned, compiled, handle_args,
                 handle_outs)


def _partitioned_sharding_spec(num_partitions: int,
                               partitions: Optional[Sequence[int]], aval):
  if aval is core.abstract_unit:
    return None

  if partitions is None:
    return pxla.ShardingSpec(
        # int(1) because pytype is confused by 1 (???)
        shards_per_axis=(int(1),) * len(aval.shape),
        is_axis_materialized=(True,) * len(aval.shape),
        replication_factor=num_partitions)
  else:
    assert len(partitions) == len(aval.shape)
    return pxla.ShardingSpec(
        shards_per_axis=tuple(partitions),
        is_axis_materialized=(True,) * len(aval.shape),
        replication_factor=1)

def _sharded_jit_translation_rule(c, axis_env, in_nodes, name_stack,
                                  num_partitions, partitions, backend, name,
                                  call_jaxpr, out_parts_thunk):
  subc = xb.make_computation_builder("jaxpr_subcomputation")  # TODO(mattjj): name

  in_parts = partitions[0]
  # We assume any extra leading in_nodes are constants and replicate them.
  assert len(in_parts) <= len(in_nodes)
  in_parts = (None,) * (len(in_nodes) - len(in_parts)) + in_parts

  args = []
  for i, (n, sharding) in enumerate(safe_zip(in_nodes, in_parts)):
    arg = xb.parameter(subc, i, c.GetShape(n))
    # Inlined calls shouldn't have shardings set directly on the inputs or
    # outputs.
    args.append(xb.set_sharding(subc, arg, sharding))

  out_nodes = xla.jaxpr_subcomp(
      subc, call_jaxpr, backend, axis_env, (),
      extend_name_stack(name_stack, wrap_name(name, "sharded_jit")), *args)
  out_parts = out_parts_thunk()
  assert len(out_parts) == len(out_nodes)
  out_nodes = [xb.set_sharding(subc, out, part)
               for out, part in safe_zip(out_nodes, out_parts)]

  out_tuple = xops.tuple(subc, out_nodes)
  subc = subc.Build(out_tuple)
  return xops.Call(c, subc, list(in_nodes))


def _execute_spatially_partitioned(compiled, in_handler, out_handler, *args):
  input_bufs = in_handler(args)
  out_bufs = compiled.execute_on_local_devices(list(input_bufs))
  return out_handler(out_bufs)


def _xla_sharded_args(c, avals, partitions):
  xla_args = []
  for i, (p, a) in enumerate(safe_zip(partitions, avals)):
    xla_args.append(xb.parameter(c, i, xla.aval_to_xla_shape(a), sharding=p))
  return xla_args


def _get_num_partitions(partitions):
  print("==== ", partitions)
  if not partitions:
    return None
  num_partitions = onp.prod(partitions)
  print("== ", num_partitions)
  return num_partitions


def get_num_partitions(partitions):
  num_partitions_set = set(
      _get_num_partitions(parts) for parts in tree_flatten(partitions)[0])
  num_partitions_set.discard(None)
  if len(num_partitions_set) == 0:
    return 1
  if len(num_partitions_set) > 1:
    raise ValueError(
        "All partition specs must use the same number of total partitions, "
        "got: %s %s" % (partitions, num_partitions_set))
  return num_partitions_set.pop()


### sharded_call


def _sharded_call_impl(fun, *args, num_partitions, partitions, name,
                       out_parts_thunk):
  compiled_fun = _sharded_callable(fun, num_partitions, partitions,
                                   out_parts_thunk, name,
                                   *map(xla.abstractify, args))
  return compiled_fun(*args)


sharded_call_p = core.Primitive("sharded_call")
sharded_call_p.call_primitive = True
sharded_call_p.multiple_results = True
sharded_call = partial(core.call_bind, sharded_call_p)
sharded_call_p.def_custom_bind(sharded_call)
sharded_call_p.def_impl(_sharded_call_impl)
xla.call_translations[sharded_call_p] = _sharded_jit_translation_rule

def _set_sharding_impl(x, partitions):
  raise NotImplementedError(
      "set_sharding() should only be called inside sharded_jit()")

set_sharding_p = core.Primitive("set_sharding")
set_sharding_p.def_impl(_set_sharding_impl)
set_sharding_p.def_abstract_eval(lambda x, partitions: x)
ad.deflinear(set_sharding_p, lambda ct, partitions: (set_sharding(ct, partitions),))

def _set_sharding_translation_rule(c, x_node, partitions):
  return xb.set_sharding(c, x_node, partitions)
xla.translations[set_sharding_p] = _set_sharding_translation_rule

def set_sharding(x, partitions):
  return set_sharding_p.bind(x, partitions=partitions)

def _flatten_axes(treedef, axis_tree):
  # given an axis spec tree axis_tree (a pytree with integers and Nones at the
  # leaves, i.e. the Nones are to be considered leaves) that is a tree prefix of
  # the given treedef, build a complete axis spec tree with the same structure
  # and return the flattened result
  # TODO(mattjj,phawkins): improve this implementation
  proxy = object()
  dummy = tree_unflatten(treedef, [object()] * treedef.num_leaves)
  axes = []
  add_leaves = lambda i, x: axes.extend([i] * len(tree_flatten(x)[0]))
  try:
    tree_multimap(add_leaves, _replace_nones(proxy, axis_tree), dummy)
  except ValueError:
    msg = ("axes specification must be a tree prefix of the corresponding "
           "value, got specification {} for value {}.")
    raise ValueError(msg.format(axis_tree, treedef))
  axes = [None if a is proxy else a for a in axes]
  assert len(axes) == treedef.num_leaves
  return axes


class PartitionSpec(tuple):

  def __new__(self, *partitions):
    return tuple.__new__(PartitionSpec, partitions)

  def __repr__(self):
    return "PartitionSpec%s" % tuple.__repr__(self)


def sharded_jit(fun, partitions, num_partitions=None):
  """
  """
  if tree_flatten(partitions)[0]:
    # There exists at least one non-None partition (tree_flatten removes Nones)
    num_parts = get_num_partitions(partitions)
  elif num_partitions is not None:
    num_parts = num_partitions
  else:
    # No partitions specified
    num_parts = 1

  if num_partitions is not None and num_partitions != num_parts:
    raise ValueError(
        "Got mismatched 'partitions' and 'num_partitions' arguments: %s vs %d "
        "('partitions' argument implies %d partitions) %s" %
        (partitions, num_partitions, num_parts, tree_flatten(partitions)))

  def wrapped(*args, **kwargs):
    if kwargs:
      raise NotImplementedError("sharded_jit over kwargs not yet supported")
    if partitions[0] and len(partitions[0]) < len(args):
      raise ValueError("sharded_jit got %d args but only %d input partitions" %
                       (len(args), len(partitions[0])))
    f = lu.wrap_init(fun)
    args_flat, in_tree = tree_flatten((args, kwargs))
    in_parts = tuple(_flatten_axes(in_tree.children()[0], partitions[0]))
    flat_fun, out_tree = flatten_fun(f, in_tree)
    # TODO(skye): having a function-typed param in a primitive seems dicey, is
    # there a better way?
    out_parts_thunk = lambda: tuple(_flatten_axes(out_tree(), partitions[1]))
    out = sharded_call(
        flat_fun,
        *args_flat,
        num_partitions=num_parts,
        partitions=(in_parts, object()),
        name=flat_fun.__name__,
        out_parts_thunk=out_parts_thunk)
    return tree_unflatten(out_tree(), out)

  return wrapped
