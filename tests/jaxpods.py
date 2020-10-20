import jax_pod_setup

import collections

import jax
from jax import lax, pmap
from jax.interpreters.sharded_jit import (sharded_jit, with_sharding_constraint,
                                          PartitionSpec as P)
import jax.numpy as jnp
import jax.lib.xla_bridge as xb
import numpy as np

print("========= jax.device_count(): %s" % jax.device_count())
print("========= local_device_count: %s" % jax.local_device_count())
print("========= devices: %s" % ",".join(map(str, jax.devices())))
print("========= host_id: %s" % jax.host_id())

# data = np.ones(8)

# # By default, pmap runs across entire pod.
# result = jax.pmap(
#     lambda x: lax.psum(x, 'i'),
#     axis_name='i')(data)
# print("========= %s" % result)
# assert list(result) == [32.] * 8

# # Explicitly specifying all devices is equivalent to above.
# result = jax.pmap(
#     lambda x: lax.psum(x, 'i'),
#     axis_name='i',
#     devices=jax.devices())(data)
# print("========= %s" % result)
# assert list(result) == [32.] * 8

# # Same as above, but with no data dependence between input and output of
# # mapped function.
# result = jax.pmap(
#     lambda x: lax.psum(1, 'i'),
#     axis_name='i')(data)
# print("========= %s" % result)
# assert list(result) == [32.] * 8

# result = jax.pmap(
#     lambda x: lax.psum(1, 'i'),
#     axis_name='i',
#     devices=jax.devices())(data)
# print("========= %s" % result)
# assert list(result) == [32.] * 8

# # Specifying a subset of devices (and hosts) to run on. Note we must pass data
# # sized according to the number of devices, and we don't call pmap on hosts
# # with no participating devices.
# host2devices = collections.defaultdict(list)
# for d in jax.devices():
#   host2devices[d.host_id].append(d)

# devices = host2devices[0] + host2devices[1][:4]
# if jax.host_id() == 1:
#   data = np.ones(4)

# if jax.host_id() in (0, 1):
#   result = jax.pmap(
#       lambda x: lax.psum(x, 'i'),
#       axis_name='i',
#       devices=devices)(data)
#   print("========= %s" % result)
#   assert list(result) == [12.] * len(data)

# # TODO: make empty data work
# # result = jax.pmap(
# #     lambda x: lax.psum(x, 'i'),
# #     axis_name='i',
# #     devices=None)(np.ones(0))
# # print("========= %s" % result)
# # assert list(result) == []

# # TODO: run on subset of devices when devices aren't explicitly provided.

# # Outer pmap over host.
# # All axis_size arguments must be provided.
# f1 = lambda x: (lax.psum(x, 'rows'), lax.psum(x, 'cols'),
#                 lax.psum(x, ('rows', 'cols')))
# f2 = lambda x: (lax.axis_index('rows'), lax.axis_index('cols'))

# result = jax.pmap(
#     jax.pmap(f1, axis_name='cols', axis_size=4),
#     axis_name='rows', axis_size=8)(np.ones((2, 4)))
# print("========= %s" % repr(result))

# result = jax.pmap(
#     jax.pmap(f2, axis_name='cols', axis_size=4),
#     axis_name='rows', axis_size=8)(np.ones((2, 4)))
# print("========= %s" % repr(result))

ldc = jax.local_device_count()
gdc = jax.device_count()

# sharded_jit

x_shape = (2 * gdc, 1 * gdc)
y_shape = (1 * gdc, 2)
x = np.arange(np.prod(x_shape)).reshape(x_shape)
y = np.arange(np.prod(y_shape)).reshape(y_shape)

x_parts = P(1, gdc)
local_x_parts = P(1, ldc)
start = jax.host_id() * ldc
end = (jax.host_id() + 1) * ldc
x_shard = x[:, start:end]

y_parts = None
local_y_parts = None
y_shard = y

out_parts = P(gdc, 1)
local_out_parts = P(ldc, 1)

result = sharded_jit(lambda x, y: jnp.dot(x, y),
                     in_parts=(x_parts, y_parts),
                     out_parts=out_parts,
                     local_in_parts=(local_x_parts, local_y_parts),
                     local_out_parts=local_out_parts)(x_shard, y_shard)
print(result.shape)
print("=========\n",repr(result))
start = jax.host_id() * 2 * ldc
end = (jax.host_id() + 1) * 2 * ldc
expected = np.dot(x,y)[start:end]
assert result.shape == expected.shape
assert (result == expected).all()

np.set_printoptions(linewidth=160)

# pmap of sharded_jit
num_replicas = 1
num_parts = gdc // num_replicas
num_local_parts = ldc // num_replicas

shape = (num_replicas, 2, num_parts // 2)
x = np.arange(np.prod(shape)).reshape(shape)
# x = np.repeat(np.arange(8), 4).reshape(shape)

x_parts = P(2, num_parts // 2)
local_x_parts = P(2, num_local_parts // 2)

slice_length = shape[2] // jax.host_count()
start = jax.host_id() * slice_length
end = (jax.host_id() + 1) * slice_length
x_shard = x[:, :, start:end]

shift_id = (jax.host_id() - 1) % jax.host_count()
shift_start = shift_id * slice_length
shift_end = (shift_id + 1) * slice_length
shift_shard = x[:, :, shift_start:shift_end]

out_parts = (None, x_parts)
local_out_parts = (None, local_x_parts)

# print("shape", shape)
# print("x_shard.shape", x_shard.shape)
# print("x")
# print(x)
# print("x_shard")
# print(x_shard)

device_map = {d.id: d for d in jax.devices()}
#devices = [device_map[i] for i in range(jax.device_count())]
partition_ids_1d = [
    0,1,2,3,
    4,5,6,7,

    16,17,18,19,
    20,21,22,23,

    8,9,10,11,
    12,13,14,15,

    24,25,26,27,
    28,29,30,31,
]
# partition_ids_1d = list(range(jax.device_count()))
devices = [device_map[i] for i in partition_ids_1d]
print("devices", partition_ids_1d)
# print("devices", [(i, d_id) for i, d_id in enumerate(partition_ids_1d)])

result = (
    pmap(
        sharded_jit(
            lambda x: (lax.psum(x, "i"), jnp.roll(x, slice_length, -1)),
            in_parts=x_parts,
            out_parts=out_parts,
            local_in_parts=local_x_parts,
            local_out_parts=local_out_parts),
        axis_name="i",
        axis_size=num_replicas,
        global_replica_shapes=(P(*x.shape[1:]),),
        devices=devices
    )(x_shard))

print("result[0]")
print(result[0])
print("result[1]")
print(result[1])
print("host id", jax.host_id())

assert result[1].shape == shift_shard.shape
assert (result[1] == shift_shard).all(), shift_shard


# Replicated argument
y = np.array([100,])
y_parts = None

result = (
    pmap(
        sharded_jit(
            lambda x, y: (jnp.roll(x, slice_length, -1) + y),
            in_parts=(x_parts, y_parts),
            out_parts=x_parts,
            local_in_parts=(local_x_parts, None),
            local_out_parts=local_x_parts),
        in_axes=(0, None),
        axis_size=num_replicas,
        global_replica_shapes=(P(*x.shape[1:]), P(*y.shape)),
        devices=devices
    )(x_shard, y))

print("result", result)

# Full replication
result = (
    pmap(
        sharded_jit(
            lambda x: (
                with_sharding_constraint(jnp.roll(x, slice_length, -1),
                                         x_parts) + 1),
            in_parts=None,
            out_parts=None,
            local_num_partitions=np.prod(local_x_parts)),
        in_axes=(0),
        axis_size=num_replicas,
        global_replica_shapes=(P(*x.shape[1:])),
        devices=devices
    )(x))

print("result", result)
assert (result == jnp.roll(x, slice_length, -1) + 1).all()


# Logical devices
logical_devices = P(*range(jax.device_count()))

result = (
    pmap(
        sharded_jit(
            lambda x: jnp.roll(x, slice_length, -1),
            in_parts=x_parts,
            out_parts=x_parts,
            local_in_parts=local_x_parts,
            local_out_parts=local_x_parts,
            logical_devices_in=logical_devices,
            logical_devices_out=logical_devices),
        axis_name="i",
        axis_size=num_replicas,
        global_replica_shapes=(P(*x.shape[1:]),),
        devices=devices
    )(x_shard))

assert result.shape == shift_shard.shape
assert (result == shift_shard).all(), f"result\n{result}\nexpected\n{shift_shard}"
