# Example Cloud TPU notebooks

JAX now runs on Cloud TPUs!

The following notebooks showcase how to use and what you can do with Cloud TPUs on Colab:

### [Pmap Cookbook](https://colab.research.google.com/github/skye/jax/blob/nbtest/docs/notebooks/draft/Pmap_Cookbook.ipynb)
A guide to getting started with `pmap`, a transform for easily distributing SPMD
computations across devices.

### [Lorentz ODE Solver](https://colab.research.google.com/github/skye/jax/blob/nbtest/docs/notebooks/draft/Lorentz_ODE_Solver.ipynb)
Contributed by Alex Alemi (alexalemi@)

Solve and plot parallel ODE solutions with `pmap`.

<img src="https://raw.githubusercontent.com/skye/jax/nbtest/cloud_tpu_colabs/images/lorentz.png" width=65%></image>

### [Wave Equation](https://colab.research.google.com/github/skye/jax/blob/nbtest/docs/notebooks/draft/Wave_Equation.ipynb)
Contributed by Stephan Hoyer (shoyer@)

Solve the wave equation with `pmap`, and make cool movies.

![](https://raw.githubusercontent.com/skye/jax/nbtest/cloud_tpu_colabs/images/wave_movie.gif)

## Running JAX on a Cloud TPU from a GCE VM

Creating a [Cloud TPU](https://cloud.google.com/tpu/docs/quickstart) involves creating the user GCE VM and the TPU node.

To create a user GCE VM, run the following command from your GCP console or your computer terminal where you have [gcloud installed](https://cloud.google.com/sdk/install).

```
export ZONE=us-central1-c
gcloud compute instances create $USER-user-vm-0001 \
   --machine-type=n1-standard-1 \
   --image-project=ml-images \
   --image-family=tf-1-14 \
   --boot-disk-size=200GB \
   --scopes=cloud-platform \
   --zone=$ZONE
```

To create a larger GCE VM, choose a different [machine type](https://cloud.google.com/compute/docs/machine-types).

Next, create the TPU node, following these [guidelines](https://cloud.google.com/tpu/docs/internal-ip-blocks) to choose a <TPU_IP_ADDRESS>.

```
export TPU_IP_ADDRESS=<TPU_IP_ADDRESS>
gcloud compute tpus create $USER-tpu-0001 \
      --zone=$ZONE \
      --network=default \
      --accelerator-type=v2-8 \
      --range=$TPU_IP_ADDRESS \
      --version=tpu_driver_nightly
```

Now that you have created both the user GCE VM and the TPU node, ssh to the GCE VM by executing the following command:

```
gcloud compute ssh $USER-user-vm-0001
```

Once you are in the VM, from your ssh terminal session, follow the example below to run a simple JAX program.

**Install jax and jaxlib wheels:**


```
pip install jax==0.1.54 jaxlib==0.1.37
```


**Create a program, simple_jax.py:**

**IMPORTANT**: Replace <TPU_IP_ADDRESS> below with the TPU node’s IP address. You can get the IP address from the GCP console: Compute Engine > TPUs.

```
from jax.config import config
from jax import random

# The following is required to use TPU Driver as JAX's backend.
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://<TPU-IP-ADDRESS>:8470"

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)
```

**Run the program:**

```
python simplejax.py
```
