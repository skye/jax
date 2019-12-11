# Example Cloud TPU notebooks

JAX now runs on Cloud TPUs!

The following notebooks showcase how to use and what you can do with Cloud TPUs on Colab:

### [Pmap Cookbook](https://colab.research.google.com/github/skye/jax/blob/nbtest/docs/notebooks/draft/Pmap_Cookbook.ipynb)
A guide to getting started with `pmap`, a transform for easily distributing SPMD
computations across devices.

### [Lorentz ODE Solver](https://colab.research.google.com/github/skye/jax/blob/nbtest/docs/notebooks/draft/Lorentz_ODE_Solver.ipynb)
Contributed by Alex Alemi (alexalemi@)

Solve and plot parallel ODE solutions with `pmap`.

![lorentz](https://raw.githubusercontent.com/skye/jax/nbtest/cloud_tpu_colabs/images/lorentz.png)

### [Wave Equation](https://colab.research.google.com/github/skye/jax/blob/nbtest/docs/notebooks/draft/Wave_Equation.ipynb)
Contributed by Stephan Hoyer (shoyer@)

Solve the wave equation with `pmap`, and make cool movies.

![wave_movie](https://raw.githubusercontent.com/skye/jax/nbtest/cloud_tpu_colabs/images/wave_movie.gif)

## Running JAX on TPU from a GCE VM

The process of creating Cloud TPU can be followed from this link: https://cloud.google.com/tpu/docs/quickstart

If you are already familiar with the process of creating Cloud TPU, the process of creating Cloud TPU involves creating the **user GCE VM** and the **TPU node**.

To create a **user GCE VM**, run the following command from your GCP console or your computer terminal where you have gcloud CLI installed (see [installing gcloud](https://cloud.google.com/sdk/install)).


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


This is a “default” configuration. To run a bigger compute VM, you may want to pick a different machine-type (see [machine types](https://cloud.google.com/compute/docs/machine-types)).

Next, create the **TPU node**. Please follow this [guideline](https://cloud.google.com/tpu/docs/internal-ip-blocks) to pick a $TPU_IP_ADDRESS.


```
export TPU_IP_ADDRESS=<pick-ip-address>
gcloud compute tpus create $USER-tpu-0001 \
      --zone=$ZONE \
      --network=default \
      --accelerator-type=v2-8 \
      --range=$TPU_IP_ADDRESS \
      --version=tpu_driver_nightly
```


The above command defaults to the ‘tpu_driver_nightly’ version. We may make the other versions available.

Now that you have created both the **user GCE VM** and the **TPU node**, ssh to the **user GCE VM** by executing the following gcloud ssh command:


```
gcloud compute ssh $USER-user-vm-0001
```


Once you are in the user GCE VM, from your ssh terminal session, follow the small instructions and example below to run a simple JAX program.

**Install jax and jaxlib wheels:**


```
pip install jax==0.1.54 jaxlib==0.1.37
```


**Create and run the following simple_jax.py program:**

**IMPORTANT**: replace the <TPU-IP-ADDRESS> below with the TPU node **IP address** (you can check this from the GCP console **<span style="text-decoration:underline;">Compute Engine > TPUs</span>**.


```
cat > simple_jax.py
from jax.config import config
from jax import random

# The following is required to use TPU Driver as JAX's backend.
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://<TPU-IP-ADDRESS>:8470"

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)
```


_[press ctrl+c to save the above program ...]_

**Run the program:**


```
python simplejax.py
```
