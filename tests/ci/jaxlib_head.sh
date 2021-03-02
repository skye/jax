
#!/bin/bash
set -x
set -u
set -e

pip install --upgrade pip
pip install --upgrade numpy==1.18.5 scipy wheel future six cython pytest absl-py opt-einsum msgpack

pushd /tmp
echo "Checking out TF..."
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
echo "TensorFlow git hash: $(git rev-parse HEAD)"
popd

pip install -e .
pip install jaxlib

num_devices=`python3 -c "import jax; print(jax.device_count())"`
if [ "$num_devices" = "1" ]; then
  echo "No TPU devices detected"
  exit 1
fi

export JAX_NUM_GENERATED_CASES=5
python3 -m pytest tests examples
