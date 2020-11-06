# Keypoint-based Multi-Object Tracking (keypoint-mot)

This repository is intended for the Multi-Object Tracking as part of the internship project.

It provides an implementation of [CenterTrack](https://arxiv.org/pdf/2004.01177.pdf) in tensorflow.

#Installation
```
#clone the repo
git clone https://github.com/googleinterns/keypoint-mot.git
cd keypoint-mot

#optional, create a new virtual environment and activate it
#install the dependencies
pip install -r requirements.t
```

Next, if this [PR](https://github.com/tensorflow/addons/pull/2196) was approved, simply install tensorflow addons.

`pip install tensorflow-addons #install tensorflow addons`

Otherwise, tensorflow addons must be built from the repo in the PR.

```
git clone https://github.com/Licht-T/addons.git #clone the repo
cd addons
git checkout add-deformable-conv

export TF_NEED_CUDA="1"

# Set these if the below defaults are different on your system
export TF_CUDA_VERSION="10.1"
export TF_CUDNN_VERSION="7"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"

# This script links project with TensorFlow dependency
python3 ./configure.py

#add the --cxxopt option only if you are using gcc 5 or higher
bazel build --enable_runfiles build_pip_pkg --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl
```

# Usage
To instantiate CenterTrack (DLASeg instance), below is a usual configuration:
```
heads = {'hm': 10, 'reg': 2, 'wh': 2, 'tracking': 2, 'dep': 1,
         'rot': 8, 'dim': 3, 'amodel_offset': 2}
head_conv = {'hm':  [256], 'reg': [256], 'wh': [256],
             'tracking': [256], 'dep': [256], 'rot': [256],
             'dim': [256], 'amodel_offset': [256]}
opt = DLASegOptions(...)
model = DLASeg(num_layers=34, heads=heads, head_convs=head_conv, opt=opt)
```

For data loading, there are two options:
- `Dataset.from_generator(dataset_instance.get_input_generator(args), output_types=dataset_instance.return_dtypes)`
- `Dataset.range(dataset_len).map(map_func=dataset_instance.get_input_py_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)`

# Acknowledgement
The code is based on the official [CenterTrack](https://github.com/xingyizhou/CenterTrack) pytorch implementation,
released under MIT License. Please see the NOTICE for details.

**This is not an officially supported Google product.**
