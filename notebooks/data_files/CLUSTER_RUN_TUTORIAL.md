# How to run GraPE Hyperparam optimization on DTU HPC cluster
### prerequisites
- you have a way to login to the node over ssh, and your keys are stored in `~/.ssh/gbar`
- git and pip are installed on the node and up to date (should be the case by default)
- you have access your own copy of the datasets you want to use
- you are on a fresh environment, or do not have conflicting versions of pytorch, pytroch-geometric, or similar installed.
This guide assumes python `3.9.18`

### Installing required packages
the version of pip on the cluster creates version mismatches if left to do things on its own, install packages in the following order to avoid this:
```
pip install torch==2.1.0
pip install torch-geometric
```
package `torch-scatter` part of `torch-geometric` is wrongly referenced in `torch-geometric`'s build 
wheel for the versions compatible with the cluster,
so we will install it from source:
```
curl -O https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_scatter-2.1.2%2Bpt21cu121-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.2%2Bpt21cu121-cp39-cp39-linux_x86_64.whl
```

### Installing GraPE
```
git clone https://github.com/aerte/GraPE.git
cd GraPE/
```
(as of early NOV2024): `git checkout hype-workflows`
```
pip install -r requirements_gpu_2.txt
pip install .
```

### moving the datasets to the cluster
A lot of the workflow files expect data to be in a `env/` directory that is at the root of where you are running th ecode from.<br>
This can for example be the root of the GraPE repo/project.<br>
navigate to the directory where you wan tto run `GraPE` from in the cluster, and run 
```
pwd
```
you should see something like `/zhome/8c/5/143289/GraPE`. copy this path.<br>
Now, there are two options:
-  **option 1** moving your local `env/` directory to the right place in the cluster:<br>
    on your local machine:
    ```
    scp -r -i ~/.ssh/gbar env <your-dtu-ID>@transfer.gbar.dtu.dk:/zhome/your/path/123456/GraPE/
    ```
-  **option 2** moving just the file you need from your local machine to the cluster, e.g. `ICP.xlsx` or `pka.csv`
    On the cluster:
    ```
    cd GraPE/
    mkdir env
    ```
    on your machine
    ```
    scp -i ~/.ssh/gbar path/to/your/file <your-dtu-ID>@transfer.gbar.dtu.dk:/zhome/your/path/123456/GraPE/env/
    ```
of course replacing your dtu ID and path where relevant

## Hyperparameter optimization
### setting up the optimization script
keep the output of `pwd` from the directory where GraPE will be run.<br>
from the root of GraPE:
```
cp notebooks/set_env_vars_cluster.sh set_env_vars_cluster.sh
```
use your favorite terminal based editor (e.g. vi, nano, or vim - they are all on the cluster) to edit the file:

```
vim set_env_vars_cluster.sh
```

and change the `export BASE_R_DIR=` to contain your own root directory variable
If your hyperparam optimization script is using any additional environment variables, 
it is in this file that you should add and set them.<br>

run the script to set environment variables:
```
chmod +x set_env_vars_cluster.sh
source set_env_vars_cluster.sh
```
Now, you should be ready to run hyperperameter optimizations, as a quick check you can perform
```
python notebooks/optimization_test.py --samples 100 pka
```
Wait for a few samples to finish, if no errors are being printed, your script is ready to be sumbitted to the job queue, so Ctrl-C a bunch times to stop it and move on to the next section.

### creating the job and submitting it to the queue
make sure you're on a GPU-node, for example
```
a110sh
```
**TODO**: finish