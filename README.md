<p align="center">

  <h1 align="center">Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM</h1>
  <p align="center">
    <a href="https://monica-mq-li.gitbook.io/"><strong>Monica Li</strong></a>
    ·
    <a href="https://lajoiepy.github.io/"><strong>Pierre-Yves Lajoie</strong></a>
    ·
    <a href="https://www.linkedin.com/in/jialing-liu-70b13b395"><strong>Jialing Liu</strong></a>
    ·
    <a href="https://mistlab.ca/people/beltrame/"><strong>Giovanni Beltrame</strong></a>
  </p>
  <div align="center"></div>
</p>



## ⚙️ Setting Things Up

Clone the repo:

```
git clone https://github.com/lemonci/Coko-SLAM.git
```

We tested the installation with ```gcc``` and ```g++``` of versions 10, 11 and 12. Also, make sure that ```nvcc --version``` matches ```nvidia-smi``` version.

Run the following commands to set up the environment
```
conda create -n coko-slam python=3.11
conda activate coko-slam
```
Install pytorch:
```
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```
Install other dependencies:
```
conda install -c conda-forge faiss-gpu=1.8.0
pip install -r requirements.txt
```

In case you are on a local machine and want to run visualization script run:
```
pip install rerun-sdk
```

We tested our code on RTX4070 and RTX H100 GPUs with Ubuntu22 and AlmaLinux 9 respectively.

## 🔨 Running Coko-SLAM

Here we elaborate on how to load the necessary data, configure Coko-SLAM for your use-case, debug it, and how to reproduce the results mentioned in the paper.

  <details>
  <summary><b>Getting the Data</b></summary>
  We tested our code on ReplicaMultiagent and AriaMultiagent datasets. Make sure to install git lfs and hugging face cli before proceeding.
  <br>
  <br>

  **ReplicaMultiagent** was created by CP-SLAM authors. However, it is a bit tricky to find it on the web. Therefore, we uploaded it to HF datasets for easier access. Install git lfs and download it by running: <br>
  <code>git lfs install</code> <br>
  <code>git clone https://huggingface.co/datasets/voviktyl/ReplicaMultiagent</code> <br>

  **AriaMultiagent** consists of clips from <a href="https://www.projectaria.com/datasets/adt/">AriaDigitalTwin</a>. First, download raw AriaDigitalTwin sequences following instructions <a href="https://www.projectaria.com/datasets/adt/">here</a>. We recommend using the data tools described <a href="https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/dataset_download">here</a>. You can find the names of the raw videos we used in <code>prepare_aria_room_data.py</code> in <code>DATASET_DICT</code>. After you have downloaded the raw videos, process them using <code>prepare_aria_room_data.py</code>. This will create a folder with the format matching our repository.
  
  </details>

  <details>
  <summary><b>Running the code</b></summary>
  Ideally, our system needs <code>n + 1</code> GPUs where <code>n</code> is the nubmer of agents. If you want to run the system for debugging purposes set <code>multi_gpu: False</code> and <code>agent_ids: [0]</code>. In this way, you will run a single agent and use the same GPU for the server and the agent. Start the system by running:

  ```
  python run_slam.py configs/<dataset_name>/<config_name> --input_path <path_to_the_scene> --output_path <output_path>
  ```
  For example:
  ```
  python run_slam.py configs/AriaMultiagent/room0.yaml --input_path <path_to>/AriaMultiagent/room0 --output_path output/AriaMultiagent/room0
  ```  
  In addition, you can set up wandb to log the results of the runs.
  </details> 

  <details>
  <summary><b>Reproducing Results</b></summary>
  While we tried to make all parts of our code deterministic, differential rasterizer of Gaussian Splatting is not. The metrics can be slightly different from run to run. In the paper we report average metrics that were computed over three seeds: 0, 1, and 2. 

  You can reproduce the results for a single scene by running:

  ```
  python run_slam.py configs/<dataset_name>/<config_name> --input_path <path_to_the_scene> --output_path <output_path>
  ```

  If you are running on a SLURM cluster, you can reproduce the results for all scenes in a dataset by running the script:
  ```
  ./scripts/reproduce_aria_sbatch.sh
  ``` 
  </details>

  <details>
  <summary><b>Demo</b></summary>
  To create the demo we first visualized the run in an offline fashion using <a href="https://rerun.io/">Rerun</a>. Further, we manually added the titles to the video. You can create the visualization by downloading the folder with the results and running <code>run_visualization.py</code> on top of it.
  </details>

## 🙏 Acknowledgments

This codebase is based on <a href="https://github.com/VladimirYugay/MAGiC-SLAM">MAGiC-SLAM</a>. We thank <a href="https://vladimiryugay.github.io/">Vladimir Yugay</a>, for his detailed explaination and the fruitful discussions.