# Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds (SFA3D)

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]

---

## 1. Features
- [x] Super fast and accurate 3D object detection based on LiDAR
- [x] Fast training, fast inference
- [x] An Anchor-free approach
- [x] No Non-Max-Suppression
- [x] Support distributed data parallel training
- [x] Release pre-trained models 




## 2. Getting Started
### 2.1. Requirement

The instructions for setting up a virtual environment is [here](https://github.com/maudzung/virtual_environment_python3).

```shell script
git clone https://github.com/maudzung/SFA3D.git SFA3D
cd SFA3D/
pip install -r requirements.txt
```

### 2.2. Data Preparation
Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

generate data script + carla + saved files generate correct labels by running another script which one ask maarten
...
...


Please make sure that you construct the source code & dataset directories structure as below.

IMAGE OF DIRECTORY STRUCTURE 

### 2.3. How to train

#### 2.3.1. How to train on local machine

train script
blablabla

#### 2.3.3. How to train on GPU Lab

docker image
blablabla




### 2.4 Parallel Training

##### 2.3.4.1. Single machine, single gpu

```shell script
python train.py --gpu_idx 0
```

##### 2.3.4.2. Distributed Data Parallel Training
- **Single machine (node), multiple GPUs**

```
python train.py --multiprocessing-distributed --world-size 1 --rank 0 --batch_size 64 --num_workers 8
```

- **Two machines (two nodes), multiple GPUs**

   - _**First machine**_
    ```
    python train.py --dist-url 'tcp://IP_OF_NODE1:FREEPORT' --multiprocessing-distributed --world-size 2 --rank 0 --batch_size 64 --num_workers 8
    ```

   - _**Second machine**_
    ```
    python train.py --dist-url 'tcp://IP_OF_NODE2:FREEPORT' --multiprocessing-distributed --world-size 2 --rank 1 --batch_size 64 --num_workers 8
    ```



### 2.4 Testing


### 2.5 Logs on Weights ands Biases



## References

[1] CenterNet: [Objects as Points paper](https://arxiv.org/abs/1904.07850), [PyTorch Implementation](https://github.com/xingyizhou/CenterNet) <br>
[2] RTM3D: [PyTorch Implementation](https://github.com/maudzung/RTM3D) <br>
[3] Libra_R-CNN: [PyTorch Implementation](https://github.com/OceanPang/Libra_R-CNN)
_The YOLO-based models with the same BEV maps input:_ <br>
[4] Complex-YOLO: [v4](https://github.com/maudzung/Complex-YOLOv4-Pytorch), [v3](https://github.com/ghimiredhikura/Complex-YOLOv3), [v2](https://github.com/AI-liu/Complex-YOLO)
*3D LiDAR Point pre-processing:* <br>
[5] VoxelNet: [PyTorch Implementation](https://github.com/skyhehe123/VoxelNet-pytorch)

