# AIonWheels
I-DistributedAI final project at the University of Antwerp

This goal of this project is to create an adaptive cruisecontrol using 
a combination of computerisation and reinforcement learning. 

The tree different modules were trained separately. A computer vision model to recognize objects 
using a camera. Another computer vision model to recognize cars in lidar data and finally a reinforcement learning
agent to control the car's acceleration. Each of these modules have their own dedicated folder to create and train the models. More info can be found in their respective readme files.
The FinalIntegration folder consist of the project code where all modules come together.
---
#Setup
To install the required packages use pip to install requirements.txt
```cmd
pip install -r requirements.txt
```
Running the final integration consists of running the main.py file in the root folder. Before running the
main.py script an instance of Carla must be running. This can be local or on a remote host.
```cmd
python --host [ip] --debug --MT --demoX main.py
```
|Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|Parameters|
|---------------|----------|
|--host [ip]|The host ip running carla. By default the localhost is used|
|--debug -d|Add this flag to enable debug mode.|
|--MT|Add this flag to enable python multithreading (warning this is not multiprocessing!)|
|--DemoX|Where X can be a value indicating which demo must be started. Multiple demo's can be created by defining a seed in the code.|

Each of the different modules has their own config file in the main.py file. Remeber these values must correspond to the model being loaded!

It is recommended to have at least 6 GB of vRAM to run the models and another 6 GB of VRAM to run Carla.

All model paths can be defined in the configs.





