#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# clone YOLOv5 repository
get_ipython().system('git clone https://github.com/ultralytics/yolov5  # clone repo')
get_ipython().run_line_magic('cd', 'yolov5')
get_ipython().system('git reset --hard 064365d8683fd002e9ad789c1e91fa3d021b44f0')


# In[ ]:


# install dependencies as necessary from requirements.txt file
get_ipython().system('pip install -qr requirements.txt  # install dependencies')
import torch

from IPython.display import Image, clear_output  # to display images
from utils.downloads import attempt_download  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


# In[ ]:


get_ipython().system('pip install roboflow #To Upload dataset from Roboflow')

from roboflow import Roboflow
rf = Roboflow(api_key="ByYtkMvwqVxOlzsFt7T0")
project = rf.workspace("muhammad-syihab-bdynf").project("parking-space-ipm1b")
version = project.version(4)
dataset = version.download("yolov5")


# In[ ]:


get_ipython().run_line_magic('cd', '/content/yolov5')


# In[ ]:


get_ipython().run_line_magic('cat', '{dataset.location}/data.yaml')


# In[ ]:


# number of classes based on YAML
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])


# In[ ]:


get_ipython().run_line_magic('cat', '/content/yolov5/models/yolov5s.yaml')


# In[ ]:


# train yolov5s on custom data for 50 epochs
# time its performance
get_ipython().run_line_magic('%time', '')
get_ipython().run_line_magic('cd', '/content/yolov5/')
get_ipython().system("python train.py --img 416 --batch 16 --epochs 50 --data {dataset.location}/data.yaml --cfg ./models/yolov5s.yaml --weights '' --name yolov5s_results  --cache")


# In[ ]:


from utils.plots import plot_results  # plot results.txt as results.png
Image(filename='/content/yolov5/runs/train/yolov5s_results/results.png', width=1000)  # view results.png


# In[ ]:


get_ipython().run_line_magic('ls', 'runs/')


# In[ ]:


get_ipython().run_line_magic('ls', 'runs/train/yolov5s_results/weights')


# In[ ]:


#Running the model on pictures in test folder to check accuracy
get_ipython().run_line_magic('cd', '/content/yolov5/')
get_ipython().system('python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source /content/yolov5/Parking-Space-4/test/images')


# In[ ]:


import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg')[:20]: #assuming JPG
    display(Image(filename=imageName))


# In[ ]:


#create onnx file for OpenCV along with the weights
get_ipython().system('python export.py --weights /content/yolov5/runs/train/yolov5s_results/weights/best.pt --include onnx --simplify --opset 12')

