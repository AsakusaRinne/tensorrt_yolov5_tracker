# tensorrt_yolov5_tracker

This is a project to deploy object tracking algorithm with yolov5 and TensorRT. Sort and Deep-sort algorithm are used to track the objects.
Thanks for the contribution of [tensorrtx](https://github.com/wang-xinyu/tensorrtx) and [sort-cpp](https://github.com/yasenh/sort-cpp)!
Now this project implements the tracki algorithm with yolov5 and sort(c++ version). Please wait for the update for deep-sort and python implemention. 

![example result of object tracking](https://github.com/AsakusaRinne/tensorrt_yolov5_tracker/blob/main/example.gif)
## Table of Contents

- [ToDo](#ToDo)
- [Install](#install)
- [Usage](#usage)
- [Reference](#Reference)
- [License](#license)

## ToDo
***1.Deep-sort tracker.
2.Multi-thread acceleration for c++.
3.Python implemention.***


## Install

```
$ git https://github.com/AsakusaRinne/tensorrt_yolov5_tracker.git
$ cd tensorrt_yolov5_tracker
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Usage
You could change the batch size in ``yolov5/yolov5.h``
You could change the input image size in ``yolov5/yololayer.h``

We provide the support for building models of yolov5-3.1 and yolov5-4.0 in our project. The default branch is corresponding to yolov5-4.0. Please change to ``c269f65`` if you want to use yolov5-3.1 models to build engines.

If other versions are wanted, please kindly refer to [tensorrtx](https://github.com/wang-xinyu/tensorrtx) to build the engine and then copy to the 


### Build engine
At first, please put the ``yolov5/gen_wts.py`` in the folder of corresponding version of [**Yolov5**](https://github.com/ultralytics/yolov5). For example, if you use the model of yolov5-4.0, please download the release of yolov5-4.0 and put the ``gen_wts.py`` in its folder. Then run the code to convert model from ``.pt`` to ``.wts``. Then please use the following instructions to build your engines.
```
$ sudo ./Tracker -s [.wts filename] [.engine filename] [s, m, l, or x] # build yolov5 model of [s, m, l, or x]

$ sudo ./Tracker -s  [.wts filename] [.engine filename] [depth] [width] # build yolov5 model of scales which are not s, m, l, or x
**for example**
$ sudo ./Tracker -s  ../yolov5/weights/yolov5.wts ../engines/yolov5.engine 0.17 0.25
```
### Process video
```
$ sudo ./Tracker -v [video filename] [engine filename]
```
The output video will be in the folder ``output``
Note that the track time includes the time of drawing boxes on images and saving them to the disk, which makes it a little long. You can delete the codes for drawing and saving if they are not needed to accelerate the process.

## Reference
[**tensorrtx**](https://github.com/wang-xinyu/tensorrtx)

[**Yolov5**](https://github.com/ultralytics/yolov5)

[**sort-cpp**](https://github.com/yasenh/sort-cpp)

## License
[MIT license](https://mit-license.org/)
