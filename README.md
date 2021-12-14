# Yolox Implementation
Implementation of various Yolox models in C++.

# Technologies used

- YoloX Models. 
  - [Yolox-s](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s_openvino.tar.gz)
  - [Yolox-tiny](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny_openvino.tar.gz)
  - [Yolox-nano](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano_openvino.tar.gz)
- OpenVINO e OpenCV latest versions.
- Visual Studio 2022.
- Docker.
- XLaunch / XMing.
- PuTTy.
- CMake.

# Requisites needed for implementation

- Docker.
  - > https://docs.docker.com/desktop/windows/install/ 
- Ubuntu image already configured.
  - > https://mega.nz/file/jQknlQiK#IZl_IN6hlW1F8itC03bRN86k8Z0Bf9tx80lEaCiEvLU
- XLauch / XMing installed in host machine.
  - > https://sourceforge.net/projects/xming/
- PuTTy configured to use X11 over SSH.
  - > https://www.putty.org/ 
- Visual Studio 2022. (Optional)
  - > https://visualstudio.microsoft.com/es/ 

# Installation

- Download the ubuntu image from the link above.  

- Import that image into Docker and create a container from it.
  ```shell 
  docker load < yolo_docker_image.tar
  docker run IMAGE_ID 
  ```

- Run the new container, open a CMD in Windows and connect to it using "docker exec".
  ```shell 
  docker exec -it IMAGE_ID bash
  ```

- Start the SSH server with:
  ```shell
  service ssh start
  ```
  - Close the window
  - _You can also input this comand through the container CLI if  you have Docker Desktop_

- Open XLaunch and press Next or Enter multiple times until the windows closes, dont modify anything.

- Open PuTTy and connect to the container, by default __127.0.0.1__ and port __2222__ but first enable X11 forwarding on the left menu inside SSH/X11 and write ":0.0" inside "X display location".

![image](https://user-images.githubusercontent.com/56027490/145901558-c2650aba-9ba9-4dc7-b74d-dee60981f8fd.png)
![image](https://user-images.githubusercontent.com/56027490/145901606-a70a66ca-f85c-4410-b928-236f953fa5cb.png)

- Insert username __"root"__ and password __"wishmaster"__

- Navigate inside /root/ to:
  ```shell 
  cd /Yolo_CMake/bin
  ```
- Run the application:
  ```shell
  ./main
  ```

## _Inside /opt/ there are three important folders._

  - > /opt/samples/videos ==> All the video samples are located here.
  - > /opt/configuration ==> Here is the Config.json for the application, you can change the video that is used and the model.
  - > /opt/models ==> All the three models used are located here

