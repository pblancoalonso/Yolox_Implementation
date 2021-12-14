# Yolox Implementation
Implementation of various Yolox models in C++.

# Technologies used

- Modelos de YoloX. (Yolox-s, Yolox-tiny, Yolox-nano)
- OpenVINO e OpenCV.
- Visual Studio 2022.
- Docker.
- XLaunch / XMing.
- PuTTy.
- CMake.

# Requisites needed for implementation

- Docker.
- Ubuntu image.
- XLauch / XMing installed in host machine.
- PuTTy configured to use X11 over SSH.
- Visual Studio 2022.

# Installation

- Download the ubuntu image from the link.
  - https://mega.nz/file/jQknlQiK#IZl_IN6hlW1F8itC03bRN86k8Z0Bf9tx80lEaCiEvLU

- Import that image into Docker and create a container from it.
  - docker load <  yolo_docker_image.tar
  - docker run IMAGE_ID

- Run the new container, open a CMD in Windows and connect to it using "docker exec".
  - docker exec -it IMAGE_ID bash

- Start the ssh server with:
  - service ssh start
  - close the window

- Open XLaunch and press Next or Enter multiple times until the windows closes.

- Open PuTTy and connect to the container, by default 127.0.0.1 and port 2222 but first enable X11 forwarding on the left menu inside SSH/X11 and write ":0.0" inside "X display location".

![image](https://user-images.githubusercontent.com/56027490/145901558-c2650aba-9ba9-4dc7-b74d-dee60981f8fd.png)
![image](https://user-images.githubusercontent.com/56027490/145901606-a70a66ca-f85c-4410-b928-236f953fa5cb.png)

- Insert username "root" and password "wishmaster"

- Navigate to /root/Yolo_CMake/bin

- Finally use ./main to run the application.

## _Inside /opt/ there are three important folders._

  - /opt/samples/videos ==> All the video samples are located here.
  - /opt/configurations ==> Here is the Config.json for the application, you can change the video that is used and the model.
  - /opt/models ==> All the three models used are located here

