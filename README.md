# Yolox_Implementation
Implementation of various Yolox models in C++.

# Requisites needed for implementation

- Docker.
- Ubuntu latest image in a running Docker container.
- XLauch / XMing installed in host machine.
- Visual Studio or other IDE.
- PuTTy configured to use X11 over SSH.

## Needed inside the Ubuntu container

- OpenVINO latest version.
- OpenSSH.
- x11-common.
- Net-tools.
- cpio.
- Configure .bashrc
- Configure ssh.config and sshd.config to allow use of X11.
- Configure /etc/X11
