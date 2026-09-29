FROM python:3.12-slim

RUN apt update
RUN apt install -y dclock python3-tk libgl1 libglib2.0-0



RUN pip install --no-cache-dir "tensorflow[and-cuda]"

# pip로 설치된 NVIDIA CUDA/cuDNN 공유 라이브러리(site-packages/nvidia/*/lib)는
# 시스템 ld.so 검색 경로에 없어 TensorFlow가 dlopen에 실패한다.
# ldconfig에 경로를 등록해 빌드 시점에 영구적으로 해결한다.
RUN python3 -c "\
import os, pathlib, nvidia; \
base = pathlib.Path(nvidia.__file__).parent; \
paths = [str(p) for p in base.glob('*/lib') if p.is_dir()]; \
open('/etc/ld.so.conf.d/nvidia-pip.conf', 'w').write('\n'.join(paths) + '\n')" \
    && ldconfig


RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install tqdm
RUN pip install opencv-python
RUN pip install gymnasium
RUN pip install "gymnasium[classic-control]"
RUN pip install "gymnasium[atari]" ale-py autorom
RUN AutoROM --accept-license

CMD ["/bin/bash"]
