sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv | tail -1 | tr -d '.')
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH
ninja
sudo ninja install