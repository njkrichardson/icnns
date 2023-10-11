# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install --yes \
    build-essential \
    gdb \
    zsh \
    wget \
    curl \
    git \
    ninja-build \
    gettext \
    cmake \
    unzip \
    parallel \
    npm \
    python3-pip \
    vim \
    nvtop \
    g++ \
    direnv \
    python3-venv

# install Python dependencies 
COPY ./requirements.txt /requirements.txt
RUN python3 -m pip install -r requirements.txt

# install zsh 
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t af-magic \
    -p git \
    -p vi-mode 

# configure zsh 
RUN cd / && git clone https://github.com/njkrichardson/dots.git && parallel cp dots/.zshrc ::: /.zshrc ~/.zshrc && echo "export XDG_CONFIG_HOME=/.config" >> /.zshrc && echo "export XDG_CONFIG_HOME=/.config" >> ~/.zshrc
RUN echo "eval \"\$(direnv hook zsh)\"" >> ~/.zshrc 

# install neovim 
RUN git clone https://github.com/neovim/neovim && cd neovim && git checkout stable && make CMAKE_BUILD_TYPE=RelWithDebInfo && make install 

# install packer 
RUN git clone --depth 1 https://github.com/wbthomason/packer.nvim ~/.local/share/nvim/site/pack/packer/start/packer.nvim

# configure neovim
RUN mkdir /.config && cd / && git clone https://github.com/njkrichardson/nvimconfig.git && cp -r nvimconfig /.config/nvim 

CMD ["/bin/zsh"]
