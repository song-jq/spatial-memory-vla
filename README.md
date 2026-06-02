1. 创建环境

  conda create -n spatial-memory-vla python=3.10 -y
  conda activate spatial-memory-vla

  2. 先装 PyTorch
  按你机器的 CUDA 版本选官方源。
  如果是常见的 CUDA 12.1，可以用：

  pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

  如果不是 cu121，把源换成你机器对应版本。

  3. 在仓库根目录做 editable install

  cd /home/data/users/sjq/spatial-memory-vla
  pip install -e .

  当前 spatial-memory-vla/pyproject.toml 里已经包含核心训练依赖，比如：

  - accelerate
  - draccus
  - peft
  - transformers 的 OpenVLA-OFT fork
  - tensorflow / tensorflow_datasets
  - dlimp
  - wandb
  - diffusers

  4. 如果要跑 LIBERO / ALOHA 评测，再补额外依赖
  这部分不在主 pyproject.toml 里，仍然要单独装：

  LIBERO：

  pip install -r /home/data/users/sjq/spatial-memory-vla/experiments/robot/libero/libero_requirements.txt

  ALOHA：

  pip install -r /home/data/users/sjq/spatial-memory-vla/experiments/robot/aloha/requirements_aloha.txt

  5. 可选：装 flash-attn
  如果你需要更高效训练，再单独装：

  pip install flash-attn==2.5.5 --no-build-isolation

  这一步不是必须，失败也不影响基础安装。

  6. 验证安装

  python -c "import prismatic; import torch; print(torch.__version__)"
  python -c "import runpy; runpy.run_path('/home/data/users/sjq/spatial-memory-vla/vla-scripts/finetune.py', run_name='__smoke__')"

  建议
  如果你的目标只是跑当前 memory 训练主线，前 3 步通常就够。
  如果你要我更稳一点，我可以下一步直接把 pyproject.toml 再整理成：

  - core
  - train
  - eval-libero
  - eval-aloha

  这种可选依赖分组版本。
