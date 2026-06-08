docker run --rm -it --gpus all -p 2222:22 \
  -v /home/songjq/spatial-memory-diffusion:/workspace/spatial-memory-diffusion \
  -v /home/data/huggingface:/home/data/huggingface \
  -v /home/data/users/sjq:/home/data/users/sjq \
  spatial-memory-diffusion:v1
