  docker run --rm -it --gpus all -p 2222:22 \
    -v /home/songjq/spatial-memory-vla-buildctx:/workspace/spatial-memory-vla \
    -v /home/data/huggingface:/home/data/huggingface \
    -v /home/data/usrs/sjq:/home/data/users/sjq \
    spatial-memory-vla:latest