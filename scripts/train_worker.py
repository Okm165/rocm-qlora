"""Entry point for torchrun workers (multi-GPU DDP).

torchrun needs a *file* to launch; running `python -m llm_finetune.train`
inside each rank process is equivalent (and used for single-GPU runs).
"""

from llm_finetune.train import main

if __name__ == "__main__":
    main()
