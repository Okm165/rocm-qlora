"""Entry point for torchrun workers (multi-GPU DDP).

torchrun needs a *file* to launch; running `python -m rocm_qlora.train`
inside each rank process is equivalent (and used for single-GPU runs).
"""

from rocm_qlora.train import main

if __name__ == "__main__":
    main()
