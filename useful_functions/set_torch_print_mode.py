import torch

def change_torch_print_opts(precision=2, sci_mode=False):
    try:
        torch.set_printoptions(precision=precision, sci_mode=sci_mode)
        return True
    except Exception as e:
        print(e)
        return False