
def tensor_to_tuple(tensor):
    return tuple(tensor[i].item() for i in range(tensor.numel()))

