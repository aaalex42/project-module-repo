"""
Reference: https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
"""


import os
import torch
from torch.autograd import Variable


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

"""CHECK IF THOSE ARE NECESSARY"""
def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))


def to_numpy(tensor):
    return tensor.cpu().data.numpy() if USE_CUDA else tensor.data.numpy()


def to_tensor(ndarray, volatile = False, requires_grad = False, dtype = FLOAT):
    """
    volatile is left out since it has no effect.
    """
    return Variable(
        torch.from_numpy(ndarray), requires_grad = requires_grad
    ).type(dtype)


# This function implements a form of smoothing or blending of the target network's 
# parameters towards the parameters of the source network.
# It is used for stabilize training and prevent sudden, drastic changes in the target network.
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_output_folder(parent_dir, env_name):
    """
    This function finds a folder with the highest run number and sets the output folder to that number + 1.
    It assumes that all folders in parent_dir are named by the convention "run_{run_no}".
    It has the advantage that it is possible to plot the results of multiple runs with the same code.

    Input: parent_dir (str) - path to parent directory
           env_name   (str) - name of environment
    
    Returns: parent_dir/run_dir - path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok = True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split("-run")[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass

    experiment_id +=1
    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + "-run{}".format(experiment_id)
    os.makedirs(parent_dir, exist_ok = True)
    return parent_dir