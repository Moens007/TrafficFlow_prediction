import random
import torch
import numpy as np

def set_seed(self, seed=32):
     random.seed(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.benchmark = True
     torch.backends.cudnn.deterministic = True