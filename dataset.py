import numpy as np
import torch
from torch.utils.data import Dataset


class EITDataset(Dataset):

    def __init__(self, npz_path):

        data = np.load(
            npz_path
        )

        self.x = data["xs"].astype(
            np.float32
        )

        self.y = data["ys"].astype(
            np.float32
        )

        self.x0 = data["xs_gn"].astype(
            np.float32
        )


    def __len__(self):

        return len(self.x)



    def __getitem__(self,index):

        return {

            "x0":
            torch.from_numpy(
                self.x0[index]
            ),

            "y":
            torch.from_numpy(
                self.y[index]
            ),

            "target":
            torch.from_numpy(
                self.x[index]
            )
        }