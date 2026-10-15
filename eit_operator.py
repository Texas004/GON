import os
import shelve
import numpy as np
import torch


class EITOperator:


    def __init__(self, hypers_path):


        db=shelve.open(
            hypers_path
        )


        fun=db["fun"]

        self.eit=fun["eit"]


        db.close()


        self.J=torch.tensor(
            self.eit.J,
            dtype=torch.float32
        )

    def to(self,device):

        self.J=self.J.to(device)

        return self


    def forward(self,x):

        """
        x:
        (batch,1422)

        return:
        (batch,208)
        """

        return torch.matmul(
            x,
            self.J.T
        )



    def adjoint(self,y):

        """
        y:
        (batch,208)

        return:
        (batch,1422)
        """

        return torch.matmul(
            y,
            self.J
        )