import torch
import torch.nn as nn
import torch.nn.functional as F



# =====================================================
# Dual update block
# =====================================================

class DualBlock(nn.Module):

    def __init__(
            self,
            input_dim=208,
            hidden_dim=512
    ):

        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                input_dim
            )

        )



    def forward(
            self,
            y,
            Ax
    ):

        z = y - Ax


        return self.net(z)




# =====================================================
# Primal correction block
# =====================================================


class PrimalBlock(nn.Module):

    def __init__(
            self,
            input_dim=2844,
            output_dim=1422,
            hidden_dim=1024
    ):

        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),


            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),


            nn.Linear(
                hidden_dim,
                output_dim
            )

        )



    def forward(
            self,
            x,
            grad
    ):


        # correction term

        z = torch.cat(
            [
                x,
                grad
            ],
            dim=1
        )


        return self.net(z)




# =====================================================
# LPD Network
# =====================================================


class LPDNet(nn.Module):


    def __init__(
            self,
            operator,
            num_iter=5
    ):


        super().__init__()


        self.operator = operator


        self.num_iter=num_iter



        # 注意：
        # 这里暂时不做参数共享
        # 下一版根据论文修改


        self.dual_blocks = nn.ModuleList(

            [

                DualBlock()

                for _ in range(num_iter)

            ]

        )


        self.primal_blocks = nn.ModuleList(

            [

                PrimalBlock()

                for _ in range(num_iter)

            ]

        )



        # learnable step size

        self.tau = nn.ParameterList(

            [

                nn.Parameter(
                    torch.tensor(0.1)
                )

                for _ in range(num_iter)

            ]

        )





    def forward(
            self,
            x0,
            y
    ):


        x=x0



        for k in range(self.num_iter):


            # forward operator

            Ax=self.operator.forward(x)



            # dual update

            u=self.dual_blocks[k](
                y,
                Ax
            )



            # adjoint

            JT_u=self.operator.adjoint(
                u
            )



            # primal correction

            dx=self.primal_blocks[k](

                x,

                JT_u

            )


            # learned step

            x = x + self.tau[k]*dx



        return x