import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import EITDataset
from eit_operator import EITOperator
from lpd_model import LPDNet



# =========================
# 路径设置
# =========================

DATA_PATH = "./processed/eit_dataset.npz"


HYPERS_PATH = (
    r"F:\VSCode\Code\Python\DuNets-RMA-main"
    r"\datasets-600\circle4_h00.07_Windows\hypers.db"
)


SAVE_DIR = "./checkpoints"


os.makedirs(
    SAVE_DIR,
    exist_ok=True
)



# =========================
# 超参数
# =========================

EPOCHS = 10

BATCH_SIZE = 20

LR = 1e-4

NUM_ITER = 5




def main():


    # device

    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "device:",
        device
    )



    # dataset


    dataset=EITDataset(
        DATA_PATH
    )


    loader=DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True

    )



    # operator


    operator=EITOperator(
        HYPERS_PATH
    )


    operator.to(device)



    # model


    model=LPDNet(
        operator,
        num_iter=NUM_ITER
    )


    model=model.to(device)



    # loss


    criterion=nn.MSELoss()



    optimizer=torch.optim.Adam(

        model.parameters(),

        lr=LR

    )



    best_loss=float("inf")



    # =========================
    # training loop
    # =========================


    for epoch in range(EPOCHS):


        model.train()


        total_loss=0



        for batch in loader:


            x0=batch["x0"].to(device)

            y=batch["y"].to(device)

            target=batch["target"].to(device)



            pred=model(
                x0,
                y
            )



            loss=criterion(
                pred,
                target
            )



            optimizer.zero_grad()


            loss.backward()


            optimizer.step()



            total_loss += loss.item()



        avg_loss = (
            total_loss
            /
            len(loader)
        )


        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss={avg_loss:.6e}"
        )



        # save best


        if avg_loss < best_loss:


            best_loss=avg_loss


            torch.save(

                {

                "epoch":epoch,

                "model":
                    model.state_dict(),

                "loss":
                    best_loss

                },

                os.path.join(
                    SAVE_DIR,
                    "best_lpd.pth"
                )

            )


            print(
                "saved best model"
            )





if __name__=="__main__":

    main()