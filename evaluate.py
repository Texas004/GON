import os
import torch
import shelve
import numpy as np


from dataset import EITDataset
from eit_operator import EITOperator
from lpd_model import LPDNet
from visualize import save_tripcolor



# ==========================
# paths
# ==========================


DATA_PATH = "./processed/eit_dataset.npz"


HYPERS_PATH = (
r"F:\\VSCode\\Code\\Python\\DuNets-RMA-LPD"
r"\\datasets-600\\circle4_h00.07_Windows\\hypers.db"
)


MODEL_PATH="./checkpoints/best_lpd.pth"


RESULT_DIR="./results"


os.makedirs(
    RESULT_DIR,
    exist_ok=True
)




# ==========================
# evaluate
# ==========================


def main():


    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "device:",
        device
    )



    # ----------------------
    # load mesh
    # ----------------------


    db=shelve.open(
        HYPERS_PATH
    )


    mesh=db["data"]["mesh_obj"]


    db.close()



    # ----------------------
    # data
    # ----------------------


    dataset=EITDataset(
        DATA_PATH
    )


    sample=dataset[0]


    x0=sample["x0"].unsqueeze(0)

    y=sample["y"].unsqueeze(0)

    target=sample["target"]



    x0=x0.to(device)

    y=y.to(device)



    # ----------------------
    # model
    # ----------------------


    operator=EITOperator(
        HYPERS_PATH
    )


    operator.to(device)



    model=LPDNet(
        operator,
        num_iter=5
    )


    checkpoint=torch.load(
        MODEL_PATH,
        map_location=device
    )


    model.load_state_dict(
        checkpoint["model"]
    )


    model.to(device)


    model.eval()



    # ----------------------
    # predict
    # ----------------------


    with torch.no_grad():

        pred=model(
            x0,
            y
        )


    pred=pred.cpu().numpy()[0]


    gn=x0.cpu().numpy()[0]


    target=target.numpy()



    # ----------------------
    # error
    # ----------------------


    gn_error=(
        np.linalg.norm(
            gn-target
        )
        /
        np.linalg.norm(
            target
        )
    )


    lpd_error=(
        np.linalg.norm(
            pred-target
        )
        /
        np.linalg.norm(
            target
        )
    )



    print(
        "GN relative error:",
        gn_error
    )


    print(
        "LPD relative error:",
        lpd_error
    )



    # ----------------------
    # save images
    # ----------------------


    save_tripcolor(

        mesh,

        target,

        os.path.join(
            RESULT_DIR,
            "ground_truth.png"
        ),

        "Ground Truth"

    )



    save_tripcolor(

        mesh,

        gn,

        os.path.join(
            RESULT_DIR,
            "gn.png"
        ),

        "GN reconstruction"

    )



    save_tripcolor(

        mesh,

        pred,

        os.path.join(
            RESULT_DIR,
            "lpd.png"
        ),

        "LPD reconstruction"

    )



    print(
        "saved results."
    )



if __name__=="__main__":

    main()