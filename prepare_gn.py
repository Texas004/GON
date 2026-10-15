import os
import numpy as np
import shelve
from tqdm import tqdm

from config import DATA_DIR, PROCESSED_DIR



def load_hyper():

    db_path = os.path.join(
        DATA_DIR,
        "hypers.db"
    )

    db = shelve.open(db_path)

    fun = db["fun"]

    db.close()

    return fun



def process_one(npz_path, eit):

    data = np.load(npz_path)


    xs = data["xs"]

    ys = data["ys"]


    gn = []


    for i in range(len(ys)):

        voltage = ys[i]


        # pyEIT 1.2.4:
        # gn() directly returns ndarray

        x_gn = eit.gn(
            voltage,
            lamb_decay=0.1,
            lamb_min=1e-5,
            maxiter=8,
            verbose=False
        )


        x_gn = np.asarray(
            x_gn
        ).reshape(-1)


        gn.append(
            x_gn
        )


    gn = np.asarray(gn)


    return xs, ys, gn




def main():


    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )


    fun = load_hyper()


    eit = fun["eit"]


    files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".npz")
    ]


    files.sort(
        key=lambda x:int(
            x.split(".")[0]
        )
    )


    all_x = []

    all_y = []

    all_gn = []



    for f in tqdm(files):


        path = os.path.join(
            DATA_DIR,
            f
        )


        xs, ys, gn = process_one(
            path,
            eit
        )


        all_x.append(xs)

        all_y.append(ys)

        all_gn.append(gn)



    all_x = np.concatenate(
        all_x,
        axis=0
    )


    all_y = np.concatenate(
        all_y,
        axis=0
    )


    all_gn = np.concatenate(
        all_gn,
        axis=0
    )


    print(
        "x:",
        all_x.shape
    )

    print(
        "y:",
        all_y.shape
    )

    print(
        "gn:",
        all_gn.shape
    )



    np.savez(
        os.path.join(
            PROCESSED_DIR,
            "eit_dataset.npz"
        ),

        xs=all_x,

        ys=all_y,

        xs_gn=all_gn
    )




if __name__=="__main__":

    main()