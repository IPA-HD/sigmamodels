import glob
import json
import os
import pandas as pd
import re
import jax
import equinox as eqx

PATH16 = os.path.abspath(os.path.dirname(__file__) + "/../../artifacts/models16")
PATH128 = os.path.abspath(os.path.dirname(__file__) + "/../../artifacts/models128")
PATH3 = os.path.abspath(os.path.dirname(__file__) + "/../../artifacts/nmodels16")
PATHEX5 = os.path.abspath(os.path.dirname(__file__) + "/../../artifacts/ex5")
PATHS = {16: PATH16, 128: PATH128, 3: PATH3, 5: PATHEX5}


def summarize(path):
    ls = [
        re.split(f"{path}/(.*?)/(.*?)/noise(0.\\d+)_alpha_(.*)_epoch(\\d+)", x)
        for x in glob.glob(path + "/**/**/*.csv")
    ]
    ls = [l[1:-1] for l in ls]
    df = pd.DataFrame(
        ls,
        columns=[
            "date",
            "time",
            "noise",
            "alpha",
            "epoch",
        ],
    )
    ls = [
        re.split(f"{path}/(.*?)/(.*?)/epoch(\\d+)", x)
        for x in glob.glob(path + "/**/**/*.eqx")
    ]
    ls = [l[1:-1] for l in ls]
    dg = pd.DataFrame(ls, columns=["date", "time", "epoch"])
    df = dg.merge(df)
    res = []
    ks = []
    dim = []
    conv_dim = []
    git = []
    for ls in df.iterrows():
        l = ls[1]
        p = f"{path}/{l['date']}/{l['time']}/noise{l['noise']}_alpha_{l['alpha']}_epoch{l['epoch']}_loss.csv"
        try:
            res += pd.read_csv(p)[-100:].mean().to_list()
        except:
            res += [pd.NA]
        try:
            with open(f"{path}/{l['date']}/{l['time']}/hps.json") as f:
                hps = json.load(f)
            ks.append(hps["ks"])
            dim.append(hps["dim"])
            conv_dim.append(hps["conv_dim"])
            git.append(hps["git"])
        except Exception as e:
            print(f"{e} was wrong")
    df["mean"] = res
    try:
        df["kernel_size"] = ks
        df["latent"] = conv_dim
        df["git"] = git
    except Exception as e:
        print(f"{e} was wrong")
    return df


def id_to_path(path, idx):
    path = PATHS[int(path)]
    df = pd.read_csv(path + "/summary.csv")
    l = df.loc[idx]
    p = f"{path}/{l['date']}/{l['time']}/"
    return p


def load_summary(path):
    path = PATHS[int(path)]
    return pd.read_csv(path + "/summary.csv", index_col=0).sort_values("mean")


def ps(path):
    print(load_summary(path))


def update(path):
    path = PATHS[int(path)]
    summarize(path).to_csv(path + "/summary.csv")
    # print(summarize(path))


def insert(dim, row):
    path = PATHS[int(dim)]
    df = load_summary(dim).sort_index()
    row = pd.DataFrame([row], columns=df.columns)
    df = pd.concat([df, row], ignore_index=True)
    # df.iloc[-1] = row
    # df.index = df.index + 1
    print(df)
    df.to_csv(path + "/summary.csv")


if __name__ == "__main__":
    import sys

    globals()[sys.argv[1]](sys.argv[2])

if __name__ == "__test__":
    exec("from handler import *")
    path = PATH128

    ls = [
        re.split(f"{path}/(.*?)/(.*?)/noise(0.\\d+)_alpha_(0.\\d+)_epoch(\\d+)", x)[1:3]
        for x in glob.glob(path + "/**/**/*.csv")
    ]
    ls

    df = load_summary(128)
    df[["date", "time"]]

    [l for l in ls if (df[["time", "date"]] == l[0]).any()]

    ls = glob.glob(path + "/**/**/*.json")
    for l in ls:
        with open(l) as f:
            dct = json.load(f)
            if "git" not in dct.keys():
                dct["git"] = "d85b06a"
        with open(l, "w") as f:
            json.dump(dct, f)

    with open(
        "/Users/jonas/coding/sigmaflow/artifacts/models128/18_08/18_41/noise0.1_alpha_0.8_epoch0_loss.csv"
    ) as f:
        xx = f.read().split()
        sum(float(x) for x in xx) / len(xx)
    pd.read_csv(
        "/Users/jonas/coding/sigmaflow/artifacts/models128/18_08/18_41/noise0.1_alpha_0.8_epoch0_loss.csv"
    ).mean().item()
