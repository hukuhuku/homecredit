import pandas as pd
import numpy as np

from base import *
from base import Feature,Data


if __name__ == '__main__':
    args = get_arguments()

    #train = pd.read_csv('input/train.csv').drop(["ID","target"],axis=1)
    #test = pd.read_csv('input/test.csv').drop("ID",axis=1)
  


    generate_features(globals(), args.force)


