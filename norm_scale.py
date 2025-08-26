import numpy as np
import argparse
import evo

def main():


    pass


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)

    args = parser.parse_args()


    pose = np.loadtxt(args.file).reshape(-1,3,4)
    t = pose[...,:3,3]

    t_scale = np.sqrt((t*t) )

    pose[...,:3,3] = pose[...,:3,3] / t_scale