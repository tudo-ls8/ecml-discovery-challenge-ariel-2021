#!/usr/bin/python3
import numpy as np
import argparse

def load_file(path):
    with open(path) as file:
        data = file.readlines()
    return np.array([ff.split()[0] for ff in data], dtype=float)

def get_score(weights, truth, prediction):
    userdata = load_file(prediction)
    truedata = load_file(truth)
    weights = load_file(weights)
    score = 10000 - np.floor(1000000 * np.sum(weights * 2 * truedata * np.abs(truedata - userdata))/np.sum(weights))
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score predictions like the organizers did during the competition.')
    parser.add_argument('weights', help="Path to weights.txt (provided by the organizers)")
    parser.add_argument('truth', help="Path to truth.txt (provided by the organizers)")
    parser.add_argument('prediction', help="Path to a predictions file")
    args = parser.parse_args()
    score = get_score(args.weights, args.truth, args.prediction)
    print("The score of {} is {}".format(args.prediction, score))
