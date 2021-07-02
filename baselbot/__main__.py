"""Train and submit daily."""
import argparse
import numpy as np
from datetime import datetime
from selenium.common.exceptions import InvalidArgumentException
from .trainpredict import trainpredict
from .submit import submit, get_key

def main(args):
    PREDICT_PATH = "/mnt/data/baselbot/pred.csv"
    password = get_key(args.key)

    if not args.no_submit_before_training:
        first_submit(args.url, args.user, password, PREDICT_PATH)

    while True:
        daily_trainpredict(PREDICT_PATH)
        daily_submit(args.url, args.user, password, PREDICT_PATH)

def daily_trainpredict(predict_path):
    t = datetime.now()
    daily_seed = 1000000*t.month + 10000*t.day + 100*t.hour + t.minute
    np.random.seed(daily_seed)
    daily_split = .05 + np.random.rand() * .1
    print(f"Starting DAILY trainpredict with seed {daily_seed} and split {daily_split}")
    trainpredict(predict=predict_path, seed=daily_seed, new_test_split=daily_split)

def daily_submit(url, user, password, predict_path):
    submit(url, user, password, predict_path+"_45", wait=True)

def first_submit(*args, **kwargs):
    try:
        daily_submit(*args, **kwargs)
    except InvalidArgumentException as x:
        print(f"WARNING: first upload failed - {x}")

# __main__
parser = argparse.ArgumentParser(description='Train and submit daily; start by submitting.', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--url', default=None, help="Optional remote driver, e.g. 'http://localhost:4444/wd/hub'") 
parser.add_argument('--user', default="TheReturnOfBasel321", help="the name of the user to log in as")
parser.add_argument('--key', default="/mnt/home/.BASELBOT", help='the file to read the user\'s password from')
parser.add_argument('--no-submit-before-training', action="store_true", help='whether to skip the first submission trial before training')
args = parser.parse_args()
main(args)
