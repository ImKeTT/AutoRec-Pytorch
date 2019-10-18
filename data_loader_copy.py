import os
import sys
import numpy as np

path_prefix = '../datasets/'

def load_data(dataset='ratings', train_ratio=0.9):
    fname = path_prefix+dataset+'.dat'
    max_uid = 0
    max_vid = 0
    records = []

    if not os.path.exists(fname):
        print('[Error] File %s not found!' % fname)
        sys.exit(-1)

    first_line_flag = True

    with open(fname,encoding = "ISO-8859-1") as f:
        for line in f:
            #user,item,rating,m = line.split()
            tks = line.strip().split('::')#把数据变成一个list
            #tks = m
            if first_line_flag:
                max_uid = int(tks[0])
                max_vid = int(tks[1])
                first_line_flag = False
                continue
            max_uid = max(max_uid, int(tks[0]))
            max_vid = max(max_vid, int(tks[1]))
            records.append((int(tks[0]) - 1, int(tks[1]) - 1, int(tks[2])))
    print("Max user ID {0}. Max item ID {1}. In total {2} ratings.".format(
        max_uid, max_vid, len(records)))
    np.random.shuffle(records)
    train_list = records[0:int(len(records)*train_ratio)]
    test_list = records[int(len(records)*train_ratio):]
    return train_list, test_list, max_uid, max_vid
