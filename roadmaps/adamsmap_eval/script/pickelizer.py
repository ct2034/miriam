#!/usr/bin/env python3
import os
import sys
import pickle
import argparse

verbs = ["list", "transform"]
parser = argparse.ArgumentParser(description='Fun with some pickle files.')
parser.add_argument('verb', type=str,
                    help='a subcommand', choices=verbs)
parser.add_argument('fname', type=str,
                    help='file to open')
parser.add_argument('-v', dest='version',
                    help='which version to transform to')
parser.add_argument('-o', dest='outfile', type=str,
                    help='which file to write to')

args = parser.parse_args()
read_fname = args.fname
write_tmp_file = args.fname + ".proto2"
write_orig_file = args.fname + ".orig"

with open(read_fname, 'rb') as f_in:
    unpickler = pickle._Unpickler(f_in)
    unpickler.encoding = 'latin1'
    data = unpickler.load()

    if args.verb == "list":
        print(data.keys())
        print("proto: " + str(unpickler.proto))

    elif args.verb == "transform":
        if unpickler.proto >= 3:
            print("proto: " + str(unpickler.proto) + ">= 3")
            print("-> transforming to 2")
        with open(write_tmp_file, 'wb') as f_out:
            pickler = pickle.Pickler(f_out, protocol=2)
            pickler.dump(data)
        f_in.close()
        os.rename(read_fname, write_orig_file)
        os.rename(write_tmp_file, read_fname)

    else:
        assert False
