#!/usr/bin/env python3
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

if args.verb == "list":
    with open(args.fname, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        print(p.keys())

elif args.verb == "transform":
    print("transform")
    assert args.version
    print(args.fname)
    print(args.version)
else:
    assert False
