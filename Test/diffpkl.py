#!/usr/bin/env python3
#

import numpy as np
import pickle
import argparse

import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
curdir = os.getcwd()
extradirs = ['../', '../../',
             '../../SANDwake3D',
             os.path.dirname(curdir),
             scriptpath,
             basepath,
            ]
for x in extradirs: sys.path.insert(1, x)

import SANDwake3D_base

def loadpickle(picklefile):
    pfile          = open(picklefile, 'rb')
    ds             = pickle.load(pfile)
    pfile.close()
    return ds


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    helpstring = """
    Diff two pickle files
    """

    # Handle arguments
    parser     = argparse.ArgumentParser(description=helpstring,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument(
        "pklfile1",
        help="first pickle file",
        type=str,
    )
    parser.add_argument(
        "pklfile2",
        help="second pickle file",
        type=str,
    )
    args      = parser.parse_args()
    pkl1      = args.pklfile1
    pkl2      = args.pklfile2
    xiter     = -1

    # Load the two pickle files
    db1     = loadpickle(pkl1)
    xvec1   = db1['xvec']
    phi1    = db1['phi']

    db2     = loadpickle(pkl2)
    xvec2   = db2['xvec']
    phi2    = db2['phi']
    
    # Compare the xvec's
    diff_xvec = np.linalg.norm(xvec1 - xvec2)
    print("Difference in xvec: %e"%diff_xvec)
    print("Test at x=%f"%xvec1[xiter])
    for v in phi1.keys():
        diffv = np.linalg.norm(phi1[v][xiter,:,:] - phi2[v][xiter,:,:])
        print("diff "+v+": %e"%diffv)
