#!/usr/bin/env python3

# Add any possible locations of amr-wind-frontend here
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))

curdir = os.getcwd()
extradirs = [scriptpath, 
             os.path.join(scriptpath, 'submodules/inputdicthelper/'),
            ]
for x in extradirs: sys.path.insert(1, x)

import inputdicthelper as idh
import SANDwake3D_keps as SANDwake3D
import SANDwake3D_base as sdb

import argparse

# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    helpstring = """
    Run the SANDwake3D code
    """
    # Load the template inputs
    inputs    = idh.inputdict(SANDwake3D.RANSinput, globalhelp="SANDwake3D inputs")
        
    # Handle arguments
    parser     = argparse.ArgumentParser(description=helpstring,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument(
        "inputfile",
        help="input yaml file",
        type=str,
    )
    parser.add_argument(
        '--printinputs',
        help="print the yaml inputs",
        default=False,
        action='store_true',
    )
        
    args      = parser.parse_args()
    inputfile = args.inputfile

    # Dump inputs if requested
    if args.printinputs:
        inputs.dumpyaml(sys.stdout)
        sys.exit(0)

    # load the input yaml file
    params = inputs.ingestyaml(inputfile, checkunused=False)

    # make the mesh
    params['ym'], params['zm'], xvec, yvec, zvec = SANDwake3D.makemesh(params['mesh'])
    Ny = len(yvec)
    Nz = len(zvec)

    
    print(yvec)
    print(zvec)
    print(params)
