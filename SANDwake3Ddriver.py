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
import time
import argparse
import pickle

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
    if params['zlo'] is None: params['zlo'] = params['mesh']['zmin']


    phiinit, Uprof, Vprof, tempT = SANDwake3D.initPhi(params, yvec, zvec)

    dy = params['mesh']['dy']
    dz = params['mesh']['dz']
    Uinit = phiinit['u']

    solveopts = params['solveroptions']
    print(solveopts)

    # March the RANS system
    start_time = time.time()
    phi = SANDwake3D.marchSystem(phiinit, xvec, dy, dz, params, 
                                 SANDwake3D.getTypicalWMBC(Uinit[0,-1], tempT, 
                                                           veerBC=Vprof, dTdz=params['lapserate'],
                                                           uBC_y=Uprof),
                                 SANDwake3D.kepsT_eqns, 
                                 advanceSys=SANDwake3D.advanceSystemKEPS,
                                 **solveopts,
                                 #verbose=2, maxiter=100, tol=1.0E-4,
                                 )
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    savepklfile = params['savepklfile'] 
    if len(savepklfile)>0:
        db={'params':params, 'xvec':xvec, 'phi':phi}
        dbfile = open(savepklfile, 'wb')
        pickle.dump(db, dbfile, protocol=2)
        dbfile.close()
        print(f'Saved to {savepklfile}')


