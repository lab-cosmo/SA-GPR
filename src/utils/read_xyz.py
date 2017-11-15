import sys
import numpy as np
from time import time

# Function which returns atom-centered environment parameters for all the species 

def find_neighbours(names,coord,cel,rcut,cweight,fwidth,npoints,sg,periodic,centers):

    start = time()
    nsmax = 26 # max number of species (up to iron)

    # Define a dictionary of atomic valence
    atom_valence =  {"H": 1,"He": 2,"Li": 3,"Be": 4,"B": 5,"C": 6,"N": 7,"O": 8,"F": 9,"Ne": 10,"Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15,"S": 16,"Cl": 17,"Ar": 18,"K": 19,"Ca": 20,"Sc": 21,"Ti": 22,"V": 23,"Cr": 24,"Mn": 25,"Fe": 26} 

    # Define a dictionary of atomic symbols
    atom_symbols = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe'}

    # Process input names
    full_names_list = [name for sublist in names for name in sublist]
    unique_names = list(set(full_names_list))
    nspecies = len(unique_names)

    print "ATOMIC IDENTITIES:", unique_names
    print "SELECTED  CENTERS:", [atom_symbols[i] for i in centers]

    # List all species according to their valence
    all_species = []
    for k in unique_names:
        all_species.append(atom_valence[k]) 

    # List number of atoms for each configuration
    natmax = len(max(names, key=len))
    nat = np.zeros(npoints,dtype=int)
    for i in xrange(npoints):
        nat[i] = len(names[i])

    # List indices for atom of the same species for each configuration
    atom_indexes = [[[] for j in range(nsmax)] for i in range(npoints)]
    for i in xrange(npoints):
        for ii in xrange(nat[i]):
              for j in all_species:
                if names[i][ii] == atom_symbols[j]:
                   atom_indexes[i][j].append(ii)

    if len(centers)==0:
       centers = all_species

    # initialize the variables needed
    nnmax = 40 # maximum number of neighbors
    coords = np.zeros((npoints,natmax,3),              dtype=float)
    nneigh = np.zeros((npoints,natmax,nspecies),       dtype=int  )
    length = np.zeros((npoints,natmax,nspecies,nnmax), dtype=float)
    theta  = np.zeros((npoints,natmax,nspecies,nnmax), dtype=float)
    phi    = np.zeros((npoints,natmax,nspecies,nnmax), dtype=float)
    efact  = np.zeros((npoints,natmax,nspecies,nnmax), dtype=float)

    # initialize coordinates
    for i in xrange(npoints):
        for k in xrange(nat[i]):
            for j in xrange(3):
                coords[i,k,j] = coord[i][k][j]

    # Are we considering a bulk system or a cluster?
    if periodic == True:
        # Bulk system

        nat = np.zeros(npoints,dtype=int)
        ncell = 2  # maximum repetition along the cell vectors
        cell   = np.zeros((npoints,3,3), dtype=float)
        # Loop over configurations
        for i in xrange(npoints):
            for k in xrange(3):
                for j in xrange(3):
                    cell[i,k,j] = cel[i,k,j]
            # Compute cell and inverse to apply PBC further down
            h = cell[i]
            ih = np.linalg.inv(h)
            
            iat = 0
            # Loop over species to center on
            for k in centers:
                nat[i] += len(atom_indexes[i][k])
                # Loop over centers of that species
                for l in atom_indexes[i][k]:
                    # Loop over all the species to use as neighbours 
                    ispe = 0
                    for ix in all_species:
                        n = 0
                        # Loop over neighbours of that species
                        for m in atom_indexes[i][ix]:
                            rr  = coords[i,m] - coords[i,l] # folds atoms in the unit cell
                            # Apply PBC
                            sr = np.dot(ih, rr)
                            sr -= np.round(sr)                                                                    
                            rml = np.dot(h, sr)
                            for ia in xrange(-ncell,ncell+1):
                                for ib in xrange(-ncell,ncell+1):
                                    for ic in xrange(-ncell,ncell+1):
                                        # Automatically build replicated cells
                                        rr = rml + ia*cell[i,0] + ib*cell[i,1] + ic*cell[i,2]

                                        # Is it a neighbour within the spherical cutoff?
                                        if np.linalg.norm(rr) <= rcut:
                                            rr /= (np.sqrt(2.0)*sg)
                                            # Do central atom?
                                            if ia==0 and ib==0 and ic==0 and m==l:
                                                nneigh[i,iat,ispe]      = nneigh[i,iat,ispe] + 1
                                                length[i,iat,ispe,n]    = 0.0                    
                                                theta[i,iat,ispe,n]     = 0.0                                 
                                                phi[i,iat,ispe,n]       = 0.0                      
                                                efact[i,iat,ispe,n]     = cweight
                                                n = n + 1
                                            else:
                                                nneigh[i,iat,ispe]      = nneigh[i,iat,ispe] + 1
                                                length[i,iat,ispe,n]    = np.linalg.norm(rr)
                                                theta[i,iat,ispe,n]     = np.arccos(rr[2]/length[i,iat,ispe,n])
                                                phi[i,iat,ispe,n]       = np.arctan2(rr[1],rr[0])
                                                efact[i,iat,ispe,n]     = np.exp(-0.5*length[i,iat,ispe,n]**2)
                                                n = n + 1
                        ispe += 1
                    iat += 1

        print "Computed neighbors", time()-start        
        return [natmax,nat,nneigh,length,theta,phi,efact,nnmax,nspecies]

    else:
        # Cluster

        nat = np.zeros(npoints,dtype=int)
        # Loop over configurations
        for i in xrange(npoints):
            iat = 0
            # Loop over species to center on
            for k in centers:
                nat[i] += len(atom_indexes[i][k])
                # Loop over centers of that species
                for l in atom_indexes[i][k]:
                    # Loop over all the spcecies to use as neighbours 
                    ispe = 0
                    for ix in all_species:
                        n = 0
                        # Loop over neighbours of that species
                        for m in atom_indexes[i][ix]:
                            rrm = coords[i,m] 
                            rr  = rrm - coords[i,l]
                            # Is it a neighbour within the spherical cutoff?
                            if np.linalg.norm(rr) <= rcut:
                                rr /= (np.sqrt(2.0)*sg)
                                # Do central atom?
                                if m==l:
                                    nneigh[i,iat,ispe]      = nneigh[i,iat,ispe] + 1
                                    length[i,iat,ispe,n]    = 0.0                    
                                    theta[i,iat,ispe,n]     = 0.0                                 
                                    phi[i,iat,ispe,n]       = 0.0                      
                                    efact[i,iat,ispe,n]     = cweight
                                    n = n + 1
                                else:
                                    nneigh[i,iat,ispe]      = nneigh[i,iat,ispe] + 1
                                    length[i,iat,ispe,n]    = np.linalg.norm(rr)
                                    theta[i,iat,ispe,n]     = np.arccos(rr[2]/length[i,iat,ispe,n])
                                    phi[i,iat,ispe,n]       = np.arctan2(rr[1],rr[0])
                                    efact[i,iat,ispe,n]     = np.exp(-0.5*length[i,iat,ispe,n]**2)
                                    n = n + 1
                        ispe += 1
                    iat += 1

        return [natmax,nat,nneigh,length,theta,phi,efact,nnmax,nspecies]


########################################################################################################################

def readftrs(ftrs,vcell):

    nf = len(ftrs)
    all_names = []
    coords = []
    cell = []

    # Read in coordinates file
    for i in range(nf):
        xyz = []
        names = []
        ln = ftrs[i].split()
        if (len(ln)%4 != 0):
            print "Coordinates file incorrectly formatted at line %i!"%i
            sys.exit(0)
        for j in range(len(ln)/4):
            names.append(ln[4*j])
            xyz.append([float(ln[4*j+1]),float(ln[4*j+2]),float(ln[4*j+3])])
        all_names.append(names)
        coords.append(xyz)

    # If applicable, read in cell vector file
    if (len(vcell)>0):
        for i in range(nf):
            ln = vcell[i].split()
            if (len(ln)%3 != 0):
                print "Cell vectors file incorrectly formatted at line %i!"%i
                sys.exit(0)
            a = np.array([float(ln[0]),float(ln[1]),float(ln[2])])
            b = np.array([float(ln[3]),float(ln[4]),float(ln[5])])
            c = np.array([float(ln[6]),float(ln[7]),float(ln[8])])
            cell.append([a,b,c])
     

    return [np.array(coords),np.array(cell),all_names]
