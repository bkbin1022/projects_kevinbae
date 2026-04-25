# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
import pandas as pd
# MAIN DRIVER
def main(nelx,nely,volfrac,penal,rmin,ft):
	print("Minimum compliance problem with OC")
	print("ndes: " + str(nelx) + " x " + str(nely))
	print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
	print("Filter method: " + ["Sensitivity based","Density based"][ft])
	# Max and min stiffness
	Emin=1e-9
	Emax=1.0
	# dofs:
	ndof = 2*(nelx+1)*(nely+1)
	# Allocate design variables (as array), initialize and allocate sens.
	x=volfrac * np.ones(nely*nelx,dtype=float)

	# --- AIRFOIL MASKING SECTION ---
    # 1. Define the coordinates of your airfoil boundary (NACA 2412 example)
    # You can load your XFOIL coordinates here instead
	t = 0.12 # 12% thickness
	x_coords = np.linspace(0, 1, nelx)
    # Basic NACA 4-digit thickness formula
	y_upper = 5 * t * (0.2969*np.sqrt(x_coords) - 0.1260*x_coords - 0.3516*x_coords**2 + 0.2843*x_coords**3 - 0.1015*x_coords**4)
    
    # 2. Identify "Passive" (Outside) elements
	passive = []
	scale_height = 3.0
	for i in range(nelx):
        # Scale y_upper to mesh coordinates (nely)
        # Center the airfoil vertically in the grid
		upper_limit = (nely/2) + (y_upper[i] * nely * scale_height)
		lower_limit = (nely/2) - (y_upper[i] * nely * scale_height)
        
		for j in range(nely):
            # If the pixel is outside the thickness envelope
			if j < lower_limit or j > upper_limit:
				passive.append(i * nely + j)

    # 3. Apply the mask to the initial design
	x[passive] = 0.001 # Force outside pixels to be "void"
    # -------------------------------
	
	xold=x.copy()
	xPhys=x.copy()
	g=0 # must be initialized to use the NGuyen/Paulino OC approach
	dc=np.zeros((nely,nelx), dtype=float)
	# FE: Build the index vectors for the for coo matrix format.
	KE=lk()
	edofMat=np.zeros((nelx*nely,8),dtype=int)
	for elx in range(nelx):
		for ely in range(nely):
			el = ely+elx*nely
			n1=(nely+1)*elx+ely
			n2=(nely+1)*(elx+1)+ely
			edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
	# Construct the index pointers for the coo format
	iK = np.kron(edofMat,np.ones((8,1))).flatten()
	jK = np.kron(edofMat,np.ones((1,8))).flatten()    
	# Filter: Build (and assemble) the index+data vectors for the coo matrix format
	nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
	iH = np.zeros(nfilter)
	jH = np.zeros(nfilter)
	sH = np.zeros(nfilter)
	cc=0
	for i in range(nelx):
		for j in range(nely):
			row=i*nely+j
			kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
			kk2=int(np.minimum(i+np.ceil(rmin),nelx))
			ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
			ll2=int(np.minimum(j+np.ceil(rmin),nely))
			for k in range(kk1,kk2):
				for l in range(ll1,ll2):
					col=k*nely+l
					fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
					iH[cc]=row
					jH[cc]=col
					sH[cc]=np.maximum(0.0,fac)
					cc=cc+1
	# Finalize assembly and convert to csc format
	H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()	
	Hs=H.sum(1)
# BC's and support (Bolting the rib to the Spars)
	dofs=np.arange(2*(nelx+1)*(nely+1))
    
    # Define Front Spar at 25% chord and Rear Spar at 75% chord
	spar_front_x = int(0.25 * nelx)
	spar_rear_x = int(0.75 * nelx)
    
    # Fix the nodes at the center of the airfoil at these spar locations
	n_front = (nely + 1) * spar_front_x + int(nely/2)
	n_rear = (nely + 1) * spar_rear_x + int(nely/2)
    
	fixed = np.array([2*n_front, 2*n_front+1, 2*n_rear, 2*n_rear+1])
	free = np.setdiff1d(dofs,fixed)

    # Solution and RHS vectors
	f=np.zeros((ndof,1))
	u=np.zeros((ndof,1))
    
    # --- CP LOAD UNWRAPPING & MAPPING SECTION ---
    # 1. Load the CSV
	df = pd.read_csv('datafiles/cp_dataset.csv')
    
    # 2. Select the row you want to optimize for (e.g., the first row)
	row = df.iloc[0] 
    
    # Extract the 200 values into a numpy array
	cp_raw = np.array([row[f'cp_{k}'] for k in range(200)])
    
    # 3. Find the Leading Edge (stagnation point is the max pressure)
	le_idx = np.argmax(cp_raw)
    
    # 4. Split into Upper and Lower surfaces
    # Upper goes TE->LE, so we reverse it ([::-1]) to go LE->TE (x=0 to x=1)
	cp_upper_raw = cp_raw[:le_idx+1][::-1] 
    # Lower already goes LE->TE
	cp_lower_raw = cp_raw[le_idx:]         
    
    # 5. Interpolate exactly to the resolution of your grid (nelx)
	cp_upper = np.interp(np.linspace(0, 1, nelx), np.linspace(0, 1, len(cp_upper_raw)), cp_upper_raw)
	cp_lower = np.interp(np.linspace(0, 1, nelx), np.linspace(0, 1, len(cp_lower_raw)), cp_lower_raw)

    # Normalizer to keep math stable (Adjust if obj is > 10,000)
	scale = 1e-1

    # 6. Apply forces to the mesh
	for i in range(nelx):
        # Find active elements in this vertical slice
		active_in_col = [j for j in range(nely) if (i * nely + j) not in passive]
        
		if len(active_in_col) > 0:
			top_node_idx = (nely + 1) * i + min(active_in_col)
			bot_node_idx = (nely + 1) * i + max(active_in_col) + 1
            
            # Apply vertical forces (positive Y is UP)
            # -cp_upper turns negative suction into upward force
			f[2 * top_node_idx + 1, 0] = -cp_upper[i] * scale
			f[2 * bot_node_idx + 1, 0] = -cp_lower[i] * scale
    # ---------------------------------------------
	# Initialize plot and plot the initial design
	plt.ion() # Ensure that redrawing is possible
	fig,ax = plt.subplots()
	im = ax.imshow(-xPhys.reshape((nelx,nely)).T, cmap='gray',\
	interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
	fig.show()
   	# Set loop counter and gradient vectors 
	loop=0
	change=1
	dv = np.ones(nely*nelx)
	dc = np.ones(nely*nelx)
	ce = np.ones(nely*nelx)
	while change>0.001 and loop<300: # Change change rate to stop
		loop=loop+1
		boundary_elements = []
		for i in range(nelx):
			active_in_col = [j for j in range(nely) if (i * nely + j) not in passive]
			if len(active_in_col) > 0:
				# Add the top and bottom pixels of the rib to a 'force-solid' list
				boundary_elements.append(i * nely + min(active_in_col))
				boundary_elements.append(i * nely + max(active_in_col))
		# Setup and solve FE problem
		sK=((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
		K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
		# Remove constrained dofs from matrix
		K = K[free,:][:,free]
		# Solve system 
		u[free,0]=spsolve(K,f[free,0])    
		# Objective and sensitivity
		ce[:] = (np.dot(u[edofMat].reshape(nelx*nely,8),KE) * u[edofMat].reshape(nelx*nely,8) ).sum(1)
		obj=( (Emin+xPhys**penal*(Emax-Emin))*ce ).sum()
		dc[:]=(-penal*xPhys**(penal-1)*(Emax-Emin))*ce
		dv[:] = np.ones(nely*nelx)
		# Sensitivity filtering:
		if ft==0:
			dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
		elif ft==1:
			dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
			dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
		# Optimality criteria
		xold[:]=x
		(x[:],g)=oc(nelx,nely,x,volfrac,dc,dv,g)
		x[passive] = 0.001
		# Filter design variables
		if ft==0:   xPhys[:]=x
		elif ft==1:	xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
		# Compute the change by the inf. norm
		change=np.linalg.norm(x.reshape(nelx*nely,1)-xold.reshape(nelx*nely,1),np.inf)
		# Plot to screen
		im.set_array(-xPhys.reshape((nelx,nely)).T)
		fig.canvas.draw()
		# Write iteration history to screen (req. Python 2.6 or newer)
		print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(\
					loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
	# Make sure the plot stays and that the shell remains	
	plt.show()
	plt.savefig('my_plot.png')
	input("Press any key...")
#element stiffness matrix
def lk():
	E=1
	nu=0.3
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
	[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
	[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
	[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
	[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
	[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
	[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
	[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
	return (KE)
# Optimality criterion
def oc(nelx,nely,x,volfrac,dc,dv,g):
	l1=0
	l2=1e9
	move=0.1
	# reshape to perform vector operations
	xnew=np.zeros(nelx*nely)
	while (l2-l1)/(l1+l2)>1e-3:
		lmid=0.5*(l2+l1)
		xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
		gt=g+np.sum((dv*(xnew-x)))
		if gt>0 :
			l1=lmid
		else:
			l2=lmid
	return (xnew,gt)
# The real main driver    
if __name__ == "__main__":
	# Default input parameters
	nelx=200
	nely=60
	volfrac=0.15
	rmin=3.0
	penal=3.0
	ft=1 # ft==0 -> sens, ft==1 -> dens
	import sys
	if len(sys.argv)>1: nelx   =int(sys.argv[1])
	if len(sys.argv)>2: nely   =int(sys.argv[2])
	if len(sys.argv)>3: volfrac=float(sys.argv[3])
	if len(sys.argv)>4: rmin   =float(sys.argv[4])
	if len(sys.argv)>5: penal  =float(sys.argv[5])
	if len(sys.argv)>6: ft     =int(sys.argv[6])
	main(nelx,nely,volfrac,penal,rmin,ft)
