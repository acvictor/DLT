import numpy as N

def Normalization(nd, x):
    '''
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

    Inputs:
     nd: number of dimensions (2 for 2D; 3 for 3D)
     x: the data to be normalized (directions at different columns and points at rows)
    Outputs:
     Tr: the transformation matrix (translation plus scaling)
     x: the transformed data
    '''

    x = N.asarray(x)
    m, s = N.mean(x,0), N.std(x)
    if nd==2:
        Tr = N.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = N.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
        
    Tr = N.linalg.inv(Tr)
    x = N.dot( Tr, N.concatenate( (x.T, N.ones((1,x.shape[0]))) ) )
    x = x[0:nd,:].T

    return Tr, x


def DLTcalib(nd, xyz, uv):
    '''
    Camera calibration by DLT using known object points and their image points.

    Input
    -----
    nd: dimensions of the object space, 3 here.
    xyz: coordinates in the object 3D space.
    uv: coordinates in the image 2D space.

    The coordinates (x,y,z and u,v) are given as columns and the different points as rows.

    There must be at least 6 calibration points for the 3D DLT.

    Output
    ------
     L: array of 11 parameters of the calibration matrix.
     err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    '''
    if (nd != 3):
        raise ValueError('%dD DLT unsupported.' %(nd))
    
    #Converting all variables to numpy array
    xyz = N.asarray(xyz)
    uv = N.asarray(uv)

    np = xyz.shape[0]

    #Validating the parameters:
    if uv.shape[0] != np:
        raise ValueError('Object (%d points) and image (%d points) have different number of points.' %(np, uv.shape[0]))

    if (xyz.shape[1] != 3):
        raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' %(xyz.shape[1],nd,nd))

    if (np < 6):
        raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.' %(nd, 2*nd, np))
        
    #Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
    #This is relevant when there is a considerable perspective distortion.
    #Normalization: mean position at origin and mean distance equals to 1 at each direction.
    Txyz, xyzn = Normalization(nd, xyz)
    Tuv, uvn = Normalization(2, uv)

    A = []

    for i in range(np):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = uvn[i, 0], uvn[i, 1]
        A.append( [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u] )
        A.append( [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v] )

    # Convert A to array
    A = N.asarray(A) 

    # Find the 11 parameters:
    U, S, V = N.linalg.svd(A)

    # The parameters are in the last line of Vh and normalize them
    L = V[-1, :] / V[-1, -1]

    # Camera projection matrix
    H = L.reshape(3, nd + 1)

    # Denormalization
    H = N.dot( N.dot( N.linalg.pinv(Tuv), H ), Txyz )
    H = H / H[-1, -1]
    L = H.flatten(0)

    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uv2 = N.dot( H, N.concatenate( (xyz.T, N.ones((1,xyz.shape[0]))) ) ) 
    uv2 = uv2 / uv2[2, :] 
    # Mean distance:
    err = N.sqrt( N.mean(N.sum( (uv2[0:2, :].T - uv)**2, 1)) ) 

    return L, err

def DLT():
    # Known 3D coordinates
    xyz = [[-875, 0, 9.755], [442, 0, 9.755], [1921, 0, 9.755], [2951, 0.5, 9.755], [-4132, 0.5, 23.618],
    [-876, 0, 23.618]]
    # Known pixel coordinates
    uv = [[76, 706], [702, 706], [1440, 706], [1867, 706], [264, 523], [625, 523]]

    nd=3
    P, err = DLTcalib(nd, xyz, uv)
    print('Matrix')
    print P
    print('\nError')
    print err

if __name__ == "__main__":
    DLT()