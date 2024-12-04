import numpy as np
import math
fovy=110
height,width=640,480
f = 0.5 * height / math.tan(fovy * math.pi / 360)
intrinsic_matrix=np.array([[f,0,width/2],[0,f,height/2],[0,0,1]])
print(intrinsic_matrix)