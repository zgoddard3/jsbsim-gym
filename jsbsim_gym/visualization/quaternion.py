import numpy as np

class Quaternion:
    def __init__(self, w=1, x=0, y=0, z=0):
        self._arr = np.array([w, x, y, z], dtype=np.float32)
    
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            q1 = self._arr
            q2 = other._arr
            q3 = np.zeros(4)
            q3[0] = q1[0]*q2[0] - q1[1:].dot(q2[1:])
            q3[1:] = q1[0]*q2[1:] + q2[0]*q1[1:] + np.cross(q1[1:], q2[1:])
            return Quaternion(*q3)
        elif isinstance(other, float):
            return Quaternion(*(other*self._arr))
        elif isinstance(other, np.ndarray):
            other = other.squeeze()
            if other.shape == (3,):
                return self * Quaternion(0, *other)
            else:
                raise ValueError(f"Array must be length 3 to multiply with Quaternion")
        elif isinstance(other, (float, int)):
            return Quaternion(*(self._arr*other))
        else:
            raise TypeError(f"Multiplication between Quaternion and {type(other)} is not supported.")

    def inv(self):
        return Quaternion(self.w, *(-self._arr[1:]))
    
    def mat(self):
        w,x,y,z = self._arr
        return np.array([[1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w)],
                        [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)],
                        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)]])
    
    @classmethod
    def from_euler(cls, phi, theta, psi, mode=0):
        q1 = Quaternion(np.cos(phi/2), np.sin(phi/2), 0, 0)
        q2 = Quaternion(np.cos(theta/2), 0, np.sin(theta/2), 0)
        q3 = Quaternion(np.cos(psi/2), 0, 0, np.sin(psi/2))
        if mode == 0:
            return q3*q2*q1
        else:
            return q2*q1*q3
    
    def __repr__(self):
        return f"<Quaternion: w={self.w}, x={self.x}, y={self.y}, z={self.z}>"
    
    def copy(self):
        return Quaternion(*self._arr)
    
    #region Properties

    @property
    def w(self):
        return self._arr[0]
    
    @property
    def x(self):
        return self._arr[1]

    @property
    def y(self):
        return self._arr[2]

    @property
    def z(self):
        return self._arr[3]
    
    @w.setter
    def w(self, w):
        self._arr[0] = w
    
    @x.setter
    def x(self, x):
        self._arr[1] = x
    
    @y.setter
    def y(self, y):
        self._arr[2] = y
    
    @z.setter
    def z(self, z):
        self._arr[3] = z
    
    #endregion Properties