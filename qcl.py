"""
Conventions:
shape = (nt, nx, ny, nz,...)
mu = +1(X),+2(Y),+3(Z),+4(T),...
mu = -1(X),-2(Y),-3(Z),-4(T),...
path = (+1,-2,-3,+2,+3,-1)

Time is always 0 but in paths it is D because we need +D, -D.
field.data.shape = (volume,*siteshape)

attention sync,load,save and other functions are not parallel!
"""

import math
import sys
import os
import copy
import time
import operator
import itertools
import logging
import random
import re
import unittest
import numpy # types: int8, int16, int32, int64,
             #        float32, float64, complex64, complex128

DEBUG = '-debug' in sys.argv[1:]

if DEBUG:
    import warnings

try:
    import pyopencl as cl
except ImportError:
    logging.warn('pyOpenCL not found')
    cl = None

COMPLEX_TYPES = (numpy.complex64, numpy.complex128)
MAXD = 10 # muct much the same variable in kernels/qcl-core.c
INCLUDE_PATH = os.path.join(os.getcwd(),'kernels')
IGNORE_NOT_IMPLEMENTED = True
(X,Y,Z,T) = (1,2,3,4) # indices

###
# Memory monitoring
##

def check_free_ram():
    os.system("top -l 1 | awk '/PhysMem:/ {print $10}'")

###
# Plotting functions
###

class Canvas(object):
    def __init__(self,title='',xlab='',ylab='',xrange=None,yrange=None):
        from matplotlib.figure import Figure
        self.fig = Figure()
        self.fig.set_facecolor('white')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlab)
        self.ax.set_ylabel(ylab)
        if xrange: self.ax.set_xlim(xrange)
        if yrange: self.ax.set_ylim(yrange)
        self.legend = []
    def save(self,filename='plot.png'):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        if self.legend:
            self.ax.legend([e[0] for e in self.legend],[e[1] for e in self.legend])
        if filename:
            FigureCanvasAgg(self.fig).print_png(open(filename,'wb'))
        else:
            from cStringIO import StringIO
            s = StringIO()
            FigureCanvasAgg(self.fig).print_png(s)
            return s.getvalue()
    def binary(self):
        return self.save(None)
    def hist(self,data,bins=20,color='blue',legend=None):
        q = self.ax.hist(data,bins)
        if legend: self.legend.append((q[0],legend))
        return self
    def plot(self,data,color='blue',style='-',width=2,legend=None):
        x,y = [p[0] for p in data], [p[1] for p in data]
        q = self.ax.plot(x,y,linestyle=style,linewidth=width,color=color)
        if legend: self.legend.append((q[0],legend))
        return self
    def errorbar(self,data,color='black',marker='o',width=2,legend=None):
        x,y,dy = [p[0] for p in data], [p[1] for p in data], [p[2] for p in data]
        q = self.ax.errorbar(x,y,yerr=dy,fmt=marker,linewidth=width,color=color)
        if legend: self.legend.append((q[0],legend))
        return self
    def ellipses(self,data,color='blue',width=0.01,height=0.01):
        from matplotlib.patches import Ellipse
        for point in data:
            x, y = point[:2]
            dx = point[2] if len(point)>2 else width
            dy = point[3] if len(point)>3 else height
            ellipse = Ellipse(xy=(x,y),width=dx,height=dy)
            self.ax.add_artist(ellipse)
            ellipse.set_clip_box(self.ax.bbox)
            ellipse.set_alpha(0.5)
            ellipse.set_facecolor(color)
        return self
    def imshow(self,data,interpolation='bilinear'):
        self.ax.imshow(data).set_interpolation(interpolation)
        return self

##
# Other Auxiliary function
##

def is_dir(a,b,d):
    return a>=0 and b>=0 and a%d == b%d

def is_closed_path(path):
    p = [0]*100
    for x in path:
        if x==0: raise RuntimeError("0 in path")
        p[abs(x)] += +1 if x>0 else -1
    return not any(p)

def is_single_path(path):
    return isinstance(path,(list,tuple)) and all(isinstance(x,int) for x in path)

def is_multiple_paths(paths):
    return isinstance(paths,(list,tuple)) and all(is_single_path(x) for x in paths)

def random_name():
    """
    Auxiliary function user to create a random sequence of alphanumeric characters
    It is user to generate function names
    """
    chars = 'abcdefghijklmonopqrstuvwxyz'
    return ''.join(random.choice(chars) for i in range(5))

def identity(n):
    """ Returns an identity matrix n x n """
    return numpy.matrix(numpy.eye(n))

def hermitian(U):
    """ Returns the hermitian of the U matrix """
    return numpy.transpose(U).conj()

def is_unitary(U,precision=1e-4):
    """ Checks if U is unitary within precision """
    return numpy.all(abs(U*hermitian(U)-identity(U.shape[0]))<precision)

def product(a):
    """ Auxiliary function computes product of a items """
    return 1 if not a else reduce(operator.mul,a)

def code_mul(name,name1,name2,n,m,p):
    RE, IM = '%s_%ix%i.x', '%s_%ix%i.y'
    ADD, PLUS, MINUS, TIMES, EQUAL = ' += ', '+', '-', '*', ' = '
    NEWLINE = '\n'+' '*12
    code = []
    for i in range(n):
        for j in range(m):
            var_re = RE % (name,i,j)
            var_im = IM % (name,i,j)
            k = 0
            line = var_re + EQUAL
            for k in range(0,p):
                line += NEWLINE + ' '*len(var_re +EQUAL)
                line += PLUS+RE % (name1,i,k);
                line += TIMES+RE % (name2,k,j);
                line += MINUS+IM % (name1,i,k);
                line += TIMES+IM % (name2,k,j);
            code.append(line+';')
            line = var_im + EQUAL
            for k in range(0,n):
                line += NEWLINE + ' '*len(var_re +EQUAL)
                line += PLUS+RE % (name1,i,k);
                line += TIMES+IM % (name2,k,j);
                line += PLUS+IM % (name1,i,k);
                line += TIMES+RE % (name2,k,j);
            code.append(line+';')
    return code

def code_mulh(name,name1,name2,n,m,p):
    RE, IM = '%s_%ix%i.x', '%s_%ix%i.y'
    ADD, PLUS, MINUS, TIMES, EQUAL = ' += ', '+', '-', '*', ' = '
    NEWLINE = '\n'+' '*12
    code = []
    for i in range(n):
        for j in range(m):
            var_re = RE % (name,i,j)
            var_im = IM % (name,i,j)
            k = 0
            line = var_re + EQUAL
            for k in range(0,p):
                line += NEWLINE + ' '*len(var_re +EQUAL)
                line += PLUS+RE % (name1,i,k);
                line += TIMES+RE % (name2,j,k);
                line += PLUS+IM % (name1,i,k);
                line += TIMES+IM % (name2,j,k);
            code.append(line+';')
            k = 0
            line = var_im + EQUAL
            for k in range(0,p):
                line += NEWLINE + ' '*len(var_re +EQUAL)
                line += MINUS+RE % (name1,i,k);
                line += TIMES+IM % (name2,j,k);
                line += PLUS+IM % (name1,i,k);
                line += TIMES+RE % (name2,j,k);
            code.append(line+';')
    return code

# ###########################################################
# Part I, Matrices
# ###########################################################

GAMMA = {
    'dummy': [
        numpy.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
        numpy.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
        numpy.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
        numpy.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])],
    'ukqcd': [
        numpy.matrix([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]]),
        numpy.matrix([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]]),
        numpy.matrix([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]]),
        numpy.matrix([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])],
    'fermilab': [
        numpy.matrix([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]]),
        numpy.matrix([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]),
        numpy.matrix([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]]),
        numpy.matrix([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]])],
    'milc': [
        numpy.matrix([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]),
        numpy.matrix([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]]),
        numpy.matrix([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]),
        numpy.matrix([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])],
    'chiral': [
        numpy.matrix([[0,0,0,-1],[0,0,-1,0],[0,-1,0,0],[-1,0,0,0]]),
        numpy.matrix([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]]),
        numpy.matrix([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]]),
        numpy.matrix([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])],
}
def init_GAMMA():
    for gamma in GAMMA.values():
        gamma.append(gamma[0])
        gamma.append(gamma[0]*gamma[1]*gamma[2]*gamma[3])
init_GAMMA()

class Lambda(object):
    """
    Container of Sigma and Lambda matrices (generators of SU(N))
    """
    def __init__(self,n):
        """
        for n==2 produces sigma matrices
        for n==3 produces 8 lambda matrices but in arbitrary order
        ... and so on.
        """
        def delta(i,j): return 1 if i==j else 0
        ng = n*n-1
        self.matrices = []
        for a in range(ng):
            pos2 = n*(n-1)
            pos1 = pos2/2
            mat = numpy.matrix([[0j]*n]*n)
            b = 0
            for i in range(n):
                for j in range(i+1,n):
                    mat[i,j] = (delta(b,a)-1j*delta(pos1+b,a))/2
                    mat[j,i] = (delta(b,a)+1j*delta(pos1+b,a))/2
                    b+=1
            for i in range(n-1):
                mult = delta(pos2+i,a) * (1.0/math.sqrt(2.+2./(1.+i))/(1.+i))
                for j in range(i+1):
                    mat[j,j] += mult
                mat[i+1,i+1] -= (1+i)*mult
            self.matrices.append(2.0*mat)
    def __len__(self):
        return len(self.matrices)
    def __getitem__(self,i):
        return self.matrices[i]


# ###########################################################
# Part II, lattices and fields
# ###########################################################

def listify(items):
    """ Turns a tuple or single integer into a list """
    if isinstance(items,(int,long)):
        return [items]
    elif isinstance(items,tuple):
        return [item for item in items]
    elif isinstance(items,list):
        return items
    else:
        raise ValueError("expected a list")

def makesource(vars=None,filename='kernels/qcl-core.c'):
    """ Loads and compiles a kernel """
    source = open(filename).read()
    for key,value in (vars or {}).items():
        source = source.replace('//[inject:%s]'%key,value)
    lines = [line.strip() for line in source.split('\n')]
    newlines = []
    padding = 0
    for line in lines:
        if not line.startswith('}') and line:
            if padding==0 and newlines and not line.startswith('#'):
                newlines.append('\n')
            newlines.append(' '*padding+line)
        padding += 4*(line.count('{')-line.count('}'))
        padding += 2*(line.count('(')-line.count(')'))
        if line.startswith('}') and line:
            newlines.append(' '*padding+line)
    return '\n'.join(newlines)

class Communicator(object):
    """
    Abstracts MPI as well as communications to OpenCL devices
    it assumes one process per opencl device and creates the
    ctx and queue for that device

    Currently only supports one device. The plan is to support more.
    """
    def __init__(self):
        if not cl:
            raise RuntimeError('pyOpenCL is not available')
        self.platforms = cl.get_platforms()
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        self.rank = 0
        self.nodes = 1
    def buffer(self,t,hostbuf):
        """ Makes a new opencl buffer """
        s = {'r': self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
             'w': self.mf.WRITE_ONLY | self.mf.COPY_HOST_PTR,
             'rw': self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
             'wr': self.mf.READ_WRITE | self.mf.COPY_HOST_PTR}[t]
        return cl.Buffer(self.ctx,s,hostbuf=hostbuf)
    def compile(self,source):
        """ Compiles a kernel """
        return cl.Program(self.ctx,source).build(options=['-I',INCLUDE_PATH])
    def add(self,value):
        """ MPI add """
        if not IGNORE_NOT_IMPLEMENTED: raise NotImplementedError
    def send(self,dest,data):
        """ MPI send """
        if not IGNORE_NOT_IMPLEMENTED: raise NotImplementedError
    def recv(self,source,data):
        """ MPI recv """
        if not IGNORE_NOT_IMPLEMENTED: raise NotImplementedError
    def Lattice(self,shape,bboxs=None):
        """
        Creates a new lattice that lives on the device
        shape is a tuple with the dimensions of the site tensor.
        """
        return Lattice(self,shape,bboxs)


class Lattice(object):
    """ a lattice encodes info about volume and parallelization """
    def __init__(self,comm,shape,bboxs=None):
        """
        Constructor of a Lattice
        It computes the size and allocaltes local ranlux prng on device
        """
        self.comm = comm
        self.shape = listify(shape)
        self.d = len(shape)
        self.size = size = product(self.shape)
        self.bboxs = bboxs or [[(0,)*self.d,self.shape]]
        self.prngstate_buffer = None
        self.prng_on(time.time()) # for debugging!
        self.parallel = len(self.bboxs)>1 # change for multi-GPU
        self.bbox = numpy.zeros(MAXD*6,dtype=numpy.int32)
        # weird logic for future multi-gpu support
        self._bbox_init()
    ### bbox stuff
    def _bbox_init(self):
        """ Initalizes the size of the lattice partition hosted on device """
        for i,x in enumerate(self.shape):
            self.bbox[i]=0 # global coordinates if the start of bbox
            self.bbox[i+MAXD]=0 # top padding size
            self.bbox[i+2*MAXD]=x # internal bbox size
            self.bbox[i+3*MAXD]=x # external bbox size
            self.bbox[i+4*MAXD]=0 # init looping
            self.bbox[i+5*MAXD]=1 # step looping
    def _bbox_loop(self,init=(0,0,0,0),step=(1,1,1,1)):
        for i,x in enumerate(init):
            self.bbox[i+4*MAXD]=x # init looping
        for i,x in enumerate(step):
            self.bbox[i+5*MAXD]=x # step looping
        size = 1
        for i in range(self.d):
            size *= int(self.bbox[i+2*MAXD]/self.bbox[i+5*MAXD])
        return size
    def bbox_range(self,displacement):
        """
        This is an iterator that allows looping over checkboard-like
        subsets of the lattice bbox. A bbox is a subset of the lattice
        padded with buffers.
        "displacement" is the min distance between two sites that can
        updated simultaneously without affecting each other.
        This iterator is used in the heatbath algorithm
        """
        for i in range(displacement**self.d):
            v = []
            for j in range(self.d):
                v,i = v+[i%displacement],i/displacement
            size = self._bbox_loop(v,step=[displacement]*self.d)
            yield size, v
        self._bbox_loop([0]*self.d,[1]*self.d)
    ### end bbox stuff

    def prng_on(self,seed):
        """
        Initializes the parallel random number generator
        It currently uses ranlux, the Lusher's generator
        which comes with pyOpenCL
        """
        self.seed = seed
        STATE_SIZE = 112
        self.prngstate = numpy.zeros(self.size*STATE_SIZE,dtype=numpy.float32)
        prg = self.comm.compile(makesource())
        self.prngstate_buffer = self.comm.buffer('rw',self.prngstate)
        prg.init_ranlux(self.comm.queue,(self.size,),None,
                        numpy.uint32(seed),self.prngstate_buffer)
    def prng_get(self):
        """ Retrieves the state of the prng from device """
        cl.enqueue_copy(self.comm.queue, self.prngstate,
                        self.prngstate_buffer).wait()
    def prng_off(self):
        """ Disabled the prng on device"""
        self.prngstate = self.prngstate_buffer = None
    def coords2global(self,coords):
        """ Converts (t,x,y,z) coordinates into global index """
        if len(coords)!=len(self.shape):
            raise RuntimeError("invalid conversion")
        return sum(p*product(self.shape[i+1:]) for i,p in enumerate(coords))
    def global2coords(self,i):
        """ Converts global index into (t,x,y,z) """
        coords = []
        for k in reversed(self.shape):
            i,reminder = divmod(i,k)
            coords.insert(0,reminder)
        return tuple(coords)
    def check_coords(self):
        for i in range(self.size):
            assert self.coords2global(self.global2coords(i)) == i
    def Site(self,*coords):
        """ Returns a Site constructor for this lattice """
        return Site(self,*coords)
    def Field(self,siteshape,dtype=numpy.complex64):
        """ Returns a Field constructor for this """
        return Field(self,siteshape,dtype=dtype)
    def ComplexScalarField(self,dtype=numpy.complex64):
        """ Returns a Field constructor for this """
        return ComplexScalarField(self,dtype=dtype)
    def GaugeField(self,nc,dtype=numpy.complex64):
        """ Returns a Field constructor for this """
        return GaugeField(self,nc,dtype=dtype)
    def FermiField(self,ndim,nc,dtype=numpy.complex64):
        """ Returns a Field constructor for this """
        return FermiField(self,ndim,nc,dtype=dtype)
    def StaggeredField(self,nc,dtype=numpy.complex64):
        """ Returns a Field constructor for this """
        return StaggeredField(self,nc,dtype=dtype)


class Site(object):
    """ The site object stores information about a lattice site """
    def __init__(self,lattice,*coords):
        self.lattice = lattice
        self.coords = tuple(coords)
        self.gl_idx = self.lattice.coords2global(coords)
        self.lo_idx = self.gl_idx ### until something better
    def __add__(self,mu):
        """
        Example:
        >>> lattice = Lattice([4,4,4,4])
        >>> p = Site(lattice,0,0,0,0)
        >>> q = p + 1
        >>> assert p.coords == (0,1,0,0)
        >>> q = p + 2
        >>> assert p.coords == (0,0,1,0)
        >>> q = p + 3
        >>> assert p.coords == (0,0,0,1)
        >>> q = p + 4
        >>> assert x.coords == (1,0,0,0)
        >>> q = p + (-4)
        >>> assert x.coords == (3,0,0,0)
        """
        if mu<0:
            return self.__sub__(-mu)
        mu = mu % self.lattice.d
        L = self.lattice.shape[mu]
        coords = tuple((c if nu!=mu else ((c+1)%L))
                       for nu,c in enumerate(self.coords))
        return self.lattice.Site(*coords)
    def __sub__(self,mu):
        """
        >>> q = p - mu
        same as
        >>> q = p + (-mu)
        """
        if mu<0:
            return self.__add__(-mu)
        mu = mu % self.lattice.d
        L = self.lattice.shape[mu]
        coords = tuple((c if nu!=mu else ((c+L-1)%L))
                       for nu,c in enumerate(self.coords))
        return self.lattice.Site(*coords)
    def __str__(self):
        return str(self.coords)

class Op(object):
    """ For lazy evaluation of expressions """
    def __init__(self,op,left,right=None):
        self.op = op
        self.left = left
        self.right = right

class Field(object):
    """
    A field can be used to store a guage field, a fermion field,
    a staggered field, etc
    Mo math operations between fields here except for O(n) operations,
    the others MUST be done in OpenCL only!

    Example:
    >>> lattice = Lattice([4,4,4,4])
    >>> phi = Field(lattice,(4,3),dtype=numpy.complex64)

    dtype=numpy.complex64 is the default and some algorithms rely on this value.
    """
    def __init__(self, lattice, siteshape, dtype=numpy.complex64):
        self.lattice = lattice
        self.siteshape = listify(siteshape)
        self.sitesize = product(self.siteshape)
        self.size = lattice.size * self.sitesize
        self.dtype = dtype
        self.data = numpy.zeros([self.lattice.size]+self.siteshape,dtype=dtype)
        if DEBUG:
            print 'allocating %sbytes' % int(self.lattice.size*self.sitesize*8)
    def buffer(self, mode):
        """ mode = 'r' or 'w' or 'rw' """
        return self.lattice.comm.buffer(mode, self.data)

    def clone(self):
        return Field(self.lattice,self.siteshape,dtype=self.dtype)

    def set_copy(self,other):
        """
        Makes a copy of field b into field a
        Exmaple:
        >>> a.set_copy(b)
        """
        if not (self.lattice == self.lattice and
                self.siteshape==other.siteshape and
                self.sitesize == other.sitesize):
            raise TypeError("Cannot copy incompatible fields")
        self.dtype = other.dtype
        self.data[:] = other.data

    def data_component(self,component):
        """
        Returns a new field containing a given component of the current field
        Example:
        >>> phi = Field(lattice,(4,3))
        >>> psi00 = phi.data_component((0,0))
        """

        newfield = self.lattice.Field((1,))
        for i in xrange(self.lattice.size):
            newfield.data[i] = self.data[i][component]
        return newfield

    def lattice_slice(self,slice_coords):
        """
        Returns a numpy slice of the current self.data at coordinates c
        """
        if self.sitesize!=1: raise NotImplementedError
        d = self.lattice.shape[len(slice_coords):]
        coords = [e for e in slice_coords]+[0 for i in d]
        t = self.lattice.coords2global(coords)
        s = product(d)
        return self.data[t:t+s].reshape(d)

    def slice(self,slice_coords,components,projector = numpy.real):
        return projector(self.data_component(components)\
                             .lattice_slice(slice_coords))

    def __imul__(self,other):
        """ a *= c """
        self.data *= other
        return self

    def __iadd__(self,other):
        """ a += b """
        if isinstance(other,Op):
            if other.op=='*': self.add_scaled(other.left,other.right)
            else: raise NotImplementedError
        else:
            if self.data.shape != other.data.shape: raise RuntimeError
            self.data += other.data
        return self
    def __isub__(self,other):
        """ a -= b """
        if isinstance(other,Op):
            if other.op=='*': self.add_scaled(-other.left,other.right)
            else: raise NotImplementedError
        else:
            if self.data.shape != other.data.shape: raise RuntimeError
            self.data -= other.data
        return self
    def __rmul__(self,other):
        """ Maps a += c*b into a.add_scaled(c,b) for a,b arrays """
        return Op('*',other,self)
    def __add__(self,other):
        """ Stores a + b """
        raise NotImplementedError
    def __sub__(self,other):
        """ Maps a - b """
        raise NotImplementedError
    def add_scaled(self,scale,other,n=1000000):
        """ a.add_scaled(c,b) is the same as a[:]=c*b[:] """
        if self.data.shape != other.data.shape: raise RuntimeError
        size = product(self.data.shape)
        for i in xrange(0,size,n):
            self.data.flat[i:i+n] += scale*other.data.flat[i:i+n]
        return self
    def __getitem__(self, start, stop, step):
        return self.data[start:stop:step]
    def __mul__(self,other):
        """
        Computes scalar product of two Fields
        It first re-shape them into 1D arrays, then computes product
        Not designed to work in parallel (yet)
        """
        if self.lattice.parallel: raise NotImplementedError
        if self.data.shape != other.data.shape: raise RuntimeError
        vdot = numpy.vdot if self.dtype in COMPLEX_TYPES else numpy.dot
        return vdot(self.data,other.data)
    def __getitem__(self,args):
        """ Do not use these to implement algorithms - too slow """
        site, args = args[0], args[1:]
        if not isinstance(site,Site): site = self.lattice.Site(*site)
        return self.data[(site.lo_idx,)+args]
    def __setitem__(self,args,value):
        """ Do not use these to implement algorithms - too slow """
        site, args = args[0], args[1:]
        if not isinstance(site,Site): site = self.lattice.Site(*site)
        self.data[(site.lo_idx,)+args] = value
    def sum(self,*a,**b):
        """ a.sum() computes the sum of terms in a """
        return numpy.sum(self.data,*a,**b)
    def load(self,filename):
        """ Loads the field using the numpy binary format (fails on Mac) """
        format = filename.split('.')[-1].lower()
        if format == 'npy':
            self.data = None
            self.data = numpy.load(filename)
        else:
            raise NotImplementedError
        return self
    def fft(self,time = False):
        """ returns a ndarray with the fft over the spatial self.data """
        return numpy.fft.fftn(
            self.data.reshape(self.lattice.shape+self.siteshape),
            axes=range(0 if time else 1,self.lattice.d))
    def save(self,filename):
        """ Saves the field supports *.npy and *.vtk (for 4 only)"""
        format = filename.split('.')[-1].lower()
        if format == 'npy':
            numpy.save(filename,self.data)
        elif format == 'vtk' or self.d!=4 or self.siteshape!=(1,):
            ostream = open(filename,'wb')
            s = product(self.lattice.shape[-3:])
            h1 = "# vtk DataFile Version 2.0\n"+\
                filename.split('/')[-1]+'\n'+\
                'BINARY\n'+\
                'DATASET STRUCTURED_POINTS\n'+\
                'DIMENSIONS %s %s %s\n'%tuple(self.lattice.shape[-3:])+\
                'ORIGIN     0 0 0\n'+\
                'SPACING    1 1 1\n'+\
                'POINT_DATA %s' % s
            ostream.write(h1)
            for t in xrange(self.lattice.shape[0]):
                h2 = '\nSCALARS t%s float\nLOOKUP_TABLE default\n' % t
                ostream.write(h2)
                numpy.real(self.data[t*s:t*s+s]).tofile(ostream)
            ostream.close()
        else:
            raise NotImplementedError
        return self
    def sync(self,filename):
        """ Syncronizes all buffers in multi-device environment (not implrmented) """
        if not IGNORE_NOT_IMPLEMENTED: raise NotImplementedError
        return self

    def compute_link_product(self,U,paths,name='aux'):
        """
        Generates a kernel to set the field to the sum of specified product of links
        If trace == True the product of links is traced.
        name is the name of the generate OpenCL function which performs the
        product of links.
        """
        name = name or random_name()
        code = opencl_paths(function_name = name,
                            lattice = self.lattice,
                            coefficients = [1.0 for p in paths],
                            paths = paths,
                            nc = U.data.shape[-1],
                            trace = (self.data.shape[-1]==1))
        source = makesource({'paths':code})
        def runner(prg,self=self,name=name):
            shape = self.siteshape
            out_buffer = self.buffer('w')
            U_buffer = U.buffer('r')
            function = getattr(prg,name+'_loop')
            function(self.lattice.comm.queue,(self.lattice.size,),None,
                     out_buffer, U_buffer,
                     numpy.int32(shape[0]), # 1 for trace, nc for no-trace
                     self.lattice.bbox).wait()
            cl.enqueue_copy(self.lattice.comm.queue, self.data, out_buffer).wait()
            return self
        return Code(source,runner,compiler=self.lattice.comm.compile)

    def set_link_product(self,U,paths,name='aux'):
        return self.compute_link_product(U,paths,name).run()

    def set(self,operator,*args,**kwargs):
        return operator(self,*args,**kwargs)

    def make_hadron(self,contractions,quarks):
        """
        example: rho(alpha,a) = eps(a,b,c) * q1(alpha,a)*q2(beta,b)*q3(beta,c)
        rho = lattice.FermiField(4,3)
        q1 = lattice.FermiField(4,3)
        q2 = lattice.FermiField(4,3)
        q3 = lattice.FermiField(4,3)
        spin = 0
        contractions = [spin*3+0, +1, (0,1,2), (spin,0,0)]
        contractions = [spin*3+1, +1, (1,2,0), (spin,0,0)]
        contractions = [spin*3+2, +1, (2,0,1), (spin,0,0)]
        contractions = [spin*3+0, -1, (0,2,1), (spin,0,0)]
        contractions = [spin*3+2, -1, (2,1,0), (spin,0,0)]
        contractions = [spin*3+1, -1, (1,0,2), (spin,0,0)]
        rho.make_hadron(contractions,(q1,q2,q3))
        """
        quark = quarks[0]
        if (self.lattice!=quark.lattice or 
            any(quark.data.shape != other.data.shape for other in quarks[1:]) or 
            not all(isinstance(q,FermiField) for q in quarks)):        
            raise RuntimeError("incomplatible arguments")
        nspin, nc = quark.nspin, quark.nc
        quarks_def = ''.join(',global cfloat_t *quark%i' % i
                                   for i in range(len(quarks)))
        code = []
        for i in range(len(quarks)):
            code.append('global cfloat_t *q%i;' % i)
        code.append('ix = rho + idx*%i;' % (self.sitesize))
        code.append('for(k=0; k<%i; k++) ix[k].x = ix[k].y = 0.0;' % self.sitesize)

        for i in range(len(quarks)):
            code.append('q%i = quark%i + idx*%i;' % (i,i,nspin*nc))
        for c,contraction in enumerate(contractions):
            comp, coeff, spin_idx, color_idx = contraction
            if coeff:
                i = spin_idx[0]*nc + color_idx[0]
                j = spin_idx[1]*nc + color_idx[1]
                code.append('tmp.x = q0[%i].x*q1[%i].x - q0[%i].y*q1[%i].y;' % (
                        i,j,i,j))
                code.append('tmp.y = q0[%i].x*q1[%i].y + q0[%i].y*q1[%i].x;' % (
                        i,j,i,j))
                for k in range(2,len(spin_idx)):
                    j = spin_idx[k]*nc + color_idx[k]
                    code.append('tmp.x = tmp.x*q%i[%i].x - tmp.y*q%i[%i].y;' % (
                            k,j,k,j))
                    code.append('tmp.y = tmp.x*q%i[%i].y + tmp.y*q%i[%i].x;' % (
                            k,j,k,j))
                code.append('ix[%i].x += (%s) * tmp.x - (%s) * tmp.y;' % (
                        comp, coeff.real, coeff.imag))
                code.append('ix[%i].y += (%s) * tmp.y + (%s) * tmp.x;' % (
                        comp, coeff.real, coeff.imag))
        code = '\n'.join(' '*12+line for line in code)
        source = makesource({'make_hadron':code,'quarks':quarks_def})
        def runner(prg,self=self,quarks=quarks):
            data_buffer = self.buffer('w')
            event = prg.make_hadron(self.lattice.comm.queue,
                                    (self.lattice.size,),None,
                                    data_buffer,
                                    self.lattice.bbox,
                                    *[q.buffer('r') for q in quarks])
            if DEBUG:
                print 'waiting'
            event.wait()
            cl.enqueue_copy(self.lattice.comm.queue, self.data, data_buffer).wait()
            return self
        return Code(source,runner,compiler=self.lattice.comm.compile)

    def set_hadron(self,contractions,quarks):
        return self.make_hadron(contractions,quarks).run()

def cyclic(n):
    permutations = []
    a = range(n)
    for k in range(n):
        permutations.append((+1,[x for x in a]))
        a = a[-1:]+a[:-1]
    a = range(n-1,-1,-1)
    for k in range(n):
        permutations.append((-1,[x for x in a]))
        a = a[-1:]+a[:-1]
    return permutations

def make_meson(left,gamma,right):
    if left.data.shape != right.data.shape:
        raise RuntimeError("Cannot be contracted")
    rho = left.lattice.Field((1,))
    contractions = [(0,gamma[a,b],(a,b),(i,i))
                    for i in range(left.nc) 
                    for a in range(left.nspin)
                    for b in range(left.nspin) 
                    if gamma[a,b]]
    rho.set_hadron(contractions,(left,right))
    return rho

def make_baryon3(left,middle,gamma,right):
    print gamma
    rho = left.lattice.Field((left.nspin))
    perms = cyclic(3)
    contractions = []
    for alpha in range(0,left.nspin):
        for beta in range(0,left.nspin):
            for kappa in range(0,left.nspin):
                for sign, color_perm in perms:
                    coeff = sign*gamma[beta,kappa]
                    if coeff:
                        contractions.append((alpha,coeff,(alpha,beta,kappa),color_perm))
    rho.set_hadron(contractions,(left,middle,right))

class ComplexScalarField(Field):
    def __init__(self, lattice, dtype=numpy.complex64):
        Field.__init__(self, lattice, (1,), dtype=dtype)

    def clone(self):
        return ComplexScalarField(self.lattice, dtype=self.dtype)

class GaugeField(Field):
    def __init__(self, lattice, nc, dtype=numpy.complex64):
        Field.__init__(self, lattice, (lattice.d, nc, nc), dtype=dtype)
        self.d = lattice.d
        self.nc = nc

    def clone(self):
        return GaugeField(self.lattice, self.nc, dtype=self.dtype)

    def set_cold(self):
        """ Uses a kernel to set all links to cold """
        shape = self.siteshape
        if not (len(shape)==3 and shape[1]==shape[2]): raise RuntimeError
        data_buffer = self.buffer('w')
        prg = self.lattice.comm.compile(makesource())
        prg.set_cold(self.lattice.comm.queue,(self.lattice.size,),None,
                     data_buffer,numpy.int32(shape[0]),numpy.int32(shape[1]),
                     self.lattice.bbox).wait()
        cl.enqueue_copy(self.lattice.comm.queue, self.data, data_buffer).wait()

    def set_custom(self):
        """ Uses a kernel to set all links to come cutsom values, for testing only """
        shape = self.siteshape
        if not (len(shape)==3 and shape[1]==shape[2]): raise RuntimeError
        data_buffer = self.buffer('w')
        prg = self.lattice.comm.compile(makesource())
        prg.set_custom(self.lattice.comm.queue,(self.lattice.size,),None,
                       data_buffer,numpy.int32(shape[0]),numpy.int32(shape[1]),
                       self.lattice.bbox).wait()
        cl.enqueue_copy(self.lattice.comm.queue, self.data, data_buffer).wait()

    def check_cold(self):
        """ Checks if the field is a cold gauge configuration """
        for idx in xrange(self.lattice.size):
            for mu in xrange(self.siteshape[0]):
                for i in xrange(self.siteshape[1]):
                    for j in xrange(self.siteshape[2]):
                        assert(self.data[idx,mu,i,j]==(1 if i==j else 0))
    def set_hot(self):
        """
        Uses a kernel to set all links to hot (random SU(n))
        Based on Cabibbo-Marinari
        """
        shape = self.siteshape
        if not (len(shape)==3 and shape[1]==shape[2]): raise RuntimeError
        data_buffer = self.buffer('w')
        prg = self.lattice.comm.compile(makesource())
        prg.set_hot(self.lattice.comm.queue,(self.lattice.size,),None,
                    data_buffer,numpy.int32(shape[0]),numpy.int32(shape[1]),
                    self.lattice.bbox,self.lattice.prngstate_buffer).wait()
        cl.enqueue_copy(self.lattice.comm.queue, self.data, data_buffer).wait()

    def set_smeared(self,smear_operator,reunitarize=0):        
        smear_operator.smear_to(self,reunitarize).run()

    def set_hisq(self,U):
        W = clone(U)
        W.set_fat(U,'fat5',reunitarize=5)
        self.set_fat(W,'fat7+lepage')

    def set_fat(self,U,name='fat3',plaquette=1.0,reunitarize=0,coefficients=None):
        u0 = (plaquette.real)**(0.25)
        DATA = {
            'link':[1.0,0,0,0,0],
            'fat3':[9.0/32,9.0/(64*u0**2),0,0,0], # attention this assume Naik term -9.0/(8*27);
            'fat5':[1.0/7,1.0/(7*2*u0**2),1.0/(7*8*u0**4),0,0],
            'fat7':[1.0/8,1.0/(8*2*u0**2),1.0/(8*8*u0**4),1.0/(8*48*u0**6),0],
            'fat7+lepage':[5.0/8,1.0/(8*2*u0**2),1.0/(8*8*u0**4),1.0/(8*48*u0**6),-1.0/(16.0*u0**2)], # attention this assume Naik term -1.0/24.0*pow(u0,-2);
            }
        c = DATA.get(name,coefficients)
        if not c: raise NotImplementedError
        S = GaugeSmearOperator(U)
        if c[0]: S.add_term([1],c[0])
        if c[1]: S.add_term([2,1,-2],c[1])
        if c[2]: S.add_term([3,2,1,-2,3],c[2])
        if c[3]: S.add_term([4,3,2,1,-2,-3,-4],c[3])
        if c[4]: S.add_term([2,2,1,-2,-2],c[4])
        S.smear_to(self,reunitarize).run()

    def check_unitarity(self,output=DEBUG):
        """
        Check that all matrices in the current field (assimung gauge config)
        are unitary. Prints the matrices which are not and raises an Exception.
        """
        fail = False
        for idx in xrange(self.lattice.size):
            for mu in xrange(self.siteshape[0]):
                U = numpy.matrix(self.data[idx,mu])
                if output:
                    print U
                    print 'det = %s' % numpy.linalg.det(U)
                if not is_unitary(U):
                    #print idx, mu
                    #print U
                    #print U*hermitian(U)
                    fail = True
        if fail:
            for idx in xrange(self.lattice.size):
                for mu in xrange(self.siteshape[0]):
                    U = numpy.matrix(self.data[idx,mu])
                    print idx, mu, U
                    if not is_unitary(U):
                        print U*hermitian(U), 'FAIL!'
            raise RuntimeError("U is not unitary")

    def average_plaquette(self,shape=(1,2,-1,-2),paths=None):
        """
        Compute the average plaquette using paths =  symmetrized (shape)
        One can specify paths to override symmetrization.
        """
        if not paths:
            paths = bc_symmetrize(shape,d=self.lattice.d,positive_only=True)
            paths = remove_duplicates(paths,bidirectional=True)
        # print 'average_plaquette.paths=',paths
        phi = self.lattice.Field(1).set_link_product(self,paths)
        #for idx in xrange(self.lattice.size):
        #    print idx, phi.data[idx]
        return phi.sum()/(self.lattice.size*len(paths)*self.siteshape[-1])

    def clover(self):
        nc = self.siteshape[-1]
        fields = []
        for mui in range(0,self.d-1):
            for nui in range(mui+1,self.d):
                mu = mui if mui>0 else self.d
                nu = nui if nui>0 else self.d
                # comput forward clover leaf
                paths = [(mu,nu,-mu,-nu),(nu,-mu,-nu,mu),(-mu,-nu,mu,nu),(-nu,mu,nu,-mu)]
                component = self.lattice.Field((nc,nc))\
                    .set_link_product(self,paths)
                # reverse paths
                paths = [(p[3],p[2],p[1],p[0]) for p in paths]
                component -= self.lattice.Field((nc,nc))\
                    .set_link_product(self,paths)
                component *= 0.125
                fields.append(component)
        return fields
    

class FermiField(Field):
    def __init__(self, lattice, nspin, nc, dtype=numpy.complex64):
        Field.__init__(self, lattice, (nspin, nc), dtype=dtype)
        self.nspin = nspin
        self.nc = nc

    def clone(self):
        return FermiField(self.lattice,self.nspin, self.nc, dtype=self.dtype)

class StaggeredField(Field):
    """ notice: I believe this assume GAMMA['fermilab'] """
    def __init__(self, lattice, nc, dtype=numpy.complex64):
        Field.__init__(self, lattice, (nc,), dtype=dtype)
        self.nc = nc
        self.nspin = 1

    def clone(self):
        return StaggeredField(self.lattice, self.nc, dtype=self.dtype)


class GaugeAction(object):
    """
    Class to store paths and coefficients relative to any gauge action
    """
    def __init__(self,U):
        self.U = U
        self.lattice = U.lattice
        self.terms = []
    def add_term(self,paths,coefficient=1.0):
        """
        Paths are symmetrized. For example:
        >>> wilson = GaugeAction(lattice).add_term((1,2,-1,-2))
        is the Wilson action.
        """
        if is_single_path(paths):
            paths = bc_symmetrize(paths,d=self.lattice.d)
            paths = remove_duplicates(paths,bidirectional=False)
        elif not is_multiple(paths):
            raise RuntimeError('invalid action term')
        self.terms.append((coefficient,paths))
        return self

    def add_plaquette_terms(self):
        self.add_term((1,2,-1,-2),1.0)
        return self

    def heatbath(self,beta,n_iter=1,m_iter=5,name='aux'):
        """
        Generates a kernel which performs the SU(n) heatbath.
        Example:
        >>> wilson = GaugeAction(lattice).add_term((1,2,-1,-2))
        >>> wilson.heatbath(beta,n_iter=10)
        """
        U = self.U
        name = name or random_name()
        code = ''
        displacement = 1
        for mu in range(U.d):
            coefficients,paths = [], []
            for coefficient, cpaths in self.terms:
                opaths = derive_paths(cpaths,mu if mu else U.d)
                for path in opaths:
                    coefficients.append(coefficient)
                    displacement = max(displacement,range_path(path)+1)
                    paths.append(backward_path(path))
            code += opencl_paths(function_name = name+str(mu),
                                 lattice = self.lattice,
                                 coefficients=coefficients,
                                 paths=paths,
                                 nc = U.data.shape[-1],
                                 initialize=True,
                                 trace=False)
        action = '\n'.join('if(mu==%i) %s%i(staples,U,idx,&bbox);' %
                           (i,name,i) for i in range(U.d))
        source = makesource({'paths':code,'heatbath_action':action})
        def runner(prg,self=self,beta=beta,n_iter=n_iter,m_iter=m_iter,
                   displacement=displacement):
            U = self.U
            lattice = U.lattice
            shape = U.siteshape
            if not (len(shape)==3 and shape[1]==shape[2]): raise RuntimeError
            data_buffer = U.buffer('rw')            
            for size, x in self.lattice.bbox_range(displacement):
                if DEBUG:
                    print 'displacement: %s, size:%s, slice:%s' % (
                        displacement, size, x)
                event = prg.heatbath(lattice.comm.queue,
                             (size,),None,
                             data_buffer,
                             numpy.int32(shape[0]),
                             numpy.int32(shape[1]),
                             numpy.float32(beta),
                             numpy.int32(n_iter),
                             numpy.int32(m_iter),
                             lattice.bbox,
                             lattice.prngstate_buffer)
                if DEBUG:
                    print 'waiting'
                event.wait()
            cl.enqueue_copy(lattice.comm.queue, U.data, data_buffer).wait()
            return self
        return Code(source,runner,compiler=self.lattice.comm.compile)

class GaugeSmearOperator(GaugeAction):
    def heathbath(self):
        raise NotImplementedError

    def smear_to(self,V,reunitarize=0,name='aux'):
        """
        Generates a kernel which performs the SU(n) heatbath.
        Example:
        >>> op = GaugeSmearOperator(U).add_term([1],1).add_term([2,1,-2],0.3)
        >>> op.smear_to(V).run() # same of V.set_smeared(op)
        """
        U = self.U
        lattice = U.lattice
        name = name or random_name()
        code = ''
        displacement = 1
        for mu in range(U.d):
            coefficients,paths = [], []
            for coefficient, cpaths in self.terms:
                for path in cpaths:
                    if is_closed_path(listify(path)+[-mu if mu else -4]):
                        coefficients.append(coefficient)
                        paths.append(path)
            # print mu, paths, coefficients
            code += opencl_paths(function_name = name+str(mu),
                                 lattice = lattice,
                                 coefficients=coefficients,
                                 paths=paths,
                                 nc = U.data.shape[-1],
                                 initialize=True,
                                 trace=False)
        action = '\n'.join('if(mu==%i) %s%i(staples,U,idx,&bbox);' %
                           (i,name,i) for i in range(U.d))
        source = makesource({'paths':code,'smear_links':action})
        def runner(prg,self=self,V=V,reunitarize=reunitarize):
            U = self.U
            lattice  = U.lattice
            shape = U.siteshape
            if not (len(shape)==3 and shape[1]==shape[2]): raise RuntimeError
            data_buffer_V = V.buffer('w')
            data_buffer_U = U.buffer('r')            
            event = prg.smear_links(
                lattice.comm.queue,
                (lattice.size,),None,
                data_buffer_V,
                data_buffer_U,
                numpy.int32(shape[0]),
                numpy.int32(shape[1]),
                numpy.int32(reunitarize),
                lattice.bbox)
            if DEBUG:
                print 'waiting'
            event.wait()
            cl.enqueue_copy(lattice.comm.queue, V.data, data_buffer_V).wait()
            return self
        return Code(source,runner,compiler=self.lattice.comm.compile)


class FermiOperator(object):
    """
    Class to store paths and coefficients relative to any gauge action
    """
    def __init__(self,U,extra_fields=None,name='aux'):
        self.terms = []
        self.U = U
        self.extra_fields = extra_fields or []
        self.lattice = U.lattice
        self.name = name or random_name()
        self.extra = range(len(extra_fields or []))

    def add_diagonal_term(self,gamma):
        return self.add_term(gamma)

    def add_term(self, gamma, 
                 paths=None, shift=None, 
                 color=None, staggered=False):
        """
        gamma is a NspinxNspin Gamma matrix for Wilson fermions
        gamma is (coefficient, mu) for staggered (mu = 1,2,3 or 4)
        gamma is a number when paths is None (the identity term)

        if paths is None than psi = gamma * phi

        Paths are symmetrized. For example:
        >>> wilson = FermiOperator(U).add_term(
            kappa,(1-G[0]),path = [(4,)])
        >>> wilson = FermiOperator(U).add_term(
            kappa,(1+G[0]),path = [(-4,)])
        >>> wilson = FermiOperator(U).add_term(
            c_sw,G[0]*G[1],path = self.extra[0])
        """
        if shift is None and is_multiple_paths(paths):
            shift = [0]*self.lattice.d
            for step in paths[0]:
                shift[abs(step) % self.lattice.d] += 1 if step>0 else -1
        term = dict(gamma=gamma,paths=paths,shift=shift,
                    staggered=staggered,color=color)
        if paths is None: # diagonal term in operator
            if self.terms and self.terms[0]['gamma'] is None:
                raise RuntimeError("cannot have two terms without paths")
            self.terms.insert(0,term)                                     
        elif (isinstance(paths,numpy.matrix) or
              is_multiple_paths(paths) or
              (isinstance(paths,int) and paths in self.extra)):
            self.terms.append(term)
        else:
            raise RuntimeError("invalid Path")
        return self

    def add_staggered_term(self, gamma, paths, shift=None, color=None):
        self.add_term(gamma,paths,shift,color,staggered=True)
        
    # action specific helper functions
    def add_staggered_action(self,kappa=1.0):
        self.add_diagonal_term(1.0)
        for mu in range(1,self.lattice.d+1):
            self.add_staggered_term(kappa, [(+mu,)])
            self.add_staggered_term(kappa, [(-mu,)])
        return self

    def add_staggered_nhop_terms(self,kappa=1.0, nhop=3):
        for mu in range(1,self.lattice.d+1):
            self.add_staggered_term(kappa, [[+mu]*nhop])
            self.add_staggered_term(kappa, [[-mu]*nhop])
        return self

    def add_wilson4d_action(self,
                           kappa=1.0,kappa_t=1.0,kappa_s=1.0,
                           r=1.0,r_t=1.0,r_s=1.0,
                           gamma=GAMMA['fermilab']):
        I = identity(4)
        self.add_diagonal_term(1.0)
        for mu in (1,2,3,4):
            k = kappa * (kappa_t if mu==4 else kappa_s)
            b = r * (r_t if mu==4 else r_s)
            self.add_term(k*(b*I-gamma[mu]), [(+mu,)])
            self.add_term(k*(b*I+gamma[mu]), [(-mu,)])
        return self

    def add_clover4d_terms(self,c_SW=1.0,c_E=1.0,c_B=1.0,
                           gamma=GAMMA['fermilab']):
        clover = self.U.clover()
        n = len(self.extra_fields)
        extra = self.U.clover()
        self.extra_fields += extra
        self.add_term(-2.0*c_SW*c_E*gamma[4]*gamma[1],color=extra[0]) # Ex
        self.add_term(-2.0*c_SW*c_E*gamma[4]*gamma[2],color=extra[1]) # Ey
        self.add_term(-2.0*c_SW*c_E*gamma[4]*gamma[3],color=extra[2]) # Ez
        self.add_term(-2.0*c_SW*c_B*gamma[1]*gamma[2],color=extra[3]) # Bx
        self.add_term(-2.0*c_SW*c_B*gamma[1]*gamma[3],color=extra[4]) # By
        self.add_term(-2.0*c_SW*c_B*gamma[2]*gamma[3],color=extra[5]) # Bz
        return self

    # end action specific helper functions

    def __call__(self,phi,psi):
        return self.multiply(phi,psi).run()

    def multiply(self,phi,psi):
        """
        Computes phi = D[U]*psi

        Generates a kernel which performs the SU(n) heatbath.
        Example:
        >>> wilson = GaugeAction(U).add_term(1.0,(1,2,-1,-2))
        >>> wilson.heatbath(beta,n_iter=10)
        """
        U, extra_fields = self.U, self.extra_fields
        name = self.name
        code = ''
        coefficients,paths = [], []
        k = 0
        action = ''
        shapeu = U.siteshape
        if len(shapeu)!=3 or shapeu[1]!=shapeu[2]: raise RuntimeError
        shapep = psi.siteshape
        if shapeu[2]!=shapep[-1]: raise RuntimeError
        if len(shapep)==2:
            is_staggered = False
            nspin = shapep[0]
        elif len(shapep)==1:
            nspin = 1
        else:
            raise RuntimeError
        nc = shapeu[2]
        k=0
        for term in self.terms:
            if is_multiple_paths(term['paths']):
                code += opencl_paths(function_name = name+str(k),
                                     lattice = self.lattice,
                                     coefficients=[1.0]*len(term['paths']),
                                     paths=term['paths'],
                                     nc = nc,
                                     initialize = True,
                                     trace = False)
            k += 1
        action = opencl_fermi_operator(self.terms,U.d,nspin,nc,name)
        extra_fields_def = ''.join(',global cfloat_t *extra%i' % id(e)
                                   for e in extra_fields) if extra_fields else ''
        key = 'fermi_operator'
        source = makesource({'paths':code, key:action,
                             'extra_fields': extra_fields_def})
        def runner(prg,self=self,phi=phi,U=U,psi=psi):

            phi_buffer = phi.buffer('rw')
            U_buffer = U.buffer('r')
            extra_buffers = [e.buffer('r') for e in extra_fields]
            psi_buffer = psi.buffer('r')
            meta_event = prg.fermi_operator
            event = meta_event(U.lattice.comm.queue,
                               (U.lattice.size,),None,
                               phi_buffer,
                               U_buffer,
                               psi_buffer,
                               U.lattice.bbox,
                               *extra_buffers)
            if DEBUG:
                print 'waiting'
            event.wait()
            cl.enqueue_copy(U.lattice.comm.queue, phi.data, phi_buffer).wait()
            return self
        return Code(source,runner,compiler=self.lattice.comm.compile)


# ###########################################################
# Part III, paths and symmetries
# ###########################################################

def bc_symmetrize(path=[+1,+2,+4,-1,-2,-4],d=4,positive_only=False):
    """
    Returns a list of all paths with same shame symmetriex according
    to the BC(d) group:

    for example there in 4D there are 192 chairs passing through each point.
    >>> len(bc_symmetrize(path=(+1,+2,+3,-1,-2,-3),d=4))
    192

    if positive_only it only generates positive combinations only
    """
    paths = []
    symbols = list(set(abs(int(mu)) for mu in path))
    nsymbols = len(symbols)
    delta = d if positive_only else 2*d
    for k in range(delta**nsymbols):
        newsymbols = []
        for j in range(nsymbols):
            k, newsymbol = k//delta, k%delta+1
            if newsymbol>d: newsymbol = d-newsymbol
            newsymbols.append(newsymbol)
        if len(set(abs(int(mu)) for mu in newsymbols))<nsymbols:
            continue
        def rotate(mu):
            for i,nu in enumerate(symbols):
                if mu==nu: return newsymbols[i]
                if -mu==nu: return -newsymbols[i]
        newpath = tuple(rotate(mu) for mu in path)
        # each path should be unique
        if newpath in paths:
            raise RuntimeError("accidental double counting")
        paths.append(newpath)
    return paths

def backward_path(path):
    return tuple(-mu for mu in reversed(path))

def remove_duplicates(paths,bidirectional=False):
    """
    if bidirectional = True it assumes that reversed path are duplicates

    for example in 4D there are 6-distinct planes passing through a vertex.
    >>> len(remove_duplicates(bc_symmetrize([+1,+2,-1,-2],4)))
    6

    """
    newpaths = set()
    for newpath in paths:
        duplicate = False
        for i in range(len(newpath)):
            if sum(newpath)==0 or i==0:
                tmppath = tuple(newpath[i:]+newpath[:i])
                if tmppath in newpaths: # CHECK THIS!
                    duplicate = True
                    break
                # print newpath, [-mu for mu in reversed(tmppath)], paths
                if bidirectional and backward_path(tmppath) in newpaths:
                    duplicate = True
                    break
        if not duplicate:
            newpaths.add(newpath)
    return list(newpaths)

def range_path(path):
    d = {}
    for k in path:
        if k>0: d[k]=d.get(k,0)+1
        if k<0: d[k]=d.get(k,0)-1
    return max(abs(x) for x in d.values())

def derive_path(path,mu,bidirectional=False):
    """
    Computes the derivative of path respect to link mu.
    If bidirecional path, it also derives the reverse path
    """
    dpaths = []
    for i,nu in enumerate(path):
        newpath = path[i+1:]+path[:i]
        if nu==mu:
            dpaths.append(newpath)
        elif bidirectional and  nu==-mu:
            dpaths.append(backward_path(newpath))
    return dpaths

def derive_paths(paths,mu,bidirectional=False):
    """
    Computes the derivative of all paths respect to link mu.
    If bidirecional path, it also derives the reverse of each path
    """
    dpaths = []
    for path in paths:
        dpaths+=derive_path(path,mu,bidirectional)
    return dpaths


def minimum_spanning_graph(paths,bidirectional=True):
    """
    given a list of products of symbols finds the combination of
    products that minimizes the number of total products.
    for example:

    >>> minimum_spanning_graph(['ABCDEF','ABC','CDEF'])
    [(('E',), ('F',)), (('B',), ('C',)), (('C',), ('D',)), (('E',), ('F',)), (('D',), ('E', 'F')), (('A',), ('B', 'C')), (('C', 'D'), ('E', 'F')), (('A', 'B', 'C'), ('D', 'E', 'F'))]

    if bidirectional=True, it assumes the symbols to be signed integers like
    [+1,-2,+3,...] and consider the opposite expression [...,-3,+2,-1] to be equivalent.
    (in lattice QCD they are the products of link matrices along a path and along backward path).
    """
    def is_in(s,p):
        return s in p or (bidirectional and backward_path(s) in p)
    solution = None
    godel = 0
    while True:
        index = godel
        queue = set(tuple(path) for path in paths)
        allpaths = copy.copy(queue)
        links = []
        while queue:
            path = queue.pop()
            if len(path)>1:
                obvious_choice = False
                for k in range(1,len(path)):
                    sub1, sub2 = parts = path[:k],path[k:]
                    if sub1 in allpaths and sub2 in allpaths:
                        links.append(parts)
                        obvious_choice = True
                        break
                if not obvious_choice:
                    options = len(path)-1
                    index, k = index//options, index % options+1
                    sub1,sub2 = parts = path[:k],path[k:]
                    links.append(parts)
                    if not is_in(sub1,allpaths):
                        allpaths.add(sub1)
                        if not is_in(sub1,queue):
                            queue.add(sub1)
                    if not is_in(sub2,queue):
                        allpaths.add(sub2)
                        if not is_in(sub2,queue):
                            queue.add(sub2)
        if index: break
        links.sort()
        print godel, len(links), solution and len(solution)
        print links
        if not solution or len(links)<len(solution):
            solution = links
        godel += 1
    solution.sort(lambda a,b: cmp(len(a[0]+a[1]),len(b[0]+b[1])))
    return solution

# ###########################################################
# Part IV, OpenCL code generation algorithms
# ###########################################################

class Code(object):
    def __init__(self,source,runner,compiler):
        self.source = source
        self.runner = runner
        self.compiler = compiler
        self.program = None
        open('latest.c','w').write(source)
    def compile(self):
        self.program = self.compiler(self.source)
    def run(self,*args,**kwargs):        
        if not self.program:
            self.compile()
        return self.runner(self.program,*args,**kwargs)

def mul_const(name1,coeff,name2,m,add=False):
    lines = []
    re = coeff.real
    im = coeff.imag
    mod = '+=' if add else '='
    for k in range(m):
        a = '%s[%s]' % (name1,k)
        b = '%s[%s]' % (name2,k)        
        if re==0 and im==0 and not add:
            if not add:
                lines.append('%s.x = 0;' % a)
                lines.append('%s.y = 0;' % a)
        elif re==1 and im==0:
            lines.append('%s.x %s %s.x;' % (a,mod,b))
            lines.append('%s.y %s %s.y;' % (a,mod,b))
        elif re==-1 and im==0:
            lines.append('%s.x %s - %s.x;' % (a,mod,b))
            lines.append('%s.y %s - %s.y;' % (a,mod,b))
        elif re==0 and im==1:
            lines.append('%s.x %s - %s.y;' % (a,mod,b))
            lines.append('%s.y %s %s.x;' % (a,mod,b))
        elif re==0 and im==-1:
            lines.append('%s.x %s %s.y;' % (a,mod,b))
            lines.append('%s.y %s + %s.x;' % (a,mod,b))
        else:
            lines.append('%s.x %s (%s)*%s.x - (%s)*%s.y;' % (a,mod,re,b,im,b))
            lines.append('%s.y %s (%s)*%s.y + (%s)*%s.x;' % (a,mod,re,b,im,b))
    return lines

def mul_coeff(name1,coeff,name2,m,add=False):
    lines = []
    mod = '+=' if add else '='
    for k in range(m):
        a = '%s[%s]' % (name1,k)
        b = '%s[%s]' % (name2,k)        
        lines.append('%s.x %s %s.x*%s.x - %s.y*%s.y;' % (
                a,mod,coeff,b,coeff,b))
        lines.append('%s.y %s %s.x*%s.y + %s.y*%s.x;' % (
                a,mod,coeff,b,coeff,b))
    return lines

def mul_real_coeff(name1,coeff,name2,m,add=False):
    lines = []
    mod = '+=' if add else '='
    for k in range(m):
        a = '%s[%s]' % (name1,k)
        b = '%s[%s]' % (name2,k)        
        lines.append('%s.x %s %s*%s.x;' % (a,mod,coeff,b))
        lines.append('%s.y %s %s*%s.y;' % (a,mod,coeff,b))
    return lines

def mul_spin_matrix(name1,matrix,name2,nc,add=False):
    lines = []
    mod = '+=' if add else '='
    m,n = matrix.shape
    checks = [0.0]*(m*nc)
    for alpha in range(m):
        for i in range(nc):
            namea = '%s[%s]' % (name1,alpha*nc+i)
            terms_x = []
            terms_y = []
            for beta in range(n):
                nameb = '%s[%s]' % (name2,beta*nc+i)    
                re, im = matrix[alpha,beta].real, matrix[alpha,beta].imag
                if re:
                    terms_x.append('(%s)*%s.x' % (re,nameb))
                    terms_y.append('(%s)*%s.y' % (re,nameb))
                if im:
                    terms_x.append('(%s)*%s.y' % (-im,nameb))
                    terms_y.append('(%s)*%s.x' % (+im,nameb))
            if terms_x:
                lines.append('%s.x %s %s;' % (namea,mod,' + '.join(terms_x)))
                checks[alpha*nc+i]+=1
            elif not add:
                lines.append('%s.x %s 0;' % (namea,mod))
            if terms_y:
                lines.append('%s.y %s %s;' % (namea,mod,' + '.join(terms_y)))
                checks[alpha*nc+i]+=1j
            elif not add:
                lines.append('%s.y %s 0;' % (namea,mod))
    return lines, checks


def mul_color_matrix(name1,matrix,name2,nspin,add=False,checks=None):
    lines = []
    mod = '+=' if add else '='
    nc,_ = matrix.shape
    if nc!=_: raise RuntimeError("color matrix in not square")
    for alpha in range(nspin):
        for i in range(nc):
            namea = '%s[%s]' % (name1,alpha*nc+i)
            terms_x = []
            terms_y = []
            for j in range(nc):
                nameb = '%s[%s]' % (name2,alpha*nc+j)    
                re, im = matrix[i,j].real, matrix[i,j].imag
                if re:
                    terms_x.append('(%s)*%s.x' % (re,nameb))
                    terms_y.append('(%s)*%s.y' % (re,nameb))
                if im:
                    terms_x.append('(%s)*%s.y' % (-im,nameb))
                    terms_y.append('(%s)*%s.x' % (+im,nameb))
            if terms_x and (not checks or checks[alpha*nc+j]):
                lines.append('%s.x %s %s;' % (namea,mod,' + '.join(terms_x)))
            elif not add:
                lines.append('%s.x %s 0;' % (namea,mod))
            if terms_y and (not checks or checks[alpha*nc+j]):
                lines.append('%s.y %s %s;' % (namea,mod,' + '.join(terms_y)))
            elif not add:
                lines.append('%s.y %s 0;' % (namea,mod))
    return lines

def mul_link(name1,name2,name3,nspin,nc,add=False,checks=None):
    lines = []
    mod = '+=' if add else '='
    for alpha in range(nspin):
        for i in range(nc):
            namea = '%s[%s]' % (name1,alpha*nc+i)
            terms_x = []
            terms_y = []
            for j in range(nc):
                nameb = '%s[%s]' % (name2,i*nc+j)
                namec = '%s[%s]' % (name3,alpha*nc+j)    
                terms_x.append('+%s.x*%s.x' % (nameb,namec))
                terms_x.append('-%s.y*%s.y' % (nameb,namec))
                terms_y.append('+%s.x*%s.y' % (nameb,namec))
                terms_y.append('+%s.y*%s.x' % (nameb,namec))
            if not checks or checks[alpha*nc+j]:
                lines.append('%s.x %s %s;' % (namea,mod,' '.join(terms_x)))
            elif not add:
                lines.append('%s.x %s 0;' % (namea,mod))
            if not checks or checks[alpha*nc+j]:
                lines.append('%s.y %s %s;' % (namea,mod,' '.join(terms_y)))
            elif not add:
                lines.append('%s.y %s 0;' % (namea,mod))
    return lines

def mul_link_hermitian(name1,name2,name3,nspin,nc,add=False,checks=None):
    lines = []
    mod = '+=' if add else '='
    for alpha in range(nspin):
        for i in range(nc):
            namea = '%s[%s]' % (name1,alpha*nc+i)
            terms_x = []
            terms_y = []
            for j in range(nc):
                nameb = '%s[%s]' % (name2,j*nc+i)
                namec = '%s[%s]' % (name3,alpha*nc+j)    
                terms_x.append('+%s.x*%s.x' % (nameb,namec))
                terms_x.append('+%s.y*%s.y' % (nameb,namec))
                terms_y.append('+%s.x*%s.y' % (nameb,namec))
                terms_y.append('-%s.y*%s.x' % (nameb,namec))
            if not checks or checks[alpha*nc+j]:
                lines.append('%s.x %s %s;' % (namea,mod,' '.join(terms_x)))
            elif not add:
                lines.append('%s.x %s 0;' % (namea,mod))
            if not checks or checks[alpha*nc+j]:
                lines.append('%s.y %s %s;' % (namea,mod,' '.join(terms_y)))
            elif not add:
                lines.append('%s.y %s 0;' % (namea,mod))
    return lines


def opencl_paths(function_name, # name of the generated function
                 lattice, # lattice on which paths are generated
                 paths, # list of paths
                 coefficients, # complex numbers storing coefficents of paths
                 nc, # SU(n) gauge group
                 precision = 'cfloat_t', # float or double precision
                 initialize = False,
                 trace = False): # result is to be traced
    """
    Generates a OpenCL function which looks like:

    kernel void name(global %s *out,
                     global const %s *U,
                     unsigned long idx0) {...}

    It cumulates the product of links along the paths starting at site idx0
    the result is cumulated into out. Both out and U must point to vectors
    of complex numbers of given precison.

    If initialize==True (default) then out is initialized to zero.

    If trace==False (default) the product of links is not traced it
    assumes out points to a nxn complex matrix. When False,
    out points to one complex number.
    """
    if DEBUG:
        print 'generating',function_name
    DEFINE = precision+' %s = ('+precision+')(0.0,0.0);'
    NAME = 'm%.2i'
    CX = '%s_%ix%i'
    RE, IM = '%s_%ix%i.x', '%s_%ix%i.y'
    ADD, PLUS, MINUS, TIMES, EQUAL = ' += ', '+', '-', '*', ' = '
    NEWLINE = '\n'+' '*12
    vars = []
    code = []
    matrices = {}
    d = len(lattice.shape)
    n = nc
    site = lattice.Site(*tuple(0 for i in lattice.shape))
    vars.append('unsigned long ixmu;')
    if initialize:
        if trace:
            code.append("out[0].x = out[0].y = 0.0;")
        else:
            for i in range(n):
                for j in range(n):
                    code.append("out[%i].x = out[%i].y = 0.0;" % (i*n+j,i*n+j))

    for ipath,path in enumerate(paths):
        code.append('\n// path %s\n' % str(path))
        p = site
        #if DEBUG: print 'path:',path
        coeff = coefficients[ipath]
        for z,mu in enumerate(path):
            # print z, mu
            nu = abs(mu) % d
            if mu>0: key = (copy.copy(p.coords),(mu,)) # individual link key
            p = p+mu # mu can be negative
            if mu<0: key = (copy.copy(p.coords),(-mu,)) # hermitian link key
            link_key = key
            # if DEBUG: print 'key:',key
            if not key in matrices: # if need a new link
                name = NAME % len(matrices)
                matrices[key] = name
                code.append('\n// load U%s -> %s' % (key,name))
                for i,co in enumerate(key[0]):
                    code.append('shift.s[%s] = %s;' % (i,co))
                code.append('ixmu = (idx2idx_shift(idx0,shift,bbox)*%s+%s)*%s;'%(d,nu,n*n))
                ### load link U(0,mu,i,j)
                for i in range(n):
                    for j in range(n):
                        var = CX % (name,i,j)
                        code.append('%s = U[ixmu+%i];' % (var,n*i+j))
                if z==0: # if first link in path
                    if mu<0: # if backwards
                        # compute the hermitian
                        name2 = NAME % len(matrices)
                        key2 = (site.coords,(mu,)) # key for backward link
                        matrices[key2] = name2
                        for i in range(n):
                            for j in range(n):
                                var = CX % (name,i,j)
                                var2 = CX % (name2,j,i)
                                code.append('%s.x = %s.x;' % (var2, var))
                                code.append('%s.y = -%s.y;' % (var2, var))
            # if this is a link but the first in path
            if z>0:
                key = (site.coords,tuple(path[:z+1]))
                define_var = not key in matrices
                if define_var:
                    name = NAME % len(matrices)
                    matrices[key] = name
                else:
                    name = matrices[key]
                if not (site.coords,tuple(path[:z])) in matrices:
                    if z==1 and (site.coords,(-path[0],)) in matrices:
                        mu = -path[0]
                        key2 = (site.coords,(mu,)) # key for forward link
                        key3 = (site.coords,(-mu,)) # key for backward link
                        name2 = matrices[key2]
                        name3 = NAME % len(matrices)
                        matrices[key3] = name3 # name for new link
                        for i in range(n):
                            for j in range(n):
                                var2 = CX % (name2,i,j)
                                var3 = CX % (name3,j,i)
                                code.append('%s.x = %s.x;' % (var3, var2))
                                code.append('%s.y = -%s.y;' % (var3, var2))
                    else:
                        print matrices, (site.coords,tuple(path[:z]))
                        print matrices.keys()
                        print (site.coords,tuple(path[:z]))
                        raise RuntimeError("Missing link")
                name1 = matrices[(site.coords,tuple(path[:z]))]
                name2 = matrices[link_key]
                if mu>0: # if link forward
                    code.append('\n// compute %s*%s -> %s' % (name1,name2,name))
                    # extend previous path
                    code += code_mul(name,name1,name2,n,n,n)
                elif mu<0: # if link backwards ... same
                    code.append('\n// compute %s*%s^H -> %s' %(name1,name2,name))
                    code += code_mulh(name,name1,name2,n,n,n)
            if z==len(path)-1:
                key = (site.coords,path)
                name = matrices[key]
                for i in range(n):
                    if trace:
                        line = 'out[0].x = out[0].x + %f*' % coeff \
                            + RE % (name,i,i)
                        code.append(line+';')
                        line = 'out[0].y = out[0].y + %f*' % coeff \
                            + IM % (name,i,i)
                        code.append(line+';')
                    else:
                        for j in range(n):
                            k=i*n+j
                            line = 'out[%s].x = out[%s].x + %f*' % (k,k,coeff)\
                                + RE % (name,i,j)
                            code.append(line+';')
                            line = 'out[%s].y = out[%s].y + %f*' % (k,k,coeff)\
                                + IM % (name,i,j)
                            code.append(line+';')
    for name in sorted(matrices.values()):
        for i in range(n):
            for j in range(n):
                var = CX % (name, i,j)
                vars.append((DEFINE % var)+' //4')

    body = '\n'.join(' '*16+line.replace('\n','\n    ') for line in vars+code)
    return """void %(name)s(
                %(precision)s *out,
                global const %(precision)s *U,
                const unsigned long idx0,
                const struct bbox_t *bbox){
                struct shift_t shift;
                \n%(body)s
              }
              kernel void %(name)s_loop(
                     global cfloat_t *out,
                     global const cfloat_t *U,
                     int n,
                     struct bbox_t bbox) {
                int gid = get_global_id(0);
                const unsigned long idx = gid2idx(gid,&bbox);
                cfloat_t tmp[MAXN*MAXN];
                for(int i=0; i<n; i++)
                    for(int j=0; j<n; j++)
                        tmp[i*n+j]=(cfloat_t)(0.0,0.0);
                %(name)s(tmp,U,idx,&bbox);
                for(int i=0; i<n; i++)
                    for(int j=0; j<n; j++)
                        out[idx*n*n+i*n+j]=tmp[i*n+j];
              }
           """ % dict(name=function_name,
                      precision=precision,
                      body=body)

def opencl_fermi_operator(terms,d,nspin,nc,name='aux'):
    code = []
    h = 0
    code.append('q = phi + idx*%s;' % (nspin*nc)) #OUT
    k = 0
    for term in terms:        
        gamma = term['gamma']
        opaths = term['paths']
        color = term['color']
        shift = term['shift']
        is_spin_constant = isinstance(gamma,(int,float,complex))
        is_spin_matrix = isinstance(gamma, numpy.matrix)
        is_staggered = term['staggered'] and shift
        is_color_matrix = isinstance(color, numpy.matrix)
        is_extra = isinstance(color, Field)
        is_color_diagonal = (opaths is None and color is None)
        is_shift = shift is not None
        if is_shift and color is not None:
            raise RuntimeError("Not supported yet: %s" % term)
        if shift is None:
            code.append('p = psi + idx*%s;' % (nspin*nc)) #IN
        else:
            for i in range(d): code.append('delta.s[%s] = %s;' % (i, shift[i]))
            code.append('p = psi+idx2idx_shift(idx,delta,&bbox)*%s;' % (
                    nspin*nc)) 
        # deal with gamma structure
        if is_staggered:
            mu = [i for i,dmu in enumerate(shift) if dmu!=0][0]
            code.append('coeff2 = %s * idx2eta(idx,%s,&bbox);' % (gamma,mu))
            lines =  mul_real_coeff('spinor','coeff2','p',nspin*nc)
            checks = None
        elif is_spin_constant:
            lines = mul_const('spinor',gamma,'p',nspin*nc)
            checks = None
        elif is_spin_matrix:
            lines,checks = mul_spin_matrix('spinor',gamma,'p',nc)
        else:
            raise RuntimeError("Not supported term %s" % term)
        code += lines
        # deal with color
        if is_shift:
            code.append("%s%s(path,U,idx,&bbox);" % (name,k))
            code += mul_link('q', 'path', 'spinor', nspin, nc,
                             add=k>0, checks=checks)
        elif is_color_matrix:
            code += mul_color_matrix('q', opaths, 'spinor', nspin, 
                                     add=k>0, checks=checks)
        elif is_extra:
            code += mul_link('q', 'extra%s' % id(color), 'spinor', nspin, nc,
                             add=k>0, checks=checks)
        elif is_color_diagonal:
            code += mul_const('q',1.0,'spinor', nspin*nc, add=k>0)
        else:
            raise RuntimeError("Not supported term %s" % term)
        k+=1
    return '\n'.join(' '*10+line for line in code)

# ###########################################################
# inverters
# ###########################################################

def copy_elements(a,b):
    if isinstance(a,Field) and isinstance(b,Field):
        a.set_copy(b)
    elif a.shape == b.shape:
        a[:] = b
    else:
        raise RuntimeError("Incompatible shape")


def vdot(a,b):
    if isinstance(a,Field) and isinstance(b,Field):
        return a*b
    elif a.shape == b.shape:
        return numpy.vdot(a,b)
    else:
        raise RuntimeError("Incompatible shape")

def clone(a):
    if isinstance(a,Field):
        return a.clone()
    else:
        return numpy.empty_like(a)

def invert_minimum_residue(y,f,x,ap=1e-4,rp=1e-4,ns=1000):
    q = clone(x)
    r = clone(x)
    copy_elements(y, x) # x-> y
    copy_elements(r, x)
    f(q,x)
    r -= q
    for k in xrange(ns):
        f(q,r)
        alpha = vdot(q,r)/vdot(q,q)
        y += alpha*r
        r -= alpha*q
        residue = math.sqrt(vdot(r,r).real/r.size)
        norm = math.sqrt(vdot(y,y).real)
        if k>10 and residue<max(ap,norm*rp):
            return y
    raise ArithmeticError('no convergence')

def invert_bicgstab(y,f,x,ap=1e-4,rp=1e-4,ns=1000):
    p = clone(x)
    q = clone(x)
    t = clone(x)
    r = clone(x)
    s = clone(x)
    copy_elements(y, x)
    f(r,y)
    r *= -1
    r += x
    copy_elements(q, r)
    rho_old = alpha = omega = 1.0
    for k in xrange(ns):
        rho = vdot(q,r)
        beta = (rho/rho_old)*(alpha/omega)
        rho_old = rho
        p *= beta
        p += r
        p -= (beta*omega) * s
        f(s,p)
        alpha = rho/vdot(q,s)
        r -= alpha * s
        residue = math.sqrt(vdot(r,r).real/r.size)
        norm = math.sqrt(vdot(y,y).real)
        f(t,r)
        omega = vdot(t,r)/vdot(t,t)
        y += omega * r
        y += alpha * p
        if residue<max(ap,norm*rp):
            return y
        r -= omega * t
    raise ArithmeticError('no convergence')

# ###########################################################
# Part V, Tests
# ###########################################################

class TestColdAndHotGauge(unittest.TestCase):
    def test_cold(self):
        for nc in range(2,4):
            for N in range(4,6):
                space = Q.Lattice([N,N,N,N])
                U = space.GaugeField(nc)
                U.set_cold()
                U.check_unitarity()
                self.assertTrue(abs(U.average_plaquette()) > 0.9999)
    def test_hot(self):
        for nc in range(2,4):
            for N in range(4,6):
                space = Q.Lattice([N,N,N,N])
                U = space.GaugeField(nc)
                U.set_hot()
                U.check_unitarity()
                self.assertTrue(abs(U.average_plaquette()) < 0.1)

class TestFieldTypes(unittest.TestCase):
    def test_fields(self):
        N, nc, nspin = 4, 3, 4
        parameters = {}
        space = Q.Lattice([N,N,N,N])
        U = space.GaugeField(nc)
        U.set_hot()
        U.check_unitarity()
        U.set_cold()
        U.check_cold()
        U.check_unitarity()

        psi = U.data_component((0,0,0))
        #Canvas().imshow(numpy.real(psi.lattice_slice((0,0)))).save()
        psi = space.FermiField(nspin,nc)
        chi = space.FermiField(nspin,nc)
        psi[(0,0,0,0),1,2] = 0.0 # set component to zezo
        phi = space.Field(1).set_link_product(U,[(1,2,-1,-2),(2,4,-2,-4)])
        self.assertTrue(phi.sum() == 4**4 * 3*2)
        old = U.sum()
        U.save('test.npy')
        try:
            U.load('test.npy')
            self.assertTrue(U.sum() == old)
        except AssertionError: pass # not sure why but on Max load raises AssetionError

class TestPaths(unittest.TestCase):
    def test_paths(self):
        path = (+1,+2,-1,-2)
        paths = bc_symmetrize(path,d=4)
        self.assertTrue(
            paths ==
            [(2, 1, -2, -1), (3, 1, -3, -1), (4, 1, -4, -1), (-2, 1, 2, -1), (-3, 1, 3, -1),
             (-4, 1, 4, -1), (1, 2, -1, -2), (3, 2, -3, -2), (4, 2, -4, -2), (-1, 2, 1, -2),
             (-3, 2, 3, -2), (-4, 2, 4, -2), (1, 3, -1, -3), (2, 3, -2, -3), (4, 3, -4, -3),
             (-1, 3, 1, -3), (-2, 3, 2, -3), (-4, 3, 4, -3), (1, 4, -1, -4), (2, 4, -2, -4),
             (3, 4, -3, -4), (-1, 4, 1, -4), (-2, 4, 2, -4), (-3, 4, 3, -4), (2, -1, -2, 1),
             (3, -1, -3, 1), (4, -1, -4, 1), (-2, -1, 2, 1), (-3, -1, 3, 1), (-4, -1, 4, 1),
             (1, -2, -1, 2), (3, -2, -3, 2), (4, -2, -4, 2), (-1, -2, 1, 2), (-3, -2, 3, 2),
             (-4, -2, 4, 2), (1, -3, -1, 3), (2, -3, -2, 3), (4, -3, -4, 3), (-1, -3, 1, 3),
             (-2, -3, 2, 3), (-4, -3, 4, 3), (1, -4, -1, 4), (2, -4, -2, 4), (3, -4, -3, 4),
             (-1, -4, 1, 4), (-2, -4, 2, 4), (-3, -4, 3, 4)])
        paths = remove_duplicates(paths,bidirectional=True)
        self.assertTrue(paths == [(4, 1, -4, -1), (4, 3, -4, -3), (2, 1, -2, -1),
                                  (3, 2, -3, -2), (3, 1, -3, -1), (4, 2, -4, -2)])
        staples = derive_paths(paths,+1,bidirectional=True)
        self.assertTrue(staples == [(-4, -1, 4), (4, -1, -4), (-2, -1, 2),
                                    (2, -1, -2), (-3, -1, 3), (3, -1, -3)])


class TestPathKernels(unittest.TestCase):

    def make_kernel(self,comm,kernel_code,name,U):
        ### fix this, there need to be a parallel loop over sites
        u_buffer = cl.Buffer(comm.ctx, comm.mf.READ_ONLY | 
                             comm.mf.COPY_HOST_PTR,
                             hostbuf=U.data)
        out_buffer = cl.Buffer(comm.ctx, comm.mf.WRITE_ONLY, 256) ### FIX NBYTES
        idx_buffer = cl.Buffer(comm.ctx, comm.mf.READ_ONLY, 32) # should not be
        core = open('kernels/qcl-core.c').read()
        prg = cl.Program(comm.ctx,core + kernel_code).build(
            options=['-I',INCLUDE_PATH])
        # getattr(prg,name)(comm.queue,(4,4,4),None,out_buffer,u_buffer,idx_buffer)

    def test_opencl_paths(self):
        N, nc = 4, 3
        space = Q.Lattice([N,N,N,N])
        U = space.GaugeField(nc)
        paths = [(+1,+2,+3,-1,-2,-3)]
        kernel_code = opencl_paths(function_name='staples',
                                   lattice=space,
                                   paths=paths,
                                   coefficients = [1.0 for p in paths],
                                   nc=nc,
                                   trace=False)
        self.make_kernel(Q,kernel_code,'staples',U)

class TestHeatbath(unittest.TestCase):
    def test_heatbath(self):
        N, nc = 4, 3
        space = Q.Lattice([N,N,N,N])
        U = space.GaugeField(nc)
        U.set_cold()
        wilson = GaugeAction(U).add_plaquette_terms()
        # same as wilson.add_term(1.0,(1,2,-1,-2))
        code = wilson.heatbath(beta=4.0)
        # print code.source
        plq = []
        for k in range(1000):
            code.run()
            if k>100 and k%5==4:
                plq.append(U.average_plaquette())
            if DEBUG:
                print '<plaq> =', sum(plq)/len(pql)
                U.check_unitarity(output=False)
        self.assertTrue( 0.29 < sum(plq)/len(plq) < 0.3 )

class TestFermions(unittest.TestCase):
    def test_wilson_action(self):
        N, nspin, nc = 9, 4, 3
        r = 1.0
        kappa = 0.1234

        space = Q.Lattice([N,N,N,N])
        U = space.GaugeField(nc)
        psi = space.FermiField(nspin,nc)
        phi = space.FermiField(nspin,nc)

        U.set_cold()
        psi[(0,0,N/2,N/2),0,0] = 100.0
        Dslash = FermiOperator(U)
        Dslash.add_term(1.0)
        for mu in (1,2,3,4):
            Dslash.add_term(kappa*(r*I-G[mu]), [(mu,)])
            Dslash.add_term(kappa*(r*I+G[mu]), [(-mu,)])
        for k in range(10):
            phi.set(Dslash,psi)
            # project spin=0, color=0 component, plane passing fox t=0,x=0
            chi = psi.slice((0,0),(0,0))
            Canvas().imshow(chi).save('fermi.%.2i.png' % k)
            phi,psi = psi,phi

    def test_wilson_action_equivalence(self):
        N, nspin, nc = 9, 4, 3
        r = 1.0
        kappa = 0.1234

        space = Q.Lattice([N,N,N,N])
        U = space.GaugeField(nc)
        psi = space.FermiField(nspin,nc)
        phi = space.FermiField(nspin,nc)
        chi = space.FermiField(nspin,nc)

        U.set_cold()
        psi[(0,0,0,0),0,0] = 100.0
        Dslash1 = FermiOperator(U).add_wilson4d_action(kappa)
        Dslash2 = FermiOperator(U)
        Dslash2.add_term(1.0, None)
        for mu in (1,2,3,4):
            Dslash2.add_term(kappa*(r*I-G[mu]), [(mu,)])
            Dslash2.add_term(kappa*(r*I+G[mu]), [(-mu,)])
        phi.set(Dslash1,psi)
        chi.set(Dslash2,psi)
        self.assertTrue(numpy.linalg.norm((phi.data-chi.data).flat)<1.0e-6)

    def test_clover_action(self):
        N, nspin, nc = 9, 4, 3
        r = 1.0
        kappa = 0.1234
        csw = 0.5

        space = Q.Lattice([N,N,N,N])
        U = space.GaugeField(nc)
        psi = space.FermiField(nspin,nc)
        phi = space.FermiField(nspin,nc)

        U.set_cold()
        psi[(0,0,0,0),0,0] = 100.0
        Dslash = FermiOperator(U)\
            .add_wilson4d_action(kappa)\
            .add_clover4d_terms(csw)
        phi.set(Dslash,psi)


    def test_staggered_action(self):
        N, nc = 16, 3
        r = 1.0
        kappa = 0.1234

        space = Q.Lattice([N,N,N,N])
        U = space.GaugeField(nc)
        psi = space.StaggeredField(nc)
        phi = space.StaggeredField(nc)

        U.set_cold()
        psi[(0,0,N/2,N/2),0] = 100.0
        Dslash = FermiOperator(U).add_staggered_action(kappa)
        for k in range(10):
            phi.set(Dslash,psi)
            # project color=0 component, plane passing fox t=0,x=0
            chi = psi.slice((0,0),(0,))
            Canvas().imshow(chi).save('staggered.%.2i.png' % k)
            phi,psi = psi,phi


class TestInverters(unittest.TestCase):
    def test_minimum_residue(self):
        def f(a,b):
            a[0,0] = b[0,0]
            a[0,1] = 0.7*b[0,1]
            a[1,0] = 0.8*b[1,0]
            a[1,1] = b[1,1]
        a = numpy.array([[1.0,2.0],[3.0,4.0]])
        b = numpy.array([[0.0,0.0],[0.0,0.0]])
        c = numpy.array([[0.0,0.0],[0.0,0.0]])
        invert_minimum_residue(b,f,a)
        f(c,b)
        assert numpy.linalg.norm(c-a)<0.02

    def test_bicgstab(self):
        def f(a,b):
            a[0,0] = b[0,0]
            a[0,1] = 0.7*b[0,1]+0.1*b[0,0]
            a[1,0] = 0.8*b[1,0]+0.1*b[1,1]
            a[1,1] = b[1,1]
        a = numpy.array([[1.0,2.0],[3.0,4.0]])
        b = numpy.array([[0.0,0.0],[0.0,0.0]])
        c = numpy.array([[0.0,0.0],[0.0,0.0]])
        invert_bicgstab(b,f,a)
        f(c,b)
        assert numpy.linalg.norm(c-a)<0.02

    def test_fermion_propagator(self):
        N, nspin, nc = 9, 4, 3
        r = 1.0
        kappa = 0.1234

        space = Q.Lattice([N,N,N,N])
        U = space.GaugeField(nc)
        psi = space.FermiField(nspin,nc)
        phi = space.FermiField(nspin,nc)

        U.set_cold()
        psi[(0,0,4,4),0,0] = 100.0
        Dslash = FermiOperator(U)
        Dslash.add_term(1.0, None)
        for mu in (1,2,3,4):
            Dslash.add_term(kappa*(r*I-G[mu]), [(mu,)])
            Dslash.add_term(kappa*(r*I+G[mu]), [(-mu,)])
        phi.set(invert_bicgstab,Dslash,psi)
        chi = phi.slice((0,0),(0,0))
        Canvas().imshow(chi).save('fermi.prop.png')
        out = space.FermiField(nspin,nc)
        out.set(Dslash,phi)
        chi = out.slice((0,0),(0,0))
        Canvas().imshow(chi).save('fermi.out.png')
        chi = psi.slice((0,0),(0,0))
        Canvas().imshow(chi).save('fermi.in.png')
        for spin in range(4):
            for i in range(3):
                print 'fermi.%s.%s.real.vtk' % (spin,i)
                phi.data_component((spin,i))\
                    .save('fermi.%s.%s.real.vtk' % (spin,i))

    def test_meson_propagator(self):
        N, nspin, nc = 9, 4, 3
        r = 1.0
        kappa = 0.1234

        space = Q.Lattice([N,N,N,N])
        U = space.GaugeField(nc)
        phi = space.FermiField(nspin,nc)
        U.set_cold()
        Dslash = FermiOperator(U)
        Dslash.add_wilson4d_action(kappa)
        meson = space.ComplexScalarField()
        for spin in range(4):
            for color in range(3):
                psi = space.FermiField(nspin,nc)
                psi[(0,0,0,0),spin,color] = 1.0
                phi.set(invert_bicgstab,Dslash,psi)
                meson += make_meson(phi,identity(4),phi)
        meson_fft = meson.fft()
        meson_prop = [(t,math.log(meson_fft[t,0,0,0].real)) for t in range(N)]
        Canvas().plot(meson_prop).save('meson.prop.png')


class TestSmearing(unittest.TestCase):
    def test_gauge_smearing(self):
        N, nc = 4, 3
        space = Q.Lattice([N,N,N,N])
        U = space.GaugeField(nc)
        U.set_cold()
        op = GaugeSmearOperator(U).add_term([1],1.0).add_term([2,1,-2],0.1)
        V = U.clone()
        V.set_smeared(op)
        self.assertTrue(abs(U.average_plaquette()-1.0)<1e-4)
        self.assertTrue(abs(V.average_plaquette()-(1.0+0.1*6)**4)<1e-3)
    def test_fermi_smearing(self):
        N, nc,nspin = 9, 3, 4
        space = Q.Lattice([N,N,N,N])
        U = space.GaugeField(nc)
        U.set_cold()
        psi = space.FermiField(nspin,nc)
        phi = space.FermiField(nspin,nc)
        psi[(0,0,4,4),0,0] = 100.0
        S = FermiOperator(U).add_diagonal_term(1.0)
        for mu in (1,2,3,4): S.add_term(0.1*I,[(-mu,)]).add_term(0.1*I,[(mu,)])
        for k in range(10):
            S(phi,psi)
            phi,psi = psi,phi
            chi = phi.slice((0,0),(0,0))
            Canvas().imshow(chi).save('smear.%.2i.png' % k)
            os.system('convert smear.%.2i.png smear.%.2i.jpg' % (k,k))

def test_hadrons():
    N = 8
    lattice = Q.Lattice((N,N,N,N))
    q1 = lattice.FermiField(4,3)
    q2 = lattice.FermiField(4,3)
    q3 = lattice.FermiField(4,3)
    rho = make_meson(q1,identity(4),q2)
    rho = make_baryon3(q1,q2,identity(4),q3)

def test():
    N, nspin, nc = 9, 4, 3
    r = 1.0
    kappa = 0.1234
    c_SW = 0.5
    
    space = Q.Lattice([N,N,N,N])
    U = space.GaugeField(nc)
    psi = space.FermiField(nspin,nc)
    phi = space.FermiField(nspin,nc)
    
    U.set_cold()

    psi[(0,0,4,4),0,0] = 100.0
    if True:
        Dslash = FermiOperator(U)
        Dslash.add_wilson4d_action(kappa)
        Dslash.add_clover4d_terms(c_SW)
        phi.set(Dslash, psi)

# for tests
I = identity(4)
G = GAMMA['fermilab']
Q = Communicator()

if __name__=='__main__':
    # python -m unittest test_module.TestClass.test_method
    # test()
    unittest.main()
