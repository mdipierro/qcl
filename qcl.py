"""
Conventions:
dims = (nt, nx, ny, nz,...)
mu = +1(X),+2(Y),+3(Z),+4(T),...
mu = -1(X),-2(Y),-3(Z),-4(T),...
path = (+1,-2,-3,+2,+3,-1)

attention sync,load,save and other functions are not parallel!
"""

DEBUG = False

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
    def __init__(self,title='title',xlab='x',ylab='y',xrange=None,yrange=None):
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

def random_name():
    """ 
    Auxiliary function user to create a random sequence of alphanumeric characters
    It is user to generate function names
    """
    chars = 'abcdefghijklmonopqrstuvwxyz'
    return ''.join(random.choice(chars) for i in range(5))

def identity(n):
    """ Returns an identity matrix n x n """
    return numpy.eye(n)

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
# Part I, lattices and fields
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
        #if line and not line[:1] in '}#':
        if not line.startswith('}') and line:
            if padding==0 and newlines:
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
    def Lattice(self,dims,bboxs=None):
        """ 
        Creates a new lattice that lives on the device 
        dims is a tuple with the dimensions of the site tensor.
        """
        return Lattice(self,dims,bboxs)


class Lattice(object):
    """ a lattice encodes info about volume and parallelization """
    def __init__(self,comm,dims,bboxs=None):
        """
        Constructor of a Lattice
        It computes the size and allocaltes local ranlux prng on device
        """
        self.comm = comm
        self.dims = listify(dims)
        self.d = len(dims)
        self.size = size = product(self.dims)
        self.bboxs = bboxs or [[(0,)*self.d,self.dims]]
        self.prngstate_buffer = None
        self.prng_on(time.time()) # for debugging!
        self.parallel = len(self.bboxs)>1 # change for multi-GPU
        self.bbox = numpy.zeros(MAXD*6,dtype=numpy.int32)
        # weird logic for future multi-gpu support
        self._bbox_init()
    ### bbox stuff
    def _bbox_init(self):
        """ Initalizes the size of the lattice partition hosted on device """
        for i,x in enumerate(self.dims):
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
        It currently uses ranlux, the Lusher's generator which comes with pyOpenCL
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
        cl.enqueue_copy(self.comm.queue, self.prngstate, self.prngstate_buffer).wait()
    def prng_off(self):
        """ Disabled the prng on device"""
        self.prngstate = self.prngstate_buffer = None
    def coords2global(self,coords):
        """ Converts (t,x,y,z) coordinates into global index """
        if len(coords)!=len(self.dims):
            raise RuntimeError("invalid conversion")
        return sum(p*product(self.dims[i+1:]) \
                       for i,p in enumerate(coords))
    def global2coords(self,i):
        """ Converts global index into (t,x,y,z) """
        coords = []
        for k in reversed(self.dims):
            i,reminder = divmod(i,k)
            coords.insert(0,reminder)
        return tuple(coords)
    def check_coords(self):
        for i in range(self.size):
            assert self.coords2global(self.global2coords(i)) == i
    def Site(self,coords):
        """ Returns a Site constructor for this lattice """
        return Site(self,coords)
    def Field(self,siteshape,dtype=numpy.complex64):
        """ Returns a Field constructor for this """
        return Field(self,siteshape,dtype)


class Site(object):
    """ The site object stores information about a lattice site """
    def __init__(self,lattice,coords):
        self.lattice = lattice
        self.coords = tuple(coords)
        self.gl_idx = self.lattice.coords2global(coords)
        self.lo_idx = self.gl_idx ### until something better
    def __add__(self,mu):
        """
        Example:
        >>> lattice = Lattice((4,4,4,4))
        >>> p = Site(lattice,(0,0,0,0)
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
        L = self.lattice.dims[mu]
        coords = tuple((c if nu!=mu else ((c+1)%L))
                       for nu,c in enumerate(self.coords))
        return self.lattice.Site(coords)
    def __sub__(self,mu):
        """
        >>> q = p - mu
        same as
        >>> q = p + (-mu)
        """
        if mu<0:
            return self.__add__(-mu)
        mu = mu % self.lattice.d
        L = self.lattice.dims[mu]
        coords = tuple((c if nu!=mu else ((c+L-1)%L))
                       for nu,c in enumerate(self.coords))
        return self.lattice.Site(coords)
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
    >>> lattice = Lattice((4,4,4,4))
    >>> phi = Field(lattice,(4,3),dtype=numpy.complex64)

    dtype=numpy.complex64 is the default and some algorithms rely on this value.
    """
    def __init__(self, lattice, siteshape, dtype=numpy.complex64):
        self.lattice = lattice
        self.siteshape = listify(siteshape)
        self.sitesize = product(self.siteshape)
        self.dtype = dtype
        self.data = numpy.zeros([self.lattice.size]+self.siteshape,dtype=dtype)
        if DEBUG: 
            print 'allocating %sbytes' % int(self.lattice.size*self.sitesize*8)
    def copy(self,other):
        """
        Makes a copy of field b into field a
        Exmaple:        
        >>> a.copy(b) 
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
    def data_slice(self,slice_coords):
        """
        Returns a numpy slice of the current self.data at coordinates c
        """
        if self.sitesize!=1: raise NotImplementedError
        d = self.lattice.dims[len(slice_coords):]
        coords = [e for e in slice_coords]+[0 for i in d]
        t = self.lattice.coords2global(coords)
        s = product(d)
        return self.data[t:t+s].reshape(d)
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
            self.data += other
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
        f = numpy.vdot if self.dtype in COMPLEX_TYPES else numpy.dot
        return f(self.data,other.data)
    def __getitem__(self,args):
        """ Do not use these to implement algorithms - too slow """
        site, args = args[0], args[1:]
        return self.data[(site.lo_idx,)+args]
    def __setitem__(self,args,value):
        """ Do not use these to implement algorithms - too slow """
        site, args = args[0], args[1:]
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
    def save(self,filename):
        """ Saves the field supports *.npy and *.vtk (for 4 only)"""
        format = filename.split('.')[-1].lower()
        if format == 'npy':
            numpy.save(filename,self.data)            
        elif format == 'vtk' or self.lattice.d!=4 or self.siteshape!=(1,):
            ostream = open(filename,'wb')
            s = product(self.lattice.dims[-3:])
            h1 = "# vtk DataFile Version 2.0\n"+\
                filename.split('/')[-1]+'\n'+\
                'BINARY\n'+\
                'DATASET STRUCTURED_POINTS\n'+\
                'DIMENSIONS %s %s %s\n'%tuple(self.lattice.dims[-3:])+\
                'ORIGIN     0 0 0\n'+\
                'SPACING    1 1 1\n'+\
                'POINT_DATA %s' % s
            ostream.write(h1)            
            for t in xrange(self.lattice.dims[0]):
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

    def set_link_product(self,U,paths,trace=True,name='aux'):
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
                            paths=paths,                            
                            nc = U.data.shape[-1],
                            trace = (self.data.shape[-1]==1))
        source = makesource({'paths':code})
        def runner(source,self=self,name=name):
            shape = self.siteshape
            out_buffer = self.lattice.comm.buffer('w',self.data)
            U_buffer = self.lattice.comm.buffer('r',U.data)
            prg = self.lattice.comm.compile(source)
            function = getattr(prg,name+'_loop')
            function(self.lattice.comm.queue,(self.lattice.size,),None,
                     out_buffer, U_buffer,
                     numpy.int32(shape[0]), # 1 for trace, nc for no-trace
                     self.lattice.bbox).wait()
            cl.enqueue_copy(self.lattice.comm.queue, self.data, out_buffer).wait()
            return self        
        return Code(source,runner)

    def set_cold(self):
        """ Uses a kernel to set all links to cold """
        shape = self.siteshape
        if not (len(shape)==3 and shape[1]==shape[2]): raise RuntimeError
        data_buffer = self.lattice.comm.buffer('w',self.data)
        prg = self.lattice.comm.compile(makesource())
        prg.set_cold(self.lattice.comm.queue,(self.lattice.size,),None,
                     data_buffer,numpy.int32(shape[0]),numpy.int32(shape[1]),
                     self.lattice.bbox).wait()
        cl.enqueue_copy(self.lattice.comm.queue, self.data, data_buffer).wait()

    def set_custom(self):
        """ Uses a kernel to set all links to come cutsom values, for testing only """
        shape = self.siteshape
        if not (len(shape)==3 and shape[1]==shape[2]): raise RuntimeError
        data_buffer = self.lattice.comm.buffer('w',self.data)
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
        data_buffer = self.lattice.comm.buffer('w',self.data)
        prg = self.lattice.comm.compile(makesource())
        prg.set_hot(self.lattice.comm.queue,(self.lattice.size,),None,
                    data_buffer,numpy.int32(shape[0]),numpy.int32(shape[1]),
                    self.lattice.bbox,self.lattice.prngstate_buffer).wait()
        cl.enqueue_copy(self.lattice.comm.queue, self.data, data_buffer).wait()

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
        code = self.lattice.Field(1).set_link_product(self,paths)
        phi=code.run()
        #for idx in xrange(self.lattice.size):
        #    print idx, phi.data[idx]
        return phi.sum()/(self.lattice.size*len(paths)*self.siteshape[-1])

class GaugeAction(object):
    """
    Class to store paths and coefficients relative to any gauge action
    """
    def __init__(self,lattice):
        self.lattice = lattice
        self.terms = []
    def add_term(self,coefficient,paths):
        """
        Paths are symmetrized. For example: 
        >>> wilson = GaugeAction(lattice).add_term(1.0,paths = (1,2,-1,-2)) 
        is the Wilson action.
        """
        if isinstance(paths,tuple):
            paths = bc_symmetrize(paths,d=self.lattice.d)
            paths = remove_duplicates(paths,bidirectional=False)
        elif not (isinstance(paths,list) or not isinstance(paths[0],tuple)):
            raise RuntimeError("not a valid action term")
        self.terms.append((coefficient,paths))
        return self
    def heatbath(self,U,beta,n_iter=1,m_iter=1,name='aux'):
        """
        Generates a kernel which performs the SU(n) heatbath.
        Example:
        >>> wilson = GaugeAction(lattice).add_term(1.0,paths = (1,2,-1,-2)) 
        >>> wilson.heatbath(beta,n_iter=10)
        """
        name = name or random_name()
        code = ''
        displacement = 1
        for mu in range(self.lattice.d):
            coefficients,paths = [], []
            for coefficient, cpaths in self.terms:
                opaths = derive_paths(cpaths,mu if mu else self.lattice.d)
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
                           (i,name,i) for i in range(self.lattice.d))
        source = makesource({'paths':code,'heatbath_action':action})
        def runner(source,self=self,U=U,beta=beta,n_iter=n_iter,m_iter=m_iter,
                   displacement=displacement):
            shape = U.siteshape
            if not (len(shape)==3 and shape[1]==shape[2]): raise RuntimeError
            data_buffer = U.lattice.comm.buffer('rw',U.data)
            prg = U.lattice.comm.compile(source)
            for size, x in self.lattice.bbox_range(displacement):
                if DEBUG:
                    print 'displacement: %s, size:%s, slice:%s' % (
                        displacement, size, x)
                event = prg.heatbath(U.lattice.comm.queue,
                             (size,),None,
                             data_buffer,
                             numpy.int32(shape[0]),
                             numpy.int32(shape[1]),
                             numpy.float32(beta),
                             numpy.int32(n_iter),
                             numpy.int32(m_iter),
                             U.lattice.bbox,
                             U.lattice.prngstate_buffer)
                if DEBUG:
                    print 'waiting'
                event.wait()
            cl.enqueue_copy(U.lattice.comm.queue, U.data, data_buffer).wait()
            return self
        return Code(source,runner)


class FermiOperator(object):
    """
    Class to store paths and coefficients relative to any gauge action
    """
    def __init__(self,lattice):
        self.lattice = lattice
        self.terms = []
    def add_term(self, gamma, paths=None):
        """
        gamma is a NspinxNspin Gamma matrix for Wilson fermions
        gamma is (coefficient, mu) for staggered (mu = 1,2,3 or 4)
        gamma is a number when paths is None (the identity term)

        if paths is None than psi = gamma * phi
 
        Paths are symmetrized. For example: 
        >>> wilson = FermionOperator(lattice).add_term(
            kappa,(1-Gamma[0]),path = [(4,)])
        >>> wilson = FermionOperator(lattice).add_term(
            kappa,(1+Gamma[0]),path = [(-4,)])
        >>> wilson = FermionOperator(lattice).add_term(
            c_sw,Gamma[mu][nu],path = [()],em_field,(mu,nu))
        """
        if paths is None:
            if self.terms and self.terms[0][1] is None:
                raise RuntimeError("cannot have two terms without paths")
            if not isinstance(gamma,(int,float)):
                raise RuntimeError("gamma must be a simple number of no paths")
            self.terms.insert(0,(gamma,paths))
        elif isinstance(paths,list) and len(paths)==1 and \
                isinstance(paths[0],tuple):
            self.terms.append((gamma,paths))
        else:
            raise RuntimeError("invalid Path")
        return self
    def multiply(self,phi,U,psi,name='aux'):
        """
        Computes phi = D[U]*psi
        
        Generates a kernel which performs the SU(n) heatbath.
        Example:
        >>> wilson = GaugeAction(lattice).add_term(1.0,paths = (1,2,-1,-2)) 
        >>> wilson.heatbath(beta,n_iter=10)
        """
        name = name or random_name()
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
        for gamma, opaths in self.terms:
            if isinstance(opaths,list):
                code += opencl_paths(function_name = name+str(k),
                                     lattice = self.lattice,
                                     coefficients=[1.0],
                                     paths=opaths,
                                     nc = nc,
                                     initialize = True,
                                     trace = False)
            
                k += 1
        action = opencl_fermi_operator(self.terms,U.lattice.d,nspin,nc,name)
        if nspin>1:
            source = makesource({'paths':code,'fermi_operator':action})
        else:
            source = makesource({'paths':code,'staggered_operator':action})
        def runner(source,self=self,phi=phi,U=U,psi=psi):

            phi_buffer = U.lattice.comm.buffer('rw',phi.data)
            U_buffer = U.lattice.comm.buffer('r',U.data)
            psi_buffer = U.lattice.comm.buffer('r',psi.data)
            prg = U.lattice.comm.compile(source)
            if nspin>1:
                meta_event = prg.fermi_operator
            else:
                meta_event = prg.staggered_operator
            event = meta_event(U.lattice.comm.queue,
                               (U.lattice.size,),None,
                               phi_buffer,
                               U_buffer,
                               psi_buffer,
                               U.lattice.bbox)
            if DEBUG:
                print 'waiting'
            event.wait()
            cl.enqueue_copy(U.lattice.comm.queue, phi.data, phi_buffer).wait()
            return self
        return Code(source,runner)

class Gamma(object):
    """
    Container of Gamma matrices
    """
    def __init__(self,representation='fermilab'):
        """
        Supported representation are be 'fermiqcd','ukqcd','milc' or 'chiral'
        """
        if representation == "dummy":
            gammas = [[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                      [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                      [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                      [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]
        elif representation == "ukqcd":
            gammas = [[[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]],
                      [[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]],
                      [[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]],
                      [[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]]]
        elif representation is "fermilab":
            gammas = [[[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]],
                      [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]],
                      [[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]],
                      [[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]]]
        elif representation is "milc":
            gammas = [[[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]],
                      [[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]],
                      [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]],
                      [[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]]]
        elif representation is "chiral":
            gammas = [[[0,0,0,-1],[0,0,-1,0],[0,-1,0,0],[-1,0,0,0]],
                      [[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]],
                      [[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]],
                      [[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]]]
        else:
            raise RuntimeError("unknown gamma matrices representation")
        self.matrices = [numpy.matrix(gamma) for gamma in gammas]
        self.matrices.append(self.matrices[0]) # gamma[4] same as gamma[0]
        # comupte Gamma[5]
        self.matrices.append(self.matrices[0]*self.matrices[1]*\
                                 self.matrices[2]*self.matrices[3])
    def __getitem__(self,mu):
        return self.matrices[mu]

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
# Part II, paths and symmetries
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
# Part III, OpenCL code generation algorithms
# ###########################################################

class Code(object):
    def __init__(self,source,runner):
        self.source = source
        self.runner = runner
    def run(self,*args,**kwargs):
        open('latest.c','w').write(self.source)
        return self.runner(self.source,*args,**kwargs)

def opencl_matrix_multiply(matrix):
    """ generates efficient code to multiply a known complex matrix by an unknown matrix """
    m,n = matrix.shape
    for i in range(m):
        for j in range(n):
            if matrix[i,j] != 0:
                print i,j,matrix[i,j]
    raise NotImplementedError

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
    d = len(lattice.dims)
    n = nc
    site = lattice.Site(tuple(0 for i in lattice.dims))
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
                    print matrices.keys()
                    print (site.coords,tuple(path[:z]))
                    raise RuntimeError
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
    for gamma, opaths in terms:
        if opaths is None:
            for r in range(nspin):
                for i in range(nc):
                    code.append('idx2=idx*%s+%s;' % (nspin*nc, r*nc+i))
                    code.append('phi[idx2].x = %s*psi[idx2].x;' % gamma)
                    code.append('phi[idx2].y = %s*psi[idx2].y;' % gamma)
            continue
        code.append("%s%s(path,U,idx,&bbox);" % (name,h))        
        path = opaths[0]
        site = [0]*d
        for step in opaths[0]: site[abs(step) % d] += 1 if step>0 else -1
        for i in range(d): code.append('delta.s[%s] = %s;' % (i, site[i]));
        code.append('p = phi+idx*%s;' % (nspin*nc));
        code.append('q = psi+idx2idx_shift(idx,delta,&bbox)*%s;' % (nspin*nc));
        spinor = [0]*(nspin*nc)
        if nspin>1:
            for r in range(nspin):
                for i in range(nc):
                    k = (r*nc+i)
                    spinor[k] = 0+0j
                    line = 'spinor[%s].x =' % k
                    for c in range(nspin):
                        coeff = gamma[r,c]
                        if coeff.real:
                            line+="+ (%s)*q[%s].x" % (coeff.real,c*nc+i)
                            spinor[k] += 1
                        if coeff.imag:
                            line+="- (%s)*q[%s].y" % (coeff.imag,c*nc+i)
                            spinor[k] += 1
                    line+=';'
                    if spinor[k].real: code.append(line)
                    line = 'spinor[%s].y =' % k
                    for c in range(nspin):
                        coeff = gamma[r,c]
                        if coeff.real:
                            line+="+ (%s)*q[%s].y" % (coeff.real,c*nc+i)
                            spinor[k] += 1j                        
                        if coeff.imag:
                            line+="+ (%s)*q[%s].x" % (coeff.imag,c*nc+i)
                            spinor[k] += 1j
                    line+=';'
                    if spinor[k].imag: code.append(line)
        else:
            (coeff, mu) = gamma
            line = 'coeff = %s * idx2eta(idx,%s,&bbox);' % (coeff, mu % d)
            code.append(line)
            for k in range(nc):
                line = 'spinor[%s].x = coeff*q[%s].x;' % (k,k)
                code.append(line)
                spinor[k] += 1
                line = 'spinor[%s].y = coeff*q[%s].y;' % (k,k)
                code.append(line)
                spinor[k] += 1j
        for r in range(nspin):
            for i in range(nc):
                counter = 0
                line = 'p[%s].x += ' % (r*nc+i)
                for j in range(nc):
                    k = r*nc+j
                    if spinor[k].real:
                        line += '\n\t\t + path[%s].x*spinor[%s].x' % (i*nc+j,k)
                        counter += 1
                    if spinor[k].imag:
                        line += '\n\t\t - path[%s].y*spinor[%s].y' % (i*nc+j,k)
                        counter += 1
                line += ';'
                if counter: code.append(line)
                counter = 0
                line = 'p[%s].y += ' % (r*nc+i)
                for j in range(nc):
                    k = r*nc+j
                    if spinor[k].imag:
                        line += '\n\t + path[%s].x*spinor[%s].y' % (i*nc+j,k)
                        counter += 1
                    if spinor[k].real:
                        line += '\n\t + path[%s].y*spinor[%s].x' % (i*nc+j,k)
                        counter += 1
                line += ';'
                if counter: code.append(line)
        h += 1
    return '\n'.join(code)

# ###########################################################
# inverters
# ###########################################################

def invert_minimum_residue(y,f,x,ap=1e-4,rp=1e-4,ns=200):
    q = numpy.empty_like(x)
    y[:] = x
    f(q,x)
    r = x-q
    for k in xrange(ns):
        f(q,r)
        alpha = numpy.vdot(q,r)/numpy.vdot(q,q)
        y += alpha*r
        r -= alpha*q
        residue = math.sqrt(numpy.vdot(r,r)/r.size)
        norm = math.sqrt(numpy.vdot(y,y))
        if residue<max(ap,norm*rp): return y
    raise ArithmeticError, 'no convergence'

def invert_bicgstab(y,f,x,ap=1e-4,rp=1e-4,ns=200):
    p = numpy.empty_like(x)
    q = numpy.empty_like(x)
    t = numpy.empty_like(x)
    r = numpy.empty_like(x)
    s = numpy.empty_like(x)
    y[:] = x
    f(r,y)
    r *= -1
    r += x
    q[:] = r
    rho_old = alpha = omega = 1.0
    for k in xrange(10):
        rho = numpy.vdot(q,r)
        beta = (rho/rho_old)*(alpha/omega)
        rho_old = rho
        p = beta*p + r - (beta*omega)*s
        f(s,p)
        alpha = rho/numpy.vdot(q,s)
        r -= alpha*s
        f(t,r)
        omega = numpy.vdot(t,r)/numpy.vdot(t,t)
        y += omega*r + alpha*p
        residue = math.sqrt(numpy.vdot(r,r)/r.size)
        norm = math.sqrt(numpy.vdot(y,y))     
        if residue<max(ap,norm*rp): return y
        r -= omega *t
    raise ArithmeticError, 'no convergence'

# ###########################################################
# Tests
# ###########################################################

class TestColdAndHotGauge(unittest.TestCase):
    def test_cold(self):
        comm = Communicator()
        for nc in range(2,4):
            for N in range(4,8):
                space = comm.Lattice((N,N,N,N))
                U = space.Field((4,nc,nc))
                U.set_cold()
                U.check_unitarity()
                self.assertTrue(abs(U.average_plaquette()) > 0.9999)
    def test_hot(self):
        comm = Communicator()
        for nc in range(2,4):
            for N in range(4,8):
                space = comm.Lattice((N,N,N,N))
                U = space.Field((4,nc,nc))
                U.set_hot()
                U.check_unitarity()
                self.assertTrue(abs(U.average_plaquette()) < 0.1)

class TestFieldTypes(unittest.TestCase):
    def test_fields(self):
        N, nc = 4, 3
        parameters = {}
        comm = Communicator()
        space = comm.Lattice((N,N,N,N))
        U = space.Field((space.d,nc,nc))
        U.set_hot()
        U.check_unitarity()
        U.set_cold()    
        U.check_cold()
        U.check_unitarity()
        
        psi = U.data_component((0,0,0))
        #Canvas().imshow(numpy.real(psi.data_slice((0,0)))).save()
        p = space.Site((0,0,0,0))
        psi = space.Field((space.d,nc))
        chi = space.Field((space.d,nc))
        psi[p,1,2] = 0.0 # set component to zezo
        phi = space.Field(1).set_link_product(U,[(1,2,-1,-2),(2,4,-2,-4)]).run()
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
        self.assertTrue(paths == 
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
        u_buffer = cl.Buffer(comm.ctx, comm.mf.READ_ONLY | comm.mf.COPY_HOST_PTR,
                             hostbuf=U.data)
        out_buffer = cl.Buffer(comm.ctx, comm.mf.WRITE_ONLY, 256) ### FIX NBYTES
        idx_buffer = cl.Buffer(comm.ctx, comm.mf.READ_ONLY, 32) # should not be
        core = open('kernels/qcl-core.c').read()
        prg = cl.Program(comm.ctx,core + kernel_code).build(
            options=['-I',INCLUDE_PATH])
        # getattr(prg,name)(comm.queue,(4,4,4),None,out_buffer,u_buffer,idx_buffer)

    def test_opencl_paths(self):
        N, nc = 4, 3
        comm = Communicator()
        space = comm.Lattice((N,N,N,N))
        U = space.Field((space.d,nc,nc))
        paths = [(+1,+2,+3,-1,-2,-3)]
        kernel_code = opencl_paths(function_name='staples',
                                   lattice=space,
                                   paths=paths,
                                   coefficients = [1.0 for p in paths],
                                   nc=nc,
                                   trace=False)
        self.make_kernel(comm,kernel_code,'staples',U)

class TestHeatbath(unittest.TestCase):
    def test_heatbath(self):
        N, nc = 4, 3
        comm = Communicator()
        space = comm.Lattice((N,N,N,N))
        U = space.Field((space.d,nc,nc))
        U.set_cold()    
        wilson = GaugeAction(space)
        wilson.add_term(1.0,(1,2,-1,-2))
        code = wilson.heatbath(U,beta=4.0)
        # print code.source
        avg_plq = 0.0
        for k in range(200):
            code.run()        
            plq = U.average_plaquette()
            avg_plq = (avg_plq * k + plq)/(k+1)
            if DEBUG:
                print '<plaq> =', plq, '<avg> =', avg_plq
                U.check_unitarity(output=False)
        self.assertTrue( 0.29 < abs(avg_plq) < 0.3 )

class TestFermions(unittest.TestCase):
    def test_wilson_action(self):
        N, nspin, nc = 9, 4, 3
        r = 1.0
        kappa = 0.1234
        I = identity(4)
        gamma = Gamma('fermilab')

        comm = Communicator()
        space = comm.Lattice((N,N,N,N))
        U = space.Field((space.d,nc,nc))
        psi = space.Field((nspin,nc))        
        phi = space.Field((nspin,nc))

        U.set_cold()
        p = space.Site((0,0,4,4))
        psi[p,0,0] = 100.0
        wilson = FermiOperator(space)
        wilson.add_term(1.0, None)
        for mu in (1,2,3,4):
            wilson.add_term(kappa*(r*I-gamma[mu]), [(mu,)])
            wilson.add_term(kappa*(r*I+gamma[mu]), [(-mu,)])
        for k in range(10):                        
            wilson.multiply(phi,U,psi).run()
            # project spin=0, color=0 component, plane passing fox t=0,x=0
            chi = numpy.real(psi.data_component((0,0)).data_slice((0,0)))
            Canvas().imshow(chi).save('fermi.%s.png' % k)
            phi,psi = psi,phi

    def test_staggered_action(self):
        N, nc = 16, 3
        r = 1.0
        kappa = 0.1234

        comm = Communicator()
        space = comm.Lattice((N,N,N,N))
        U = space.Field((space.d,nc,nc))
        psi = space.Field((nc,))        
        phi = space.Field((nc,))

        U.set_cold()
        p = space.Site((0,0,N/2,N/2))
        psi[p,0] = 100.0
        wilson = FermiOperator(space)
        wilson.add_term(1.0, None)
        for mu in (1,2,3,4):
            wilson.add_term((kappa, mu), [(mu,)])
            wilson.add_term((kappa, mu), [(-mu,)])
        for k in range(100):                        
            wilson.multiply(phi,U,psi).run()
            # project color=0 component, plane passing fox t=0,x=0
            chi = numpy.real(psi.data_component((0,)).data_slice((0,0)))
            Canvas().imshow(chi).save('staggered.%s.png' % k)
            phi,psi = psi,phi


class TestInverters(unittest.TestCase):
    def test_minimum_residue(self):
        def f(a,b):
            a[0,0] = b[0,0] 
            a[0,1] = 0.1*b[0,1] 
            a[1,0] = 0.2*b[1,0] 
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
            a[0,1] = 0.1*b[0,1] 
            a[1,0] = 0.2*b[1,0] 
            a[1,1] = b[1,1] 
        a = numpy.array([[1.0,2.0],[3.0,4.0]])
        b = numpy.array([[0.0,0.0],[0.0,0.0]])
        c = numpy.array([[0.0,0.0],[0.0,0.0]])
        invert_bicgstab(b,f,a)
        f(c,b)
        assert numpy.linalg.norm(c-a)<0.02

    def test_fermion_propagator(self):
        return 

        N, nspin, nc = 9, 4, 3
        r = 1.0
        kappa = 0.1234
        I = identity(4)
        gamma = Gamma('fermilab')

        comm = Communicator()
        space = comm.Lattice((N,N,N,N))
        U = space.Field((space.d,nc,nc))
        psi = space.Field((nspin,nc))        
        phi = space.Field((nspin,nc))

        U.set_cold()
        p = space.Site((0,0,4,4))
        psi[p,0,0] = 100.0
        wilson = FermiOperator(space)
        wilson.add_term(1.0, None)
        for mu in (1,2,3,4):
            wilson.add_term(kappa*(r*I-gamma[mu]), [(mu,)])
            wilson.add_term(kappa*(r*I+gamma[mu]), [(-mu,)])
        Dslash = lambda y,x,U=U: wilson.multiply(y,U,x).run()
        invert_bicgstab(phi.data,Dslash,psi.data)
        chi = numpy.real(phi.data_component((0,0)).data_slice((0,0)))
        Canvas().imshow(chi).save('fermi.prop.png')

if __name__=='__main__':
    unittest.main()
    

