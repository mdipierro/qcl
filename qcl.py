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
    chars = 'abcdefghijklmonopqrstuvwxyz'
    return ''.join(random.choice(chars) for i in range(5))

def identity(n):
    """ returns an identity matrix nxn """
    return numpy.eye(n)

def hermitian(U):
    """ returns the hermitian of the U matrix """
    return numpy.transpose(U).conj()

def is_unitary(U,precision=1e-4):
    """ checks if U is unitary within precision """
    return numpy.all(abs(U*hermitian(U)-identity(U.shape[0]))<precision)

def product(a):
    """ auxiliary function computes product of a items """
    return 1 if not a else reduce(operator.mul,a)

# ###########################################################
# Part I, lattices and fields
# ###########################################################

def listify(items):
    """ turns a tuple or single integer into a list """
    if isinstance(items,(int,long)):
        return [items]
    elif isinstance(items,tuple):
        return [item for item in items]
    elif isinstance(items,list):
        return items
    else:
        raise ValueError, "expected a list"

def makesource(vars=None,filename='kernels/qcl-core.c'):
    """ loads and compiles a kernel """
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
    abstracts MPI as well as communications to OpenCL devices
    it assumes one process per opencl device and creates the
    ctx and queue for that device

    currently only supports one device.
    """
    def __init__(self):
        if not cl:
            raise RuntimeError, 'pyOpenCL is not available'
        self.platforms = cl.get_platforms()
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        self.rank = 0
        self.nodes = 1
    def buffer(self,t,hostbuf):
        """ make a new opencl buffer """
        s = {'r': self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
             'w': self.mf.WRITE_ONLY | self.mf.COPY_HOST_PTR,
             'rw': self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
             'wr': self.mf.READ_WRITE | self.mf.COPY_HOST_PTR}[t]
        return cl.Buffer(self.ctx,s,hostbuf=hostbuf)
    def compile(self,source):
        """ compiles a kernel """
        return cl.Program(self.ctx,source).build(options=['-I',INCLUDE_PATH])
    def add(self,value):
        """ mpi add """
        if not IGNORE_NOT_IMPLEMENTED: raise NotImplementedError
    def send(self,dest,data):
        """ mpi send """
        if not IGNORE_NOT_IMPLEMENTED: raise NotImplementedError
    def recv(self,source,data):
        """ mpi recv """
        if not IGNORE_NOT_IMPLEMENTED: raise NotImplementedError
    def Lattice(self,dims,bboxs=None):
        """ create a new lattice that lives on the device """
        return Lattice(self,dims,bboxs)


class Lattice(object):
    """ a lattice encodes info about volume and parallelization """
    def __init__(self,comm,dims,bboxs=None):
        """
        constructor of a Lattice
        it computes the size and allocaltes local ranlux prng on device
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
        """ initialize the parallel random number generator """
        self.seed = seed
        STATE_SIZE = 112
        self.prngstate = numpy.zeros(self.size*STATE_SIZE,dtype=numpy.float32)
        prg = self.comm.compile(makesource())
        self.prngstate_buffer = self.comm.buffer('rw',self.prngstate)
        prg.init_ranlux(self.comm.queue,(self.size,),None,
                        numpy.uint32(seed),self.prngstate_buffer)
    def prng_get(self):
        """ retrieves the state of the prng """
        cl.enqueue_copy(self.comm.queue, self.prngstate, self.prngstate_buffer).wait()
    def prng_off(self):
        """ disabled the prng """
        self.prngstate = self.prngstate_buffer = None
    def coords2global(self,coords):
        """ converts (t,x,y,z) coordinates into global index """
        if len(coords)!=len(self.dims):
            raise RuntimeError, "invalid conversion"
        return sum(p*product(self.dims[i+1:]) \
                       for i,p in enumerate(coords))
    def global2coords(self,i):
        """ converts global index into (t,x,y,z) """
        coords = []
        for k in reversed(self.dims):
            i,reminder = divmod(i,k)
            coords.insert(0,reminder)
        return tuple(coords)
    def check_coords(self):
        for i in range(self.size):
            assert self.coords2global(self.global2coords(i)) == i
    def Site(self,coords):
        """ returns a Site constructor for this lattice """
        return Site(self,coords)
    def Field(self,siteshape,dtype=numpy.complex64):
        """ returns a Field constructor for this """
        return Field(self,siteshape,dtype)


class Site(object):
    """ a site object stores information about a lattice site """
    def __init__(self,lattice,coords):
        self.lattice = lattice
        self.coords = tuple(coords)
        self.gl_idx = self.lattice.coords2global(coords)
        self.lo_idx = self.gl_idx ### until something better
    def __add__(self,mu):
        if mu<0:
            return self.__sub__(-mu)
        mu = mu % self.lattice.d
        L = self.lattice.dims[mu]
        coords = tuple((c if nu!=mu else ((c+1)%L))
                       for nu,c in enumerate(self.coords))
        return self.lattice.Site(coords)
    def __sub__(self,mu):
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
    def __init__(self,op,left,right=None):
        self.op = op
        self.left = left
        self.right = right

class Field(object):
    """
    a field can be used to store a guage field, a fermion field,
    a staggered field, etc
    no math operations between fields here except for O(n) operations,
    the others MUST be done in OpenCL only!
    """
    def __init__(self, lattice, siteshape, dtype=numpy.complex64):
        self.lattice = lattice
        self.siteshape = listify(siteshape)
        self.size = product(self.siteshape)
        self.sitesize = product(self.siteshape)
        self.dtype = dtype
        self.data = numpy.zeros([self.lattice.size]+self.siteshape,dtype=dtype)
        if DEBUG: 
            print 'allocating %sbytes' % int(self.lattice.size*self.sitesize*8)
    def copy(self,other):
        """
        a.copy(b) # makes a copy of field b into field a.
        """
        if not (self.lattice == self.lattice and
                self.siteshape==other.siteshape and
                self.sitesize == other.sitesize):
            raise TypeError, "Cannot copy incompatible fields"
        self.dtype = other.dtype
        self.data[:] = other.data
    def data_component(self,component):
        newfield = self.lattice.Field((1,))
        for i in xrange(self.lattice.size):
            newfield.data[i] = self.data[i][component]
        return newfield
    def data_slice(self,slice_coords):
        """
        returns a numpy slice of the current self.data at coordinates c
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
            self.data -= other
        return self
    def __rmul__(self,other):
        """ maps a += c*b into a.add_scaled(c,b) for a,b arrays """
        return Op('*',other,self)
    def add_scaled(self,scale,other,n=1000000):
        """ a.add_scaled(c,b) is the same as a[:]=c*b[:] """
        if self.data.shape != other.data.shape: raise RuntimeError
        size = product(self.data.shape)
        for i in xrange(0,size,n):
            self.data.flat[i:i+n] += scale*other.data.flat[i:i+n]
        return self
    def __mul__(self,other):
        """
        computes scalar product of two Fields
        it first re-shape them into 1D arrays, then computes product
        Not designed to work in parallel (yet)
        """
        if self.lattice.parallel: raise NotImplementedError
        if self.data.shape != other.data.shape: raise RuntimeError
        f = numpy.vdot if self.dtype in COMPLEX_TYPES else numpy.dot
        return f(self.data,other.data)
    def __getitem__(self,args):
        """ do not use these to implement algorithms - too slow """
        site, args = args[0], args[1:]
        return self.data[(site.lo_idx,)+args]
    def __setitem__(self,args,value):
        """ do not use these to implement algorithms - too slow """
        site, args = args[0], args[1:]
        self.data[(site.lo_idx,)+args] = value
    def sum(self,*a,**b):
        """ a.sum() computes the sum of terms in a """
        return numpy.sum(self.data,*a,**b)
    def load(self,filename):
        """ loads the field """
        format = filename.split('.')[-1].lower()
        if format == 'npy':
            self.data = None
            self.data = numpy.load(filename)
        else:
            raise NotImplementedError
        return self
    def save(self,filename):
        """ saves the field """
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
        """ uses commuicator to sync devices """
        if not IGNORE_NOT_IMPLEMENTED: raise NotImplementedError
        return self

    def set_link_product(self,U,paths,trace=True,name='aux'):
        name = name or random_name()
        code = opencl_paths(function_name = name,
                            lattice = self.lattice,
                            coefficients = [1.0 for p in paths],
                            paths=paths,                            
                            sun = U.data.shape[-1],
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
                     numpy.int32(shape[0]), # 1 for trace, sun for no-trace
                     self.lattice.bbox).wait()
            cl.enqueue_copy(self.lattice.comm.queue, self.data, out_buffer).wait()
            return self        
        return Code(source,runner)

    def set_cold(self):
        """ uses a kernel to set all links to cold """
        shape = self.siteshape
        if not (len(shape)==3 and shape[1]==shape[2]): raise RuntimeError
        data_buffer = self.lattice.comm.buffer('w',self.data)
        prg = self.lattice.comm.compile(makesource())
        prg.set_cold(self.lattice.comm.queue,(self.lattice.size,),None,
                     data_buffer,numpy.int32(shape[0]),numpy.int32(shape[1]),
                     self.lattice.bbox).wait()
        cl.enqueue_copy(self.lattice.comm.queue, self.data, data_buffer).wait()

    def set_custom(self):
        """ uses a kernel to set all links to cold """
        shape = self.siteshape
        if not (len(shape)==3 and shape[1]==shape[2]): raise RuntimeError
        data_buffer = self.lattice.comm.buffer('w',self.data)
        prg = self.lattice.comm.compile(makesource())
        prg.set_custom(self.lattice.comm.queue,(self.lattice.size,),None,
                       data_buffer,numpy.int32(shape[0]),numpy.int32(shape[1]),
                       self.lattice.bbox).wait()
        cl.enqueue_copy(self.lattice.comm.queue, self.data, data_buffer).wait()

    def check_cold(self):
        for idx in xrange(self.lattice.size):
            for mu in xrange(self.siteshape[0]):
                for i in xrange(self.siteshape[1]):
                    for j in xrange(self.siteshape[2]):
                        assert(self.data[idx,mu,i,j]==(1 if i==j else 0))
    def set_hot(self):
        """ uses a kernel to set all links to cold """
        shape = self.siteshape
        if not (len(shape)==3 and shape[1]==shape[2]): raise RuntimeError
        data_buffer = self.lattice.comm.buffer('w',self.data)
        prg = self.lattice.comm.compile(makesource())
        prg.set_hot(self.lattice.comm.queue,(self.lattice.size,),None,
                    data_buffer,numpy.int32(shape[0]),numpy.int32(shape[1]),
                    self.lattice.bbox,self.lattice.prngstate_buffer).wait()
        cl.enqueue_copy(self.lattice.comm.queue, self.data, data_buffer).wait()

    def check_unitarity(self,output=DEBUG):
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
            raise RuntimeError, "U is not unitary"

    def average_plaquette(self,shape=(1,2,-1,-2)):
        paths = bc_symmetrize(shape,d=self.lattice.d,positive_only=True)
        paths = remove_duplicates(paths,bidirectional=True)        
        # print 'average_plaquette.paths=',paths        
        code = self.lattice.Field(1).set_link_product(self,paths)
        phi=code.run()
        #for idx in xrange(self.lattice.size):
        #    print idx, phi.data[idx]
        return phi.sum()/(self.lattice.size*len(paths)*self.siteshape[-1])

class GaugeAction(object):
    def __init__(self,lattice):
        self.lattice = lattice
        self.terms = []
    def add_term(self,coefficient,paths):
        if isinstance(paths,tuple):
            paths = bc_symmetrize(paths,d=self.lattice.d)
            paths = remove_duplicates(paths,bidirectional=False)
        elif not (isinstance(paths,list) and isinstance(paths[0],tuple)):
            raise RuntimeError, "not a valid action term" 
        self.terms.append((coefficient,paths))
        return self
    def heatbath(self,U,beta,n_iter=1,name='aux'):
        """ uses a kernel to set all links to cold """
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
                                 sun = U.data.shape[-1],
                                 trace = (U.data.shape[-1]==1))
        action = opencl_heatbath_action(self.lattice.d,name)
        source = makesource({'paths':code,'heatbath_action':action})
        def runner(source,self=self,U=U,beta=beta,n_iter=n_iter,
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
                             U.lattice.bbox,
                             U.lattice.prngstate_buffer)
                if DEBUG:
                    print 'waiting'
                event.wait()
            cl.enqueue_copy(U.lattice.comm.queue, U.data, data_buffer).wait()
            return self
        return Code(source,runner)

class Gamma(object):
    def __init__(self,representation='fermilab'):
        """
        representation can be 'fermiqcd','ukqcd','milc' or 'chiral'
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
            raise RuntimeError, "unknown gamma matrices representation"
        self.matrices = [numpy.matrix(gamma) for gamma in gammas]
        self.matrices.append(None) # No Gamma(...)[4]
        # comupte Gamma[5]
        self.matrices.append(self.matrices[0]*self.matrices[1]*\
                                 self.matrices[2]*self.matrices[3])
    def __getitem__(self,mu):
        return self.matrices[mu]

class Lambda(object):
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



class Action(object):
    def __init__(self,U,parameters):
        pass

def BiCGStabInverter(phi,psi,action):
    pass

def test_gauge():
    Nc = 2
    for N in range(1,30):
        print N
        comm = Communicator()
        space = comm.Lattice((N,N,N,N))
        U = space.Field((4,Nc,Nc))
        print 'setting cold'
        U.set_cold()
        if N<8: U.check_unitarity()
        assert abs(U.average_plaquette()) > 0.9999
        print 'setting hot'
        U.set_hot()
        if N<8: U.check_unitarity()
        assert abs(U.average_plaquette()) < 0.5
        print 'done'

def test_gauge2():
    """ using test_custom to figure out lattice layout """
    N, Nc = 2, 2
    comm = Communicator()
    space = comm.Lattice((N,N,N,N))
    U = space.Field((space.d,Nc,Nc))
    U.set_custom()
    print U.average_plaquette()

def test_lattice_fields():
    N = int(sys.argv[1]) if len(sys.argv)>1 else 4
    parameters = {}
    comm = Communicator()
    space = comm.Lattice((N,N,N,N))

    U = space.Field((4,3,3))
    t0 = time.time()
    U.set_hot()
    psi = U.data_component((0,0,0))
    Canvas().imshow(numpy.real(psi.data_slice((0,0)))).save()

    print 'hot',time.time()-t0
    if N<8: U.check_unitarity()
    t0 = time.time()
    U.set_cold()    
    print 'cold',time.time()-t0
    if N<8: U.check_cold()
    if N<8: U.check_unitarity()
    
    p = space.Site((0,0,0,0))
    psi = space.Field((4,3))
    chi = space.Field((4,3))
    psi[p,1,2] = 0.0 # set component to zezo

    phi = space.Field(1).set_link_product(U,[(1,2,-1,-2),(2,4,-2,-4)]).run()
    assert phi.sum() == 4**4 * 3*2

    old = U.sum()
    U.save('test.npy')
    U.load('test.npy')
    assert U.sum() == old

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
            raise RuntimeError, "accidental double counting"
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
    computes the derivative of path respect to link mu.
    if bidirecional path, it also derives the reverse path
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
    computes the derivative of all paths respect to link mu.
    if bidirecional path, it also derives the reverse of each path
    """
    dpaths = []
    for path in paths:
        dpaths+=derive_path(path,mu,bidirectional)
    return dpaths

def test_paths():
    path = (+1,+2,-1,-2)
    paths = bc_symmetrize(path,d=4)
    print paths
    paths = remove_duplicates(paths,bidirectional=True)
    print paths
    staples = derive_paths(paths,+1,bidirectional=True)
    print staples


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
                 sun, # SU(n) gauge group
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
    RE = '%s_%ix%i.x'
    IM = '%s_%ix%i.y'
    ADD = ' += '
    PLUS = '+'
    MINUS = '-'
    TIMES = '*'
    EQUAL = ' = '
    NEWLINE = '\n'+' '*12
    vars = []
    code = []
    matrices = {}
    d = len(lattice.dims)
    n = sun
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
                                code.append('%s.x = %s.x;' %(var2,var))
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
                    for i in range(n):
                        for j in range(n):
                            var_re = RE % (name,i,j)
                            var_im = IM % (name,i,j)
                            k = 0
                            line = var_re +EQUAL
                            line += PLUS+RE % (name1,i,k);
                            line += TIMES+RE % (name2,k,j);
                            line += MINUS+IM % (name1,i,k);
                            line += TIMES+IM % (name2,k,j);
                            for k in range(1,n):
                                line += NEWLINE
                                line += PLUS+RE % (name1,i,k);
                                line += TIMES+RE % (name2,k,j);
                                line += MINUS+IM % (name1,i,k);
                                line += TIMES+IM % (name2,k,j);
                            code.append(line+';')
                            k = 0
                            line = var_im + EQUAL
                            line += PLUS+RE % (name1,i,k);
                            line += TIMES+IM % (name2,k,j);
                            line += PLUS+IM % (name1,i,k);
                            line += TIMES+RE % (name2,k,j);
                            for k in range(1,n):
                                line += NEWLINE
                                line += PLUS+RE % (name1,i,k);
                                line += TIMES+IM % (name2,k,j);
                                line += PLUS+IM % (name1,i,k);
                                line += TIMES+RE % (name2,k,j);
                            code.append(line+';')
                elif mu<0: # if link backwards ... same
                    code.append('\n// compute %s*%s^H -> %s' %(name1,name2,name))
                    for i in range(n):
                        for j in range(n):
                            var_re = RE % (name,i,j)
                            var_im = IM % (name,i,j)
                            k = 0
                            line = var_re + EQUAL
                            line += PLUS+RE % (name1,i,k);
                            line += TIMES+RE % (name2,j,k);
                            line += PLUS+IM % (name1,i,k);
                            line += TIMES+IM % (name2,j,k);
                            for k in range(1,n):
                                line += NEWLINE
                                line += PLUS+RE % (name1,i,k);
                                line += TIMES+RE % (name2,j,k);
                                line += PLUS+IM % (name1,i,k);
                                line += TIMES+IM % (name2,j,k);
                            code.append(line+';')
                            k = 0
                            line = var_im + EQUAL
                            line += MINUS+RE % (name1,i,k);
                            line += TIMES+IM % (name2,j,k);
                            line += PLUS+IM % (name1,i,k);
                            line += TIMES+RE % (name2,j,k);
                            for k in range(1,n):
                                line += NEWLINE
                                line += MINUS+RE % (name1,i,k);
                                line += TIMES+IM % (name2,j,k);
                                line += PLUS+IM % (name1,i,k);
                                line += TIMES+RE % (name2,j,k);
                            code.append(line+';')
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
            
    body = '\n'.join('    '+line.replace('\n','\n    ') for line in vars+code)
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

def opencl_heatbath_action(d,name):    
    return '\n'.join('if(mu==%i) %s%i(staples,U,idx,&bbox);' % (i,name,i) for i in range(d))

### rewrite below here

def test_kernel(comm,kernel_code,name,U):
    ### fix this, there need to be a parallel loop over sites
    u_buffer = cl.Buffer(comm.ctx, comm.mf.READ_ONLY | comm.mf.COPY_HOST_PTR,
                         hostbuf=U.data)
    out_buffer = cl.Buffer(comm.ctx, comm.mf.WRITE_ONLY, 256) ### FIX NBYTES
    idx_buffer = cl.Buffer(comm.ctx, comm.mf.READ_ONLY, 32) # should not be
    core = open('kernels/qcl-core.c').read()
    prg = cl.Program(comm.ctx,core + kernel_code).build(
        options=['-I',INCLUDE_PATH])
    # getattr(prg,name)(comm.queue,(4,4,4),None,out_buffer,u_buffer,idx_buffer)

def test_opencl_paths():
    sun = 2
    comm = Communicator()
    space = comm.Lattice((4,4,4,4))
    U = space.Field((4,sun,sun))
    #paths = bc_symmetrize((+1,+2,-1,-2))
    #paths = remove_duplicates(paths,bidirectional=True)
    #paths = derive_paths(paths,1,bidirectional=True)
    paths = [(+1,+2,+3,-1,-2,-3)]
    kernel_code = opencl_paths(function_name='staples',
                               lattice=space,
                               paths=paths,
                               coefficients = [1.0 for p in paths],
                               sun=sun,
                               trace=False)
    print kernel_code
    test_kernel(comm,kernel_code,'staples',U)

def test_something():
    comm = Communicator()
    space = comm.Lattice((4,4))
    U = space.Field((space.d,2,2))
    for k in range(1000):
        print k
        U.set_hot()
    

def test_heatbath():
    N = 10
    Nc = 2
    comm = Communicator()
    space = comm.Lattice((N,N))
    U = space.Field((space.d,Nc,Nc))
    U.set_hot()
    print '<plq> = ',U.average_plaquette()
    wilson = GaugeAction(space)
    wilson.add_term(1.0,(2,1,-2,-1))
    #wilson.add_term(1.0,(1,2,-1,-2))
    #wilson.add_term(1.0,(1,3,-1,-3))
    #wilson.add_term(1.0,(1,4,-1,-4))
    #wilson.add_term(1.0,(2,3,-2,-3))
    #wilson.add_term(1.0,(2,4,-2,-4))
    #wilson.add_term(1.0,(3,4,-3,-4))
    code = wilson.heatbath(U,beta=100.0)
    # print code.source
    avg_plq = 0.0
    for k in range(10000):
        code.run()        
        plq = U.average_plaquette()
        avg_plq = (avg_plq * k + plq)/(k+1)
        print '<plaq> =', plq, '<avg> =', avg_plq
        if N<8:
            U.check_unitarity(output=False)
        if len(space.dims) == 4 and N==12:
            psi = U.data_component((0,0,0))
            Canvas().imshow(numpy.real(psi.data_slice((0,0)))).save()
    print 'done!'



if __name__=='__main__':
    #test_gauge()
    #test_gauge2()
    #test_paths()
    #test_lattice_fields()
    #test_opencl_paths()
    test_heatbath()
    #test_something()
    pass



