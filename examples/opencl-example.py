import pyopencl as cl
import numpy
import numpy.linalg as la

for key in dir(cl):
    print key, eval('cl.'+key)

N = 100**4

a = numpy.random.rand(N).astype(numpy.float32)
b = numpy.random.rand(N).astype(numpy.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

prg = cl.Program(ctx, """
#include <pyopencl-ranluxcl.cl>
#include <pyopencl-complex.h>

    __kernel void sum(__global const cfloat_t *a,
    __global const cfloat_t *b, __global cfloat_t *c)
    {
      int gid = get_global_id(0);
      c[gid].x = a[gid].x + b[gid].x;
      c[gid].y = a[gid].y + b[gid].y;
    }
    """).build()

shape = (a.shape[0]/2,)
prg.sum(queue, shape, None, a_buf, b_buf, dest_buf)

a_plus_b = numpy.empty_like(a)
cl.enqueue_copy(queue, a_plus_b, dest_buf)

print la.norm(a_plus_b - (a+b))
