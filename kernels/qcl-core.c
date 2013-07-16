#include <pyopencl-complex.h>
#include <pyopencl-ranluxcl.cl>
#define PRINTF
#ifdef PRINTF
#pragma OPENCL EXTENSION cl_intel_printf : enable
// #pragma OPENCL EXTENSION cl_amd_printf : enable
#endif
#define Pi 3.141592653589793
#define MAXD 10
#define MAXN 32

void assert(bool value) {
  if(!value) exit(1);
}

struct bbox_t {
  int a[MAXD]; // global coordinate of content
  int b[MAXD]; // top border (CHECK THE USE OF THIS)
  int c[MAXD]; // content size
  int d[MAXD]; // outer size
  int e[MAXD]; // init looping position
  int f[MAXD]; // step looping position
};

struct shift_t {
  int s[MAXD];
};

inline struct shift_t gid2shift(const size_t gid, const struct bbox_t *bbox) {
  struct shift_t shift;
  int dd;
  size_t i = gid;
  size_t j;
  for(int k=MAXD-1; k>=0; k--) {
    dd = (*bbox).c[k]/(*bbox).f[k];
    if((*bbox).d[k]>0 && dd>0) {
      j = i % dd;
      shift.s[k] = j*(*bbox).f[k]+(*bbox).e[k];
      i = (i - j) / dd;
    } else {
      shift.s[k] = (*bbox).e[k]; // unused coordinate (set to zero)
    }
  }
  assert(i==0);
  return shift;
}

inline struct shift_t idx2shift(const size_t idx, const struct bbox_t *bbox) {
  struct shift_t shift;
  int dd;
  size_t i=idx;
  for(int k=MAXD-1; k>=0; k--) {
    if((*bbox).d[k]==0)
      shift.s[k]=0;
    else {
      dd = (*bbox).d[k];
      shift.s[k] = i % dd - (*bbox).b[k];
      i = i / dd;
    }
  }
  assert(i==0);
  return shift;
}

inline double idx2eta(const size_t idx, const int mu,
                      const struct bbox_t *bbox) {
  int dd, tmp=0;
  size_t i=idx;
  for(int k=MAXD-1; k>=0; k--) {
    if((*bbox).d[k]>0) {
      dd = (*bbox).d[k];
      if(k<mu) tmp += i % dd - (*bbox).b[k];
      i = i / dd;
    }
  }
  assert(i==0);
  return (tmp % 2==0)?(+1.0):(-1.0);
}

inline struct shift_t shiftdelta(const struct shift_t shift,
                                 const struct shift_t delta) {
  struct shift_t t;
  for(int k=0; k<MAXD; k++)
    t.s[k] = shift.s[k]+delta.s[k];
  return t;
}

inline size_t shift2idx(const struct shift_t shift,
                        const struct bbox_t *bbox) {
  size_t idx=0;
  int d2;
  for(int k=0; k<MAXD; k++) {
    d2 = (*bbox).d[k];
    if(d2>0) {
      idx *= d2;
      idx += (shift.s[k] + (*bbox).b[k] + d2) % d2;
    }
  }
  return idx;
}

inline size_t gid2idx(const size_t gid,
                      const struct bbox_t *bbox) {
  size_t idx = shift2idx(gid2shift(gid,bbox),bbox);
  return idx;
}

inline size_t gid2idx_shift(const size_t gid,
                            struct shift_t delta,
                            const struct bbox_t *bbox) {
  return shift2idx(shiftdelta(gid2shift(gid,bbox),delta),bbox);
}

inline size_t idx2idx_shift(const size_t idx,
                            struct shift_t delta,
                            const struct bbox_t *bbox) {
  return shift2idx(shiftdelta(idx2shift(idx,bbox),delta),bbox);
}

float4 uniform4(ranluxcl_state_t* prst) {
  // this generates 4 uniform random numbers (x,y,z,w)
  return ranluxcl32(prst);
}

float4 gaussian4(ranluxcl_state_t* prst) {
  return ranluxcl32norm(prst);
}

void SU(global cfloat_t *link, int n, ranluxcl_state_t *prst, int gid) {

  cfloat_t a00,a01,a10,a11;
  cfloat_t b00,b10;
  int i,j,k;
  float alpha, sin_alpha;
  float phi, cos_theta, sin_theta;
  float4 v;
  if(n==1) {
    alpha = 2.0*Pi*uniform4(prst).x;
    link[0].x = cos(alpha);
    link[0].y = sin(alpha);
  } else {
    // set tmp to a the identity matrix
    for(i=0; i<n; i++)
      for(j=0; j<n; j++) {
        link[i*n+j].x = (i==j)?1:0;
        link[i*n+j].y = 0;
      }
    for(i=0; i<n-1; i++)
      for(j=i+1; j<n; j++) {
        v = uniform4(prst);
        alpha=Pi*v.x;
        phi=2.0*Pi*v.y;
        cos_theta=2.0*v.z-1;
        sin_theta=sqrt(1.0-cos_theta*cos_theta);
        sin_alpha=sin(alpha);
        a00.x = a11.x = cos(alpha);
        a01.y = a10.y = sin_alpha*sin_theta*cos(phi);
        a01.x = sin_alpha*sin_theta*sin(phi);
        a10.x = -a01.x;
        a00.y = sin_alpha*cos_theta;
        a11.y = -a00.y;
        for(k=0;k<n;k++) {
          b00 = link[i*n+k];
          b10 = link[j*n+k];
          link[i*n+k] = cfloat_add(cfloat_mul(a00,b00),cfloat_mul(a01,b10));
          link[j*n+k] = cfloat_add(cfloat_mul(a10,b00),cfloat_mul(a11,b10));
        }
      }
  }
}

kernel void init_ranlux(private uint ins,
                        global ranluxcl_state_t *ranluxcltab) {
  ranluxcl_initialization(ins, ranluxcltab);
}

kernel void set_cold(global cfloat_t *U,
                     int d,
                     int n,
                     struct bbox_t bbox) {
  size_t gid = get_global_id(0);
  size_t idx = gid2idx(gid,&bbox);
  for(int mu=0; mu<d; mu++)
    for(int i=0; i<n; i++)
      for(int j=0; j<n; j++) {
        U[(idx*d+mu)*n*n+i*n+j].x = (i==j)?1:0;
        U[(idx*d+mu)*n*n+i*n+j].y = 0;
      }
}

kernel void set_custom(global cfloat_t *U,
                     int d,
                     int n,
                     struct bbox_t bbox) {
  size_t gid = get_global_id(0);
  size_t idx = gid2idx(gid,&bbox);
  for(int mu=0; mu<d; mu++)
    for(int i=0; i<n; i++)
      for(int j=0; j<n; j++) {
        U[(idx*d+mu)*n*n+i*n+j].x = (i==j)?(idx+1):0;
        U[(idx*d+mu)*n*n+i*n+j].y = (i==j)?0:0;
      }
}

kernel void set_hot(global cfloat_t *U,
                    int d,
                    int n,
                    struct bbox_t bbox,
                    global ranluxcl_state_t *ranluxcltab) {
  size_t gid = get_global_id(0);
  size_t idx = gid2idx(gid,&bbox);
  ranluxcl_state_t rst;
  ranluxcl_download_seed(&rst,ranluxcltab);
  for(int mu=0; mu<d; mu++)
    SU(U+(idx*d+mu)*n*n,n,&rst,gid);
  ranluxcl_upload_seed(&rst,ranluxcltab);
}


void heatbath_SU2(cfloat_t *a,
                  float beta_eff,
                  ranluxcl_state_t *prst) {
  float4 e,r,b;
  float dk, p0;
  cfloat_t u0,u1,u2,u3;
  cfloat_t v0,v1,v2,v3;
  float delta, phi, sin_alpha, sin_theta, cos_theta;
  e.x=a[0].x+a[3].x;
  e.y=a[1].y+a[2].y;
  e.z=a[1].x-a[2].x;
  e.w=a[0].y-a[3].y;
  dk=sqrt(e.x*e.x+e.y*e.y+e.z*e.z+e.w*e.w);
  p0=(dk*beta_eff);
  u0=(cfloat_t)(e.x/dk,-e.w/dk);
  u1=(cfloat_t)(-e.z/dk,-e.y/dk);
  u2=(cfloat_t)(e.z/dk,-e.y/dk);
  u3=(cfloat_t)(e.x/dk,e.w/dk);
  do {
    do {
      r = uniform4(prst);
    } while (r.x<0.0001 || r.y<0.0001);
    r.x = -log(r.x)/p0;
    r.y = -log(r.y)/p0;
    r.z = cos(2.0*Pi*r.z);
    r.z = r.z*r.z;
    delta = r.y+r.x*r.z;
  } while (r.w*r.w > (1.0-0.5*delta));
  b.x = 1.0-delta;
  r = uniform4(prst);
  cos_theta=2.0*r.x-1.0;
  sin_theta=sqrt(1.0-cos_theta*cos_theta);
  sin_alpha=sqrt(1-b.x*b.x);
  phi=2.0*Pi*r.y;
  b.y=sin_alpha*sin_theta*cos(phi);
  b.z=sin_alpha*sin_theta*sin(phi);
  b.w=sin_alpha*cos_theta;
  v0=(cfloat_t)(b.x,b.w);
  v1=(cfloat_t)(b.z,b.y);
  v2=(cfloat_t)(-b.z,b.y);
  v3=(cfloat_t)(b.x,-b.w);
  a[0]=cfloat_add(cfloat_mul(v0,u0),cfloat_mul(v1,u2));
  a[1]=cfloat_add(cfloat_mul(v0,u1),cfloat_mul(v1,u3));
  a[2]=cfloat_add(cfloat_mul(v2,u0),cfloat_mul(v3,u2));
  a[3]=cfloat_add(cfloat_mul(v2,u1),cfloat_mul(v3,u3));
}

//[inject:paths]

kernel void heatbath(global cfloat_t *U,
                     int d,
                     int n,
                     float beta,
                     int n_iter,
                     int m_iter,
                     struct bbox_t bbox,
                     global ranluxcl_state_t *ranluxcltab) {
  size_t gid = get_global_id(0);
  size_t idx = gid2idx(gid,&bbox);
  size_t ixmu, ik, jk;
  cfloat_t a[4], tmp;
  cfloat_t staples[MAXN*MAXN];

  ranluxcl_state_t rst;
  ranluxcl_download_seed(&rst,ranluxcltab);

  for(int step=0; step<n_iter; step++)
    for(int mu=0; mu<d; mu++) {
      ixmu = (idx*d+mu)*n*n;

      //[inject:heatbath_action]
      // something like
      // if(mu==0) aux0(staples,U,idx,&bbox);
      // if(mu==1) aux1(staples,U,idx,&bbox);
      // if(mu==2) aux2(staples,U,idx,&bbox);
      // if(mu==3) aux3(staples,U,idx,&bbox);

      for(int iter=0; iter<m_iter; iter++)
        for(int i=0; i<n-1; i++)
          for(int j=i+1; j<n; j++) {
            a[0]=(cfloat_t)(0.0,0.0);
            a[1]=(cfloat_t)(0.0,0.0);
            a[2]=(cfloat_t)(0.0,0.0);
            a[3]=(cfloat_t)(0.0,0.0);
            for(int k=0; k<n; k++) {

              a[0] = cfloat_add(a[0],cfloat_mul(U[ixmu+i*n+k],
                                                cfloat_conj(staples[i*n+k])));
              a[1] = cfloat_add(a[1],cfloat_mul(U[ixmu+i*n+k],
                                                cfloat_conj(staples[j*n+k])));
              a[2] = cfloat_add(a[2],cfloat_mul(U[ixmu+j*n+k],
                                                cfloat_conj(staples[i*n+k])));
              a[3] = cfloat_add(a[3],cfloat_mul(U[ixmu+j*n+k],
                                                cfloat_conj(staples[j*n+k])));
            }
            heatbath_SU2(a,beta/n,&rst);
            for(int k=0; k<n; k++) {
              ik = ixmu+i*n+k;
              jk = ixmu+j*n+k;
              tmp = cfloat_add(cfloat_mul(a[0],U[ik]),cfloat_mul(a[1],U[jk]));
              U[jk] = cfloat_add(cfloat_mul(a[2],U[ik]),cfloat_mul(a[3],U[jk]));
              U[ik] = tmp;
            }
          }
    }
  ranluxcl_upload_seed(&rst,ranluxcltab);
}


kernel void smear_links(global cfloat_t *V,
                        global cfloat_t *U,
                        int d,
                        int n,
                        struct bbox_t bbox) {
  size_t gid = get_global_id(0);
  size_t idx = gid2idx(gid,&bbox);
  size_t ixmu;
  cfloat_t staples[MAXN*MAXN];

  for(int mu=0; mu<d; mu++) {
    ixmu = (idx*d+mu)*n*n;

    //[inject:smear_links]
    // something like
    // if(mu==0) aux0(V+ixmu,U,idx,&bbox);
    // if(mu==1) aux1(V+ixmu,U,idx,&bbox);
    // if(mu==2) aux2(V+ixmu,U,idx,&bbox);
    // if(mu==3) aux3(V+ixmu,U,idx,&bbox);
    for(int k=0; k<n*n; k++) V[ixmu+k] = staples[k];
  }
}


kernel void fermi_operator(global cfloat_t *phi,
                           global cfloat_t *U,
                           global cfloat_t *psi,
                           struct bbox_t bbox
                           //[inject:extra_fields]
                           ) {

  size_t gid = get_global_id(0);
  size_t idx = gid2idx(gid,&bbox);
  size_t idx2;
  global cfloat_t *p;
  global cfloat_t *q;
  struct shift_t delta;
  cfloat_t path[MAXN*MAXN];
  cfloat_t spinor[MAXN*MAXN];
  cfloat_t coeff;

  //[inject:fermi_operator]
}


kernel void staggered_operator(global cfloat_t *phi,
                               global cfloat_t *U,
                               global cfloat_t *psi,
                               struct bbox_t bbox
                               //[inject:extra_fields]
                               ) {

  size_t gid = get_global_id(0);
  size_t idx = gid2idx(gid,&bbox);
  size_t idx2;
  global cfloat_t *p;
  global cfloat_t *q;
  struct shift_t delta;
  cfloat_t path[MAXN*MAXN];
  cfloat_t spinor[MAXN];
  double coeff;

  //[inject:staggered_operator]
}
