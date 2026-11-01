// The SwiGLU of an expert's gate and up values, silu(g) * u, for the file that includes this and defines silu. Built
// with -DSWIGLU_LIMIT=L, gate becomes min(g, L) and up clamp(u, -L, L) before the SiLU, as ExpertShape::swiglu_limit
// does on the CPU (docs/spec/MODEL_LAYER.md, section 4.3). Without it the macro is the plain product.
#ifndef KERNELS_SWIGLU_H
#define KERNELS_SWIGLU_H

#ifdef SWIGLU_LIMIT
#define SWIGLU(g, u) \
  (silu(fmin((g), (float)(SWIGLU_LIMIT))) * clamp((u), -(float)(SWIGLU_LIMIT), (float)(SWIGLU_LIMIT)))
#else
#define SWIGLU(g, u) (silu(g) * (u))
#endif

#endif
