// Dual-architecture XMX matrix helpers: 16-lane Xe2 (xmx_xe2.h) or 8-lane Xe-HPG (xmx_hpg.h).
#ifndef KERNELS_XMX_H
#define KERNELS_XMX_H

#include "builtins.h"

#ifdef ENGINE_HPG
#include "xmx_hpg.h"
#else
#include "xmx_xe2.h"
#endif

#endif
