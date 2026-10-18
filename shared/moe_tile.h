// A tile of the prompt path's expert products: up to MOE_TILE consecutive rows of the gathered activations, all
// routed to one expert (kernels/quant/qgemm.cl). Written by the routing kernels on the card, read by the products.
#ifndef SHARED_MOE_TILE_H
#define SHARED_MOE_TILE_H

#include "quant.h"
#include "types.h"

#define MOE_TILE 64  // a multiple of 32

typedef struct {
  u32 expert;
  u32 slot;   // the expert's cache slot, or its staging slot
  u32 start;  // first row
  u32 count;  // rows, 1 .. MOE_TILE
} MoeTile;

SHARED_ASSERT(sizeof(MoeTile) == 16, "MoeTile layout");

// The decode step's experts grouped (kernels/quant/moe.cl and matvec.cl): one cached expert and the assignments
// (row * experts per row + rank) of the step that route to it, so its weights are decoded once for all of them.
#define MOE_GROUP_MAX 8
typedef struct {
  i32 slot;                  // the expert's cache slot
  u32 n;                     // assignments, 1 .. MOE_GROUP_MAX
  u32 asg[MOE_GROUP_MAX];
} MoeGroup;

SHARED_ASSERT(sizeof(MoeGroup) == 40, "MoeGroup layout");

// The shape of kernels/quant/moe.cl's products, for groups of up to a assignments: every work-group computes
// MOE_WG_ROWS rows of a group's expert, each sub-group MOE_GU_ROWS(a) rows of gate and up or MOE_DOWN_ROWS(type, a)
// rows of down.
#define MOE_WG_ROWS 32
#define MOE_GU_ROWS(a) ((a) <= 2 ? 4 : 2)
#define MOE_DOWN_ROWS(type, a) ((type) == QT_Q2_0 && (a) <= 2 ? 8 : 4)

#endif
