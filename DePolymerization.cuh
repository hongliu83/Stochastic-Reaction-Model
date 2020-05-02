
#include "NeighborList.cuh"
#include "AllInfo.cuh"
#include "saruprngCUDA.h"


#ifndef __DE_POLYMERIZATION_CUH__
#define __DE_POLYMERIZATION_CUH__



cudaError_t gpu_depolymerization_compute(Real4* d_pos,
										unsigned int* d_rtag,
										unsigned int *d_cris,
										const gpu_boxsize &box,
										unsigned int* d_n_tag_bond,
										uint2* d_tag_bonds,
										unsigned int* d_n_idx_bond,
										uint2* d_idx_bonds,
										unsigned int pitch,
										Real4* d_params,
										Real T,
										unsigned int seed,
										unsigned int coeff_width,
										unsigned int Np,
										unsigned int* d_change_type,
										int blocksize);

#endif


