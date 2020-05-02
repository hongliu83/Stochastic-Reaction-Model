

#include "NeighborList.cuh"
#include "AllInfo.cuh"
#include "saruprngCUDA.h"


#ifndef __POLYMERIZATION_DM_CUH__
#define __POLYMERIZATION_DM_CUH__

struct Reaction_Data
	{
	unsigned int* d_n_tag_bond;
	uint2* d_tag_bonds;
	unsigned int* d_n_idx_bond;
	uint2* d_idx_bonds;
	uint2* d_bond_state;
	unsigned int bond_pitch;
	bool bond_exclusions;
	
	unsigned int* d_n_tag_angle;
	uint4* d_tag_angles;
	unsigned int* d_n_idx_angle;
	uint4* d_idx_angles;
	unsigned int angle_pitch;
	bool angle_generate;
	bool angle_exclusions;

	unsigned int* d_n_ex_tag;
	unsigned int* d_ex_list_tag;
	unsigned int* d_n_ex_idx;
	unsigned int* d_ex_list_idx;
	Index2D ex_list_indexer;
	
	unsigned int* d_bond_type_table;
	unsigned int* d_angle_type_table;
	bool bond_type_by_pair;
	bool angle_type_by_pair;
	unsigned int* d_change_type;
	Real angle_limit;
	};

cudaError_t gpu_FRP_DM_compute(Real4* d_pos,
					unsigned int* d_tag,
					unsigned int* d_rtag,
					const gpu_boxsize &box, 
					unsigned int *d_n_neigh,
					unsigned int *d_nlist,
					const Index2D& nli,
					const Reaction_Data& reaction_data,
					unsigned int *d_cris,
					unsigned int seed,
					Real rcutsq,
					unsigned int coeff_width,
					Real3* d_pr,
					unsigned int* h_ninit,
					unsigned int* d_ninit,
					unsigned int* d_init_group,
					unsigned int* d_init,
					unsigned int Np,
					unsigned int new_bond_type,
					unsigned int new_angle_type,					
					unsigned int period_R,					
					int blocksize);
					
cudaError_t gpu_exchange_DM_compute(Real4* d_pos,
					unsigned int* d_tag,
					unsigned int* d_rtag,
					const gpu_boxsize &box, 
					const unsigned int *d_n_neigh,
					const unsigned int *d_nlist,
					const Index2D& nli,
					const Reaction_Data& reaction_data,
					unsigned int *d_cris,
					unsigned int seed,
					Real rcutsq,
					unsigned int coeff_width,
					Real3* d_pr,
					unsigned int* h_ninit,
					unsigned int* d_ninit,
					unsigned int* d_init_group,
					unsigned int* d_init,
					unsigned int* d_maxcris,
					unsigned int Np,
					unsigned int period_R,					
					int blocksize);

cudaError_t gpu_SGAP_DM_compute(Real4* d_pos,
					unsigned int* d_tag,
					unsigned int* d_rtag,
					const gpu_boxsize &box, 
					unsigned int *d_n_neigh,
					unsigned int *d_nlist,
					const Index2D& nli,
					const Reaction_Data& reaction_data,
					unsigned int *d_cris,
					unsigned int seed,
					Real rcutsq,
					unsigned int coeff_width,
					Real3* d_pr,
					unsigned int* h_ninit,
					unsigned int* d_ninit,
					unsigned int* d_init_group,
					unsigned int* d_init,					
					unsigned int* d_maxcris,
					unsigned int Np,
					unsigned int new_bond_type,
					unsigned int new_angle_type,
					unsigned int period_R,
					int blocksize);
							
#endif


