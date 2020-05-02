

#include "PolymerizationDM.cuh"
#include <assert.h>

texture<float4, 1, cudaReadModeElementType> pos_tex;


__device__ inline void filter_nlist(unsigned int *d_n_neigh,
								 unsigned int *d_nlist,
								 Index2D nli,
								 unsigned int idxi,
								 unsigned int idxj)
    {
    const unsigned int n_neighi = d_n_neigh[idxi];
    const unsigned int n_neighj = d_n_neigh[idxj];
	unsigned int new_n_neighi =0;
	unsigned int new_n_neighj =0;	
    for (unsigned int cur_neigh_idx = 0; cur_neigh_idx < n_neighi; cur_neigh_idx++)
        {
        unsigned int cur_neigh = d_nlist[nli(idxi, cur_neigh_idx)];
        if (cur_neigh!=idxj)
            {
            d_nlist[nli(idxi, new_n_neighi)] = cur_neigh;
            new_n_neighi++;
            }
        }
    for (unsigned int cur_neigh_idx = 0; cur_neigh_idx < n_neighj; cur_neigh_idx++)
        {
        unsigned int cur_neigh = d_nlist[nli(idxj, cur_neigh_idx)];
        if (cur_neigh!=idxi)
            {
            d_nlist[nli(idxj, new_n_neighj)] = cur_neigh;
            new_n_neighj++;
            }
        }
    d_n_neigh[idxi] = new_n_neighi;
    d_n_neigh[idxj] = new_n_neighj;
    }


__global__ void gpu_compute_FRP_DM_kernel(Real4* d_pos,
										unsigned int* d_tag,
										unsigned int* d_rtag,
										gpu_boxsize box, 
										unsigned int *d_n_neigh,
										unsigned int *d_nlist, 
										Index2D nli, 
										Reaction_Data reaction_data,
										unsigned int *d_cris,
										unsigned int seed,
										Real rcutsq,
										unsigned int coeff_width,					
										Real3* d_pr,
										unsigned int* ninit,
										unsigned int* d_init_group,
										unsigned int* d_init,
										unsigned int new_bond_type,
										unsigned int new_angle_type,	
										unsigned int period_R)
	{
	extern __shared__ Real3 s_pr[];
	for (unsigned int cur_offset = 0; cur_offset < coeff_width*coeff_width; cur_offset += blockDim.x)
		{
		if (cur_offset + threadIdx.x < coeff_width*coeff_width)
			s_pr[cur_offset + threadIdx.x] = d_pr[cur_offset + threadIdx.x];
		}
	__syncthreads();
	
	unsigned int goup_idx = blockIdx.x * blockDim.x + threadIdx.x;	
	if (goup_idx >= ninit[0])
		return;

    unsigned int tagi = d_init_group[goup_idx];
	uint2 state = reaction_data.d_bond_state[tagi];
	if(state.x>0)
		{
		state.x += 1;
		if (state.x>=period_R+1)
			state.x = 0;
		reaction_data.d_bond_state[tagi] = state;
//		printf("thread tagi state %d, %d\n", tagi, state);
		}
	if(state.x>0)
		return;

	unsigned int idx = d_rtag[tagi];

	unsigned int n_neigh = d_n_neigh[idx];
	Real4 pos = GetPos(idx);
	int typi  = __real_as_int(pos.w);

	unsigned int mintag = 0;
	unsigned int minidx = 0;
	Real mindisq = 10000.0f;
	unsigned int mintype =0;
	unsigned int mintype_changed =0;
	
    unsigned int cur_neigh = 0;
    unsigned int next_neigh = d_nlist[nli(idx, 0)];
	
	for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
		{
		cur_neigh = next_neigh;
		next_neigh = d_nlist[nli(idx, neigh_idx+1)];
		
		Real4 neigh_pos = GetPos(cur_neigh);
		
		Real dx = pos.x - neigh_pos.x;
		Real dy = pos.y - neigh_pos.y;
		Real dz = pos.z - neigh_pos.z;
			
		dx -= box.lx * rint(dx * box.lxinv);
		dy -= box.ly * rint(dy * box.lyinv);
		dz -= box.lz * rint(dz * box.lzinv);
			

		Real rsq = dx*dx + dy*dy + dz*dz;
        unsigned int tagj = d_tag[cur_neigh];
		unsigned int initj = d_init[tagj]; 
        unsigned int crisj = d_cris[tagj];
		
		int typj = __real_as_int(neigh_pos.w);
		
		if (rsq < rcutsq && crisj == 0 && initj==0)
			{	
			if(rsq<mindisq)
				{
				mindisq=rsq;
				mintag = tagj;
				minidx = cur_neigh;
				mintype = typj;
				mintype_changed = reaction_data.d_change_type[mintype];
				}
		    }			
		}
		
	if (mindisq<10000.0f)
		{
		SaruGPU RNG(seed, tagi, mintag);
		int typ_pair = typi * coeff_width + mintype;	
		Real3 Pr = s_pr[typ_pair];
		
		Real ran = Rondom(0.0f,1.0f);
		if(ran < Pr.z)
			{
			unsigned int old = atomicMax(&d_cris[mintag],1);
			if(old ==0)
				{
				d_init_group[goup_idx] = mintag;
				d_init[mintag] = 1;
				d_cris[tagi] += 1;				  
				d_init[tagi] = 0;
				if(mintype!=mintype_changed)
					d_pos[minidx].w = __int_as_real(mintype_changed);
				unsigned int numi = reaction_data.d_n_tag_bond[tagi];
				unsigned int numj = reaction_data.d_n_tag_bond[mintag];

				if(reaction_data.angle_generate)
					{
					for(unsigned int i = 0; i<numi; i++)
						{
						uint2 bondi = reaction_data.d_tag_bonds[i*reaction_data.bond_pitch + tagi];
						unsigned int taga = bondi.x;
						unsigned int idxa = d_rtag[taga];
						Real4 posa = GetPos(idxa);
						int typa = __real_as_int(posa.w);
						
						unsigned int angle_type = new_angle_type;
						if(reaction_data.angle_type_by_pair)
							angle_type = reaction_data.d_angle_type_table[typa * coeff_width * coeff_width + typi * coeff_width + mintype_changed];
						
						unsigned int num_angle_a = reaction_data.d_n_tag_angle[taga];						
						unsigned int num_angle_b = reaction_data.d_n_tag_angle[tagi];
						unsigned int num_angle_c = reaction_data.d_n_tag_angle[mintag];
						
						reaction_data.d_tag_angles[num_angle_a*reaction_data.angle_pitch + taga] = make_uint4(tagi, mintag, angle_type, 0);
						reaction_data.d_tag_angles[num_angle_b*reaction_data.angle_pitch + tagi] = make_uint4(taga, mintag, angle_type, 1);
						reaction_data.d_tag_angles[num_angle_c*reaction_data.angle_pitch + mintag] = make_uint4(taga, tagi, angle_type, 2);	

						reaction_data.d_idx_angles[num_angle_a*reaction_data.angle_pitch + idxa] = make_uint4(idx, minidx, angle_type, 0);
						reaction_data.d_idx_angles[num_angle_b*reaction_data.angle_pitch + idx] = make_uint4(idxa, minidx, angle_type, 1);
						reaction_data.d_idx_angles[num_angle_c*reaction_data.angle_pitch + minidx] = make_uint4(idxa, idx, angle_type, 2);
						
						reaction_data.d_n_tag_angle[taga] = num_angle_a + 1;	
						reaction_data.d_n_tag_angle[tagi] = num_angle_b + 1;
						reaction_data.d_n_tag_angle[mintag] = num_angle_c + 1;

						reaction_data.d_n_idx_angle[idxa] = num_angle_a + 1;	
						reaction_data.d_n_idx_angle[idx] = num_angle_b + 1;
						reaction_data.d_n_idx_angle[minidx] = num_angle_c + 1;
						
						if(reaction_data.angle_exclusions)
							{
							unsigned int nexi = reaction_data.d_n_ex_tag[taga];
							unsigned int nexj = reaction_data.d_n_ex_tag[mintag];
							
							reaction_data.d_ex_list_tag[reaction_data.ex_list_indexer(taga, nexi)] = mintag;
							reaction_data.d_ex_list_tag[reaction_data.ex_list_indexer(mintag, nexj)] = taga;
							
							reaction_data.d_ex_list_idx[reaction_data.ex_list_indexer(idxa, nexi)] = minidx;
							reaction_data.d_ex_list_idx[reaction_data.ex_list_indexer(minidx, nexj)] = idxa;
							
							reaction_data.d_n_ex_tag[taga] = nexi + 1;
							reaction_data.d_n_ex_tag[mintag] = nexj + 1;
							
							reaction_data.d_n_ex_idx[idxa] = nexi + 1;
							reaction_data.d_n_ex_idx[minidx] = nexj + 1;
							filter_nlist(d_n_neigh, d_nlist, nli, idxa, minidx);
							}
						}
					for(unsigned int i = 0; i<numj; i++)
						{
						uint2 bondk = reaction_data.d_tag_bonds[i*reaction_data.bond_pitch + mintag];
						unsigned int tagk = bondk.x;
						unsigned int idxk = d_rtag[tagk];
						Real4 posk = GetPos(idxk);
						int typk = __real_as_int(posk.w);
						
						unsigned int angle_type = new_angle_type;
						if(reaction_data.angle_type_by_pair)
							angle_type = reaction_data.d_angle_type_table[typi * coeff_width * coeff_width + mintype_changed* coeff_width + typk];
						
						unsigned int num_angle_i = reaction_data.d_n_tag_angle[tagi];						
						unsigned int num_angle_j = reaction_data.d_n_tag_angle[mintag];
						unsigned int num_angle_k = reaction_data.d_n_tag_angle[tagk];
						
						reaction_data.d_tag_angles[num_angle_i*reaction_data.angle_pitch + tagi] = make_uint4(mintag, tagk, angle_type, 0);
						reaction_data.d_tag_angles[num_angle_j*reaction_data.angle_pitch + mintag] = make_uint4(tagi, tagk, angle_type, 1);
						reaction_data.d_tag_angles[num_angle_k*reaction_data.angle_pitch + tagk] = make_uint4(tagi, mintag, angle_type, 2);	

						reaction_data.d_idx_angles[num_angle_i*reaction_data.angle_pitch + idx] = make_uint4(minidx, idxk, angle_type, 0);
						reaction_data.d_idx_angles[num_angle_j*reaction_data.angle_pitch + minidx] = make_uint4(idx, idxk, angle_type, 1);
						reaction_data.d_idx_angles[num_angle_k*reaction_data.angle_pitch + idxk] = make_uint4(idx, minidx, angle_type, 2);
						
						reaction_data.d_n_tag_angle[tagi] = num_angle_i + 1;	
						reaction_data.d_n_tag_angle[mintag] = num_angle_j + 1;
						reaction_data.d_n_tag_angle[tagk] = num_angle_k + 1;

						reaction_data.d_n_idx_angle[idx] = num_angle_i + 1;	
						reaction_data.d_n_idx_angle[minidx] = num_angle_j + 1;
						reaction_data.d_n_idx_angle[idxk] = num_angle_k + 1;
						
						if(reaction_data.angle_exclusions)
							{
							unsigned int nexi = reaction_data.d_n_ex_tag[tagi];
							unsigned int nexk = reaction_data.d_n_ex_tag[tagk];
							
							reaction_data.d_ex_list_tag[reaction_data.ex_list_indexer(tagi, nexi)] = tagk;
							reaction_data.d_ex_list_tag[reaction_data.ex_list_indexer(tagk, nexk)] = tagi;
							
							reaction_data.d_ex_list_idx[reaction_data.ex_list_indexer(idx, nexi)] = idxk;
							reaction_data.d_ex_list_idx[reaction_data.ex_list_indexer(idxk, nexk)] = idx;
							
							reaction_data.d_n_ex_tag[tagi] = nexi + 1;
							reaction_data.d_n_ex_tag[tagk] = nexk + 1;
							
							reaction_data.d_n_ex_idx[idx] = nexi + 1;
							reaction_data.d_n_ex_idx[idxk] = nexk + 1;
							filter_nlist(d_n_neigh, d_nlist, nli, idx, idxk);
							}
						}
					}
				unsigned int bond_type = new_bond_type;
				if(reaction_data.bond_type_by_pair)
					bond_type = reaction_data.d_bond_type_table[typi * coeff_width + mintype_changed];
				
				reaction_data.d_tag_bonds[numi*reaction_data.bond_pitch + tagi] = make_uint2(mintag, bond_type);
				reaction_data.d_tag_bonds[numj*reaction_data.bond_pitch + mintag] = make_uint2(tagi, bond_type);
				reaction_data.d_n_tag_bond[tagi] = numi + 1;
				reaction_data.d_n_tag_bond[mintag] = numj + 1;
				  
				reaction_data.d_idx_bonds[numi*reaction_data.bond_pitch + idx] = make_uint2(minidx, bond_type);
				reaction_data.d_idx_bonds[numj*reaction_data.bond_pitch + minidx] = make_uint2(idx, bond_type);
				reaction_data.d_n_idx_bond[idx] = numi + 1;
				reaction_data.d_n_idx_bond[minidx] = numj + 1;				  
				reaction_data.d_bond_state[mintag] = make_uint2(1, tagi);

				if(reaction_data.bond_exclusions)
					{
					unsigned int nexi = reaction_data.d_n_ex_tag[tagi];
					unsigned int nexj = reaction_data.d_n_ex_tag[mintag];
					
					reaction_data.d_ex_list_tag[reaction_data.ex_list_indexer(tagi, nexi)] = mintag;
					reaction_data.d_ex_list_tag[reaction_data.ex_list_indexer(mintag, nexj)] = tagi;
					
					reaction_data.d_ex_list_idx[reaction_data.ex_list_indexer(idx, nexi)] = minidx;
					reaction_data.d_ex_list_idx[reaction_data.ex_list_indexer(minidx, nexj)] = idx;
					
					reaction_data.d_n_ex_tag[tagi] = nexi + 1;
					reaction_data.d_n_ex_tag[mintag] = nexj + 1;
					
					reaction_data.d_n_ex_idx[idx] = nexi + 1;
					reaction_data.d_n_ex_idx[minidx] = nexj + 1;
					filter_nlist(d_n_neigh, d_nlist, nli, idx, minidx);
					//printf("thread %d, %d, %d, %d, %d, %d\n", tagi, mintag, idx, minidx, nexi, nexj);
					}
				atomicAdd(&ninit[3], (unsigned int)1);
//		printf("thread tagi tagj %d, %d\n", tagi, mintag);			  
				}
		     }
		   			
		}

	}
	
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
							int blocksize)
	{

    pos_tex.normalized = false;
    pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pos_tex, d_pos, sizeof(float4) * Np);
    if (error != cudaSuccess)
        return error;
    dim3 grid( (int)ceil((Real)h_ninit[0] / (Real)blocksize), 1, 1);
    dim3 threads(blocksize, 1, 1);	
	gpu_compute_FRP_DM_kernel<<< grid, threads, sizeof(Real3)*coeff_width*coeff_width>>>(d_pos,
																						d_tag,
																						d_rtag, 
																						box, 
																						d_n_neigh,
																						d_nlist, 
																						nli,				
																						reaction_data,
																						d_cris,
																						seed,
																						rcutsq,
																						coeff_width,
																						d_pr,
																						d_ninit,
																						d_init_group,
																						d_init,
																						new_bond_type,
																						new_angle_type,																							
																						period_R);	

																																	

    return cudaSuccess;
	}	

__global__ void gpu_compute_exchange_DM_kernel(Real4* d_pos,
											unsigned int* d_tag,
											unsigned int* d_rtag, 
											gpu_boxsize box, 
											const unsigned int *d_n_neigh,
											const unsigned int *d_nlist, 
											Index2D nli, 
											unsigned int* d_n_tag_bond,
											uint2* d_tag_bonds,
											unsigned int* d_n_idx_bond,
											uint2* d_idx_bonds,
											uint2* d_bond_state,
											unsigned int bond_pitch,
											unsigned int *d_cris,
											unsigned int seed,
											Real rcutsq,
											unsigned int coeff_width,					
											Real3* d_pr,
											unsigned int* ninit,
											unsigned int* d_init_group,
											unsigned int* d_init,
											unsigned int* d_maxcris,
											unsigned int period_R)
	{
	extern __shared__ Real3 s_pr[];
	for (unsigned int cur_offset = 0; cur_offset < coeff_width*coeff_width*coeff_width; cur_offset += blockDim.x)
		{
		if (cur_offset + threadIdx.x < coeff_width*coeff_width*coeff_width)
			s_pr[cur_offset + threadIdx.x] = d_pr[cur_offset + threadIdx.x];
		}
	__syncthreads();
	
	unsigned int goup_idx = blockIdx.x * blockDim.x + threadIdx.x;	
	if (goup_idx >= ninit[0])
		return;

    unsigned int tagi = d_init_group[goup_idx];
	uint2 state = d_bond_state[tagi];
	if(state.x>0)
		{
		state.x += 1;
		if (state.x>=period_R+1)
			state.x = 0;
		d_bond_state[tagi] = state;
//		printf("thread tagi state %d, %d\n", tagi, state);
		}
	if(state.x>0)
		return;
		
	unsigned int idx = d_rtag[tagi];

	unsigned int n_neigh = d_n_neigh[idx];
	Real4 pos = GetPos(idx);
	int typi  = __real_as_int(pos.w);	
	unsigned int numi = d_n_tag_bond[tagi];
	
	unsigned int mintag = 0;
	unsigned int minidx = 0;
	Real mindisq = 10000.0f;
	unsigned int mintype =0;
	Real minPossi = 0.0f;
	unsigned int minmaxcris=0;

    unsigned int cur_neigh = 0;
    unsigned int next_neigh = d_nlist[nli(idx, 0)];
	
	for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
		{
		cur_neigh = next_neigh;
		next_neigh = d_nlist[nli(idx, neigh_idx+1)];
		
		Real4 neigh_pos = GetPos(cur_neigh);
		
		Real dx = pos.x - neigh_pos.x;
		Real dy = pos.y - neigh_pos.y;
		Real dz = pos.z - neigh_pos.z;
			
		dx -= box.lx * rint(dx * box.lxinv);
		dy -= box.ly * rint(dy * box.lyinv);
		dz -= box.lz * rint(dz * box.lzinv);

		Real rsq = dx*dx + dy*dy + dz*dz;
        unsigned int tagj = d_tag[cur_neigh];
		unsigned int initj = d_init[tagj]; 		
        unsigned int crisj = d_cris[tagj];
		int typj = __real_as_int(neigh_pos.w);
		unsigned int max_crisj = d_maxcris[typj];

		if (rsq < rcutsq && crisj < max_crisj && initj==0)
			{
			Real Possib =0.0f;
			bool bonded = false; 			
			for( unsigned int b =0; b<numi; b++)
				{
				unsigned int temp_tagj = d_tag_bonds[b*bond_pitch + tagi].x;
				if(temp_tagj==tagj)
					bonded=true;				
				unsigned int temp_idxj = d_rtag[temp_tagj];					  
				Real4 temp_posj = GetPos(temp_idxj);
				int temp_typj = __real_as_int(temp_posj.w);
				int typ_pair = typj * coeff_width* coeff_width + typi * coeff_width + temp_typj;	
				Real3 Pr = s_pr[typ_pair];
				Possib += Pr.z;
				}
			if(rsq<mindisq&&Possib>0.0f&&!bonded)
				{
				mindisq=rsq;
				mintag = tagj;
				minidx = cur_neigh;
				mintype = typj;
				minPossi = Possib;
				minmaxcris=max_crisj;
				}
		    }			
		}

	if (mindisq<10000.0f)
		{			
		SaruGPU RNG(seed, tagi, mintag);
		Real ran = Rondom(0.0f,1.0f);
		unsigned int initi = 1;	
		if(ran < minPossi)
			{
			unsigned int old_tagj = atomicMax(&d_cris[mintag],NO_INDEX);
			if(old_tagj < minmaxcris)
				{
				bool success =false;
				for( unsigned int b =0; b<numi&&initi==1; b++)
					{
					unsigned int temp_tagj = d_tag_bonds[b*bond_pitch + tagi].x;
					unsigned int temp_bond_type = d_tag_bonds[b*bond_pitch + tagi].y;
					unsigned int temp_idxj = d_rtag[temp_tagj];					  
					Real4 temp_posj = GetPos(temp_idxj);
					int temp_typj = __real_as_int(temp_posj.w);
					int typ_pair = mintype * coeff_width* coeff_width + typi * coeff_width + temp_typj;	
					Real3 Pr = s_pr[typ_pair];
					ran -= Pr.z;
					unsigned int cris_tempj = d_cris[temp_tagj];
					if(ran<=0.0f&&cris_tempj>0&&cris_tempj!=NO_INDEX)
						{
						unsigned int old_temp_tagj = atomicMax(&d_cris[temp_tagj],NO_INDEX);
						if(old_temp_tagj!=NO_INDEX&&old_temp_tagj>0)
							{
							d_tag_bonds[b*bond_pitch + tagi].x = mintag;
							d_idx_bonds[b*bond_pitch + idx].x = minidx;							
							unsigned int temp_numj = d_n_tag_bond[temp_tagj];
							unsigned int count =0;
							for(unsigned int j =0; j<temp_numj;j++)
								{
								uint2 temp_tag_bond = d_tag_bonds[j*bond_pitch + temp_tagj];
								uint2 temp_idx_bond = d_idx_bonds[j*bond_pitch + temp_idxj];								
								if(temp_tag_bond.x!=tagi)
									{
									d_tag_bonds[count*bond_pitch + temp_tagj] = temp_tag_bond;
									d_idx_bonds[count*bond_pitch + temp_idxj] = temp_idx_bond;									
									count +=1;
									}
								}
							d_n_tag_bond[temp_tagj] = count;
							d_n_idx_bond[temp_idxj] = count;
							d_cris[temp_tagj] =old_temp_tagj-1;
							initi = 0;
							unsigned int numj = d_n_tag_bond[mintag];
							d_tag_bonds[numj*bond_pitch + mintag] = make_uint2(tagi, temp_bond_type);
							d_idx_bonds[numj*bond_pitch + minidx] = make_uint2(idx, temp_bond_type);							
							d_n_tag_bond[mintag] = numj + 1;
							d_n_idx_bond[minidx] = numj + 1;
							d_bond_state[tagi] = make_uint2(1, mintag);
							success = true;
							}
						else if(old_temp_tagj==0)
							d_cris[temp_tagj]=0;
						} 
					}
				if(success)
					d_cris[mintag] = old_tagj+1;
				else
					d_cris[mintag] = old_tagj;
				}
			else if(old_tagj !=NO_INDEX)
				{
				d_cris[mintag] = old_tagj;
				}
		    }			
		}

	}

	
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
								int blocksize)
	{

    pos_tex.normalized = false;
    pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pos_tex, d_pos, sizeof(float4) * Np);
    if (error != cudaSuccess)
        return error;
    dim3 grid( (int)ceil((Real)h_ninit[0] / (Real)blocksize), 1, 1);
    dim3 threads(blocksize, 1, 1);	
	gpu_compute_exchange_DM_kernel<<< grid, threads, sizeof(Real3)*coeff_width*coeff_width*coeff_width>>>(d_pos,
																							d_tag,
																							d_rtag, 
																							box, 
																							d_n_neigh,
																							d_nlist, 
																							nli,					
																							reaction_data.d_n_tag_bond,
																							reaction_data.d_tag_bonds,
																							reaction_data.d_n_idx_bond,
																							reaction_data.d_idx_bonds,
																							reaction_data.d_bond_state,
																							reaction_data.bond_pitch,
																							d_cris,
																							seed,
																							rcutsq,
																							coeff_width,
																							d_pr,
																							d_ninit,
																							d_init_group,
																							d_init,
																							d_maxcris,
																							period_R);	
	
																																	

    return cudaSuccess;
	}

__global__ void gpu_compute_SGAP_DM_kernel(Real4* d_pos,
								unsigned int* d_tag,
								unsigned int* d_rtag,  
								gpu_boxsize box, 
								unsigned int *d_n_neigh,
								unsigned int *d_nlist, 
								Index2D nli, 
								Reaction_Data reaction_data,
								unsigned int *d_cris,
								unsigned int seed,
								Real rcutsq,
								unsigned int coeff_width,
								Real3* d_pr,					
								unsigned int* ninit,
								unsigned int* d_init_group,
								unsigned int* d_init,					
								unsigned int* d_maxcris,
								unsigned int new_bond_type,
								unsigned int new_angle_type,									
								unsigned int period_R)
	{
	extern __shared__ Real3 s_pr[];
	for (unsigned int cur_offset = 0; cur_offset < coeff_width*coeff_width; cur_offset += blockDim.x)
		{
		if (cur_offset + threadIdx.x < coeff_width*coeff_width)
			s_pr[cur_offset + threadIdx.x] = d_pr[cur_offset + threadIdx.x];
		}
	__syncthreads();
	
	unsigned int goup_idx = blockIdx.x * blockDim.x + threadIdx.x;	
	if (goup_idx >= ninit[0])
		return;
    unsigned int tagi = d_init_group[goup_idx];
	uint2 state = reaction_data.d_bond_state[tagi];
	if(state.x>0)
		{
		state.x += 1;
		if (state.x>=period_R+1)
			state.x = 0;
		reaction_data.d_bond_state[tagi] = state;
//		printf("thread tagi state %d, %d, %d\n", tagi, state.x, state.y);
		}
	if(state.x>0)
		return;	

	unsigned int idx = d_rtag[tagi];
    unsigned int initi = d_init[tagi];
    unsigned int numi = reaction_data.d_n_tag_bond[tagi];	
	unsigned int n_neigh = d_n_neigh[idx];
	Real4 pos = GetPos(idx);
	
	unsigned int typi = __real_as_int(pos.w);
	unsigned int max_crisi = d_maxcris[typi];
    unsigned int crisi = d_cris[tagi];
	if(crisi>=max_crisi||initi==0)
		{
		return;
		}
		
	unsigned int mintag = 0;
	unsigned int minidx = 0;
	Real mindisq = 10000.0f;
	unsigned int mintype = 0;
	unsigned int mincris = 0;
	unsigned int mintype_changed = 0;		

    unsigned int cur_neigh = 0;
    unsigned int next_neigh = d_nlist[nli(idx, 0)];
	
	for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
		{
		cur_neigh = next_neigh;
		next_neigh = d_nlist[nli(idx, neigh_idx+1)];
		
		Real4 neigh_pos = GetPos(cur_neigh);
		
		Real dx = pos.x - neigh_pos.x;
		Real dy = pos.y - neigh_pos.y;
		Real dz = pos.z - neigh_pos.z;
		unsigned int typj = __real_as_int(neigh_pos.w);
		dx -= box.lx * rintf(dx * box.lxinv);
		dy -= box.ly * rintf(dy * box.lyinv);
		dz -= box.lz * rintf(dz * box.lzinv);
			

		Real rsq = dx*dx + dy*dy + dz*dz;
        unsigned int tagj = d_tag[cur_neigh];	
        unsigned int crisj = d_cris[tagj];
		unsigned int max_crisj = d_maxcris[typj];
		unsigned int initj = d_init[tagj];
		
		if (rsq < rcutsq && crisj < max_crisj && initj==0 )
			{
			if(rsq<mindisq)
				{
				bool bonded = false; 
				for( unsigned int b =0; b<numi; b++)
					{
					unsigned int temp_tagj = reaction_data.d_tag_bonds[b*reaction_data.bond_pitch + tagi].x;
					if(temp_tagj==tagj)
						bonded=true;
					}
		
				if (bonded)
					continue;
	// angle check
/* 				Real4 a_pos = neigh_pos;
				Real4 b_pos = pos;
				for(unsigned int i = 0; i<numi; i++)
					{
					uint2 bondi = reaction_data.d_tag_bonds[i*reaction_data.bond_pitch + tagi];
					unsigned int taga = bondi.x;
					unsigned int idxa = d_rtag[taga];
					Real4 c_pos = GetPos(idxa);
					
					Real dxab = a_pos.x - b_pos.x;
					Real dyab = a_pos.y - b_pos.y;
					Real dzab = a_pos.z - b_pos.z;
					
					Real dxcb = c_pos.x - b_pos.x;
					Real dycb = c_pos.y - b_pos.y;
					Real dzcb = c_pos.z - b_pos.z;

					dxab -= box.lx * rintf(dxab * box.lxinv);
					dxcb -= box.lx * rintf(dxcb * box.lxinv);
					
					dyab -= box.ly * rintf(dyab * box.lyinv);
					dycb -= box.ly * rintf(dycb * box.lyinv);
					
					dzab -= box.lz * rintf(dzab * box.lzinv);
					dzcb -= box.lz * rintf(dzcb * box.lzinv);

					Real rsqab = dxab*dxab+dyab*dyab+dzab*dzab;
					Real rab = sqrt(rsqab);
					Real rsqcb = dxcb*dxcb+dycb*dycb+dzcb*dzcb;
					Real rcb = sqrt(rsqcb);
					
					Real c_abbc = dxab*dxcb+dyab*dycb+dzab*dzcb;
					c_abbc /= rab*rcb;

					if(c_abbc>reaction_data.angle_limit)
						angle_fit = false;
					}
					
				a_pos = pos;
				b_pos = neigh_pos;
				unsigned int numj = reaction_data.d_n_tag_bond[tagj];	
				for(unsigned int i = 0; i<numj; i++)
					{
					uint2 bondk = reaction_data.d_tag_bonds[i*reaction_data.bond_pitch + tagj];
					unsigned int tagk = bondk.x;
					unsigned int idxk = d_rtag[tagk];
					Real4 c_pos = GetPos(idxk);
					
					Real dxab = a_pos.x - b_pos.x;
					Real dyab = a_pos.y - b_pos.y;
					Real dzab = a_pos.z - b_pos.z;
					
					Real dxcb = c_pos.x - b_pos.x;
					Real dycb = c_pos.y - b_pos.y;
					Real dzcb = c_pos.z - b_pos.z;

					dxab -= box.lx * rintf(dxab * box.lxinv);
					dxcb -= box.lx * rintf(dxcb * box.lxinv);
					
					dyab -= box.ly * rintf(dyab * box.lyinv);
					dycb -= box.ly * rintf(dycb * box.lyinv);
					
					dzab -= box.lz * rintf(dzab * box.lzinv);
					dzcb -= box.lz * rintf(dzcb * box.lzinv);

					Real rsqab = dxab*dxab+dyab*dyab+dzab*dzab;
					Real rab = sqrt(rsqab);
					Real rsqcb = dxcb*dxcb+dycb*dycb+dzcb*dzcb;
					Real rcb = sqrt(rsqcb);

					Real c_abbc = dxab*dxcb+dyab*dycb+dzab*dzcb;
					c_abbc /= rab*rcb;

					if(c_abbc>reaction_data.angle_limit)
						angle_fit = false;				
					}*/
					
				mindisq = rsq;
				mintag = tagj;
				minidx = cur_neigh;
				mintype = typj;
				mincris = crisj;
				mintype_changed = reaction_data.d_change_type[mintype];					
				} 
			}
		}
//angle check finished
	if (mindisq<10000.0f)
		{
		SaruGPU RNG(seed, tagi, mintag);	
		int typ_pair = typi* coeff_width + mintype;	
		Real3 pr = s_pr[typ_pair];
		Real ran = Rondom(0.0f,1.0f);
        Real factor = powf(pr.y, crisi);
		if(ran < pr.x*factor)
			{
			unsigned int old = atomicMax(&d_cris[mintag], mincris+1);
			if(old == mincris)
				{
				d_cris[tagi] +=1;
				if(mintype!=mintype_changed)
					d_pos[minidx].w = __int_as_real(mintype_changed);
				unsigned int numj = reaction_data.d_n_tag_bond[mintag];				
				if(reaction_data.angle_generate)
					{
					for(unsigned int i = 0; i<numi; i++)
						{
						uint2 bondi = reaction_data.d_tag_bonds[i*reaction_data.bond_pitch + tagi];
						unsigned int taga = bondi.x;
						unsigned int idxa = d_rtag[taga];
						Real4 posa = GetPos(idxa);
						int typa = __real_as_int(posa.w);
						
						unsigned int angle_type = new_angle_type;
						if(reaction_data.angle_type_by_pair)
							angle_type = reaction_data.d_angle_type_table[typa * coeff_width * coeff_width + typi * coeff_width + mintype_changed];
						
						unsigned int num_angle_a = reaction_data.d_n_tag_angle[taga];						
						unsigned int num_angle_b = reaction_data.d_n_tag_angle[tagi];
						unsigned int num_angle_c = reaction_data.d_n_tag_angle[mintag];
						
						reaction_data.d_tag_angles[num_angle_a*reaction_data.angle_pitch + taga] = make_uint4(tagi, mintag, angle_type, 0);
						reaction_data.d_tag_angles[num_angle_b*reaction_data.angle_pitch + tagi] = make_uint4(taga, mintag, angle_type, 1);
						reaction_data.d_tag_angles[num_angle_c*reaction_data.angle_pitch + mintag] = make_uint4(taga, tagi, angle_type, 2);	

						reaction_data.d_idx_angles[num_angle_a*reaction_data.angle_pitch + idxa] = make_uint4(idx, minidx, angle_type, 0);
						reaction_data.d_idx_angles[num_angle_b*reaction_data.angle_pitch + idx] = make_uint4(idxa, minidx, angle_type, 1);
						reaction_data.d_idx_angles[num_angle_c*reaction_data.angle_pitch + minidx] = make_uint4(idxa, idx, angle_type, 2);
						
						reaction_data.d_n_tag_angle[taga] = num_angle_a + 1;	
						reaction_data.d_n_tag_angle[tagi] = num_angle_b + 1;
						reaction_data.d_n_tag_angle[mintag] = num_angle_c + 1;

						reaction_data.d_n_idx_angle[idxa] = num_angle_a + 1;	
						reaction_data.d_n_idx_angle[idx] = num_angle_b + 1;
						reaction_data.d_n_idx_angle[minidx] = num_angle_c + 1;
						
						if(reaction_data.angle_exclusions)
							{
							unsigned int nexi = reaction_data.d_n_ex_tag[taga];
							unsigned int nexj = reaction_data.d_n_ex_tag[mintag];
							
							reaction_data.d_ex_list_tag[reaction_data.ex_list_indexer(taga, nexi)] = mintag;
							reaction_data.d_ex_list_tag[reaction_data.ex_list_indexer(mintag, nexj)] = taga;
							
							reaction_data.d_ex_list_idx[reaction_data.ex_list_indexer(idxa, nexi)] = minidx;
							reaction_data.d_ex_list_idx[reaction_data.ex_list_indexer(minidx, nexj)] = idxa;
							
							reaction_data.d_n_ex_tag[taga] = nexi + 1;
							reaction_data.d_n_ex_tag[mintag] = nexj + 1;
							
							reaction_data.d_n_ex_idx[idxa] = nexi + 1;
							reaction_data.d_n_ex_idx[minidx] = nexj + 1;
							filter_nlist(d_n_neigh, d_nlist, nli, idxa, minidx);
//							printf("thread angle a %d, %d, %d, %d, %d, %d\n", taga, tagi, mintag, idxa, idx, minidx);							
							}
						}
					for(unsigned int i = 0; i<numj; i++)
						{
						uint2 bondk = reaction_data.d_tag_bonds[i*reaction_data.bond_pitch + mintag];
						unsigned int tagk = bondk.x;
						unsigned int idxk = d_rtag[tagk];
						Real4 posk = GetPos(idxk);
						int typk = __real_as_int(posk.w);
						
						unsigned int angle_type = new_angle_type;
						if(reaction_data.angle_type_by_pair)
							angle_type = reaction_data.d_angle_type_table[typi * coeff_width * coeff_width + mintype_changed* coeff_width + typk];
						
						unsigned int num_angle_i = reaction_data.d_n_tag_angle[tagi];						
						unsigned int num_angle_j = reaction_data.d_n_tag_angle[mintag];
						unsigned int num_angle_k = reaction_data.d_n_tag_angle[tagk];
						
						reaction_data.d_tag_angles[num_angle_i*reaction_data.angle_pitch + tagi] = make_uint4(mintag, tagk, angle_type, 0);
						reaction_data.d_tag_angles[num_angle_j*reaction_data.angle_pitch + mintag] = make_uint4(tagi, tagk, angle_type, 1);
						reaction_data.d_tag_angles[num_angle_k*reaction_data.angle_pitch + tagk] = make_uint4(tagi, mintag, angle_type, 2);	

						reaction_data.d_idx_angles[num_angle_i*reaction_data.angle_pitch + idx] = make_uint4(minidx, idxk, angle_type, 0);
						reaction_data.d_idx_angles[num_angle_j*reaction_data.angle_pitch + minidx] = make_uint4(idx, idxk, angle_type, 1);
						reaction_data.d_idx_angles[num_angle_k*reaction_data.angle_pitch + idxk] = make_uint4(idx, minidx, angle_type, 2);
						
						reaction_data.d_n_tag_angle[tagi] = num_angle_i + 1;	
						reaction_data.d_n_tag_angle[mintag] = num_angle_j + 1;
						reaction_data.d_n_tag_angle[tagk] = num_angle_k + 1;

						reaction_data.d_n_idx_angle[idx] = num_angle_i + 1;	
						reaction_data.d_n_idx_angle[minidx] = num_angle_j + 1;
						reaction_data.d_n_idx_angle[idxk] = num_angle_k + 1;
						
						if(reaction_data.angle_exclusions)
							{
							unsigned int nexi = reaction_data.d_n_ex_tag[tagi];
							unsigned int nexk = reaction_data.d_n_ex_tag[tagk];
							
							reaction_data.d_ex_list_tag[reaction_data.ex_list_indexer(tagi, nexi)] = tagk;
							reaction_data.d_ex_list_tag[reaction_data.ex_list_indexer(tagk, nexk)] = tagi;
							
							reaction_data.d_ex_list_idx[reaction_data.ex_list_indexer(idx, nexi)] = idxk;
							reaction_data.d_ex_list_idx[reaction_data.ex_list_indexer(idxk, nexk)] = idx;
							
							reaction_data.d_n_ex_tag[tagi] = nexi + 1;
							reaction_data.d_n_ex_tag[tagk] = nexk + 1;
							
							reaction_data.d_n_ex_idx[idx] = nexi + 1;
							reaction_data.d_n_ex_idx[idxk] = nexk + 1;
							filter_nlist(d_n_neigh, d_nlist, nli, idx, idxk);
//							printf("thread angle k %d, %d, %d, %d, %d, %d\n", tagi, mintag, tagk, idx, minidx, idxk);
							}
						}
					}
				unsigned int bond_type = new_bond_type;
				if(reaction_data.bond_type_by_pair)
					bond_type = reaction_data.d_bond_type_table[typi * coeff_width + mintype_changed];

				reaction_data.d_tag_bonds[numi*reaction_data.bond_pitch + tagi] = make_uint2(mintag, bond_type);
				reaction_data.d_tag_bonds[numj*reaction_data.bond_pitch + mintag] = make_uint2(tagi, bond_type);
				reaction_data.d_n_tag_bond[tagi] = numi + 1;
				reaction_data.d_n_tag_bond[mintag] = numj + 1;
				  
				reaction_data.d_idx_bonds[numi*reaction_data.bond_pitch + idx] = make_uint2(minidx, bond_type);
				reaction_data.d_idx_bonds[numj*reaction_data.bond_pitch + minidx] = make_uint2(idx, bond_type);
				reaction_data.d_n_idx_bond[idx] = numi + 1;
				reaction_data.d_n_idx_bond[minidx] = numj + 1;				  
				reaction_data.d_bond_state[tagi] = make_uint2(1, mintag);

				if(reaction_data.bond_exclusions)
					{
					unsigned int nexi = reaction_data.d_n_ex_tag[tagi];
					unsigned int nexj = reaction_data.d_n_ex_tag[mintag];
					
					reaction_data.d_ex_list_tag[reaction_data.ex_list_indexer(tagi, nexi)] = mintag;
					reaction_data.d_ex_list_tag[reaction_data.ex_list_indexer(mintag, nexj)] = tagi;
					
					reaction_data.d_ex_list_idx[reaction_data.ex_list_indexer(idx, nexi)] = minidx;
					reaction_data.d_ex_list_idx[reaction_data.ex_list_indexer(minidx, nexj)] = idx;
					
					reaction_data.d_n_ex_tag[tagi] = nexi + 1;
					reaction_data.d_n_ex_tag[mintag] = nexj + 1;
					
					reaction_data.d_n_ex_idx[idx] = nexi + 1;
					reaction_data.d_n_ex_idx[minidx] = nexj + 1;
					filter_nlist(d_n_neigh, d_nlist, nli, idx, minidx);
//				printf("thread bond %d, %d, %d, %d, %d, %d\n", tagi, mintag, idx, minidx, nexi, nexj);
					}				
				atomicAdd(&ninit[3], (unsigned int)1);
				}
			}			
		}
	}
	
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
							int blocksize)
	{

    pos_tex.normalized = false;
    pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pos_tex, d_pos, sizeof(float4) * Np);
    if (error != cudaSuccess)
        return error;
    dim3 grid( (int)ceil((Real)h_ninit[0] / (Real)blocksize), 1, 1);
    dim3 threads(blocksize, 1, 1);	
	   gpu_compute_SGAP_DM_kernel<<< grid, threads, sizeof(Real3)*coeff_width*coeff_width>>>(d_pos,
																							d_tag,
																							d_rtag, 
																							box, 
																							d_n_neigh,
																							d_nlist, 
																							nli,				
																							reaction_data,
																							d_cris,
																							seed,
																							rcutsq,
																							coeff_width,
																							d_pr,
																							d_ninit,
																							d_init_group,
																							d_init,
																							d_maxcris,
																							new_bond_type,
																							new_angle_type,																							
																							period_R);

																																	

    return cudaSuccess;
	}
