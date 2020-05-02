

#include "Polymerization.cuh"
#include <assert.h>

texture<float4, 1, cudaReadModeElementType> pos_tex;

__global__ void gpu_compute_FRP_kernel(Real4* d_pos,
										unsigned int* d_tag,
										unsigned int* d_rtag,
										gpu_boxsize box, 
										const unsigned int *d_n_neigh,
										const unsigned int *d_nlist, 
										Index2D nli, 
										unsigned int* d_react_list,											
										unsigned int* d_n_tag_bond,
										uint2* d_tag_bonds,
										unsigned int* d_n_idx_bond,
										uint2* d_idx_bonds,
										unsigned int pitch,
										unsigned int *d_cris,
										unsigned int seed,
										Real rcutsq,
										unsigned int coeff_width,					
										Real2* d_pr,
										unsigned int* ninit,
										unsigned int* d_init_group,
										unsigned int* d_reaction_times,
										unsigned int* d_is_init,
										unsigned int new_bond_type,
										unsigned int* d_change_type)
	{
	extern __shared__ Real2 s_pr[];
	for (unsigned int cur_offset = 0; cur_offset < coeff_width*coeff_width; cur_offset += blockDim.x)
		{
		if (cur_offset + threadIdx.x < coeff_width*coeff_width)
			s_pr[cur_offset + threadIdx.x] = d_pr[cur_offset + threadIdx.x];
		}
	__syncthreads();
	
	unsigned int goup_idx = blockIdx.x * blockDim.x + threadIdx.x;	
	unsigned int num_init = ninit[0];
	if (goup_idx >= num_init)
		return;
	
	if (d_reaction_times[goup_idx] >= ninit[3])
		return;

    unsigned int tagi = d_init_group[goup_idx];
	unsigned int idx = d_rtag[tagi];
    unsigned int initi = 1;
	unsigned int n_neigh = d_n_neigh[idx];
	Real4 pos = GetPos(idx);
	int typi = __real_as_int(pos.w);
	
	for (int ni = 0; ni < n_neigh; ni++)
		d_react_list[ni*num_init + goup_idx] = d_nlist[nli(idx, ni)];		
			
	SaruGPU RNN(seed, tagi);
	unsigned int offset = (unsigned int)(RNN.f(0.0f, 1.0f)*float(n_neigh));	
	
	for (int ni = 0; ni < n_neigh; ni++)
		{
		unsigned int exch_ni = (unsigned int)(RNN.f<10>(0.0f, 1.0f)*float(n_neigh));
		unsigned int cur_neigh = d_react_list[ni*num_init + goup_idx];
		unsigned int exch_neigh = d_react_list[exch_ni*num_init + goup_idx];		
		d_react_list[ni*num_init + goup_idx] = exch_neigh;
		d_react_list[exch_ni*num_init + goup_idx] = cur_neigh;			
		}
	
    unsigned int cur_neigh = 0;
    unsigned int next_neigh = d_react_list[offset*num_init + goup_idx];
	
	for (int ni = 0; ni < n_neigh && initi ==1; ni++)
		{
		unsigned int neigh_idx = offset+ni+1;
		if(neigh_idx>=n_neigh)
			neigh_idx -= n_neigh;	
	
		cur_neigh = next_neigh;
		next_neigh = d_react_list[neigh_idx*num_init + goup_idx];
		
		Real4 neigh_pos = GetPos(cur_neigh);
		
		Real dx = pos.x - neigh_pos.x;
		Real dy = pos.y - neigh_pos.y;
		Real dz = pos.z - neigh_pos.z;
			
		dx -= box.lx * rint(dx * box.lxinv);
		dy -= box.ly * rint(dy * box.lyinv);
		dz -= box.lz * rint(dz * box.lzinv);
			

		Real rsq = dx*dx + dy*dy + dz*dz;
        unsigned int tagj = d_tag[cur_neigh];
		unsigned int is_init_j = d_is_init[tagj]; 
        unsigned int crisj = d_cris[tagj];		
		SaruGPU RNG(seed, tagi, tagj);
		
		if (rsq < rcutsq && crisj == 0 && is_init_j==0)
			{
			int typj = __real_as_int(neigh_pos.w);
			int typ_pair = typi * coeff_width + typj;	
			Real2 Pr = s_pr[typ_pair];
			Real ran = Rondom(0.0f,1.0f);
		
			if(ran < Pr.x)
			    {
			    unsigned int old = atomicMax(&d_cris[tagj],1);
                if(old ==0)
					{
					d_init_group[goup_idx] = tagj;
					d_is_init[tagj] = 1;
					d_cris[tagi] += 1;				  
					d_is_init[tagi] = 0;
					initi = 0;
					
					unsigned int type_changed = d_change_type[typj];
					if(typj!=type_changed)
						d_pos[cur_neigh].w = __int_as_real(type_changed);
					
					unsigned int numi = d_n_tag_bond[tagi];
					unsigned int numj = d_n_tag_bond[tagj];
					unsigned int type = new_bond_type;
					d_tag_bonds[numi*pitch + tagi] = make_uint2(tagj, type);
					d_tag_bonds[numj*pitch + tagj] = make_uint2(tagi, type);
					d_n_tag_bond[tagi] = numi + 1;
					d_n_tag_bond[tagj] = numj + 1;
			
					d_idx_bonds[numi*pitch + idx] = make_uint2(cur_neigh, type);
					d_idx_bonds[numj*pitch + cur_neigh] = make_uint2(idx, type);
					d_n_idx_bond[idx] = numi + 1;
					d_n_idx_bond[cur_neigh] = numj + 1;
					d_reaction_times[goup_idx] +=1;
//		printf("thread %d,%d, %d,%d,%d,%d,%d\n",goup_idx,ninit[0],pitch, d_bonds[numi*pitch + tagi].x, d_bonds[numj*pitch + tagj].x, numi, numj); 				  
					}
				}
		    }			
		}

	}
	
cudaError_t gpu_FRP_compute(Real4* d_pos,
							unsigned int* d_tag,
							unsigned int* d_rtag,
							const gpu_boxsize &box, 
							const unsigned int *d_n_neigh,
							const unsigned int *d_nlist,
							const Index2D& nli,
							unsigned int* d_react_list,								
							unsigned int* d_n_tag_bond,
							uint2* d_tag_bonds,
							unsigned int* d_n_idx_bond,
							uint2* d_idx_bonds,
							unsigned int pitch,
							unsigned int *d_cris,
							unsigned int seed,
							Real rcutsq,	
							unsigned int coeff_width,
							Real2* d_pr,
							unsigned int* h_ninit,
							unsigned int* d_ninit,
							unsigned int* d_init_group,
							unsigned int* d_reaction_times,
							unsigned int* d_is_init,
							unsigned int Np,
							unsigned int new_bond_type,	
							unsigned int* d_change_type,							
							int blocksize)
	{
    // setup the grid to run the kernel


    pos_tex.normalized = false;
    pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pos_tex, d_pos, sizeof(float4) * Np);
    if (error != cudaSuccess)
        return error;
    dim3 grid( (int)ceil((Real)h_ninit[0] / (Real)blocksize), 1, 1);
    dim3 threads(blocksize, 1, 1);	
	gpu_compute_FRP_kernel<<< grid, threads, sizeof(Real2)*coeff_width*coeff_width>>>(d_pos,
																						d_tag,
																						d_rtag, 
																						box, 
																						d_n_neigh,
																						d_nlist, 
																						nli,
																						d_react_list,																						
																						d_n_tag_bond,
																						d_tag_bonds,
																						d_n_idx_bond,
																						d_idx_bonds,
																						pitch,
																						d_cris,
																						seed,
																						rcutsq,
																						coeff_width,
																						d_pr,
																						d_ninit,
																						d_init_group,
																						d_reaction_times,
																						d_is_init,
																						new_bond_type,
																						d_change_type);	

																																	

    return cudaSuccess;
	}	

__global__ void gpu_compute_exchange_kernel(Real4* d_pos,
											unsigned int* d_tag,
											unsigned int* d_rtag, 
											gpu_boxsize box, 
											const unsigned int *d_n_neigh,
											const unsigned int *d_nlist, 
											Index2D nli, 
											unsigned int* d_react_list,												
											unsigned int* d_n_tag_bond,
											uint2* d_tag_bonds,
											unsigned int* d_n_idx_bond,
											uint2* d_idx_bonds,
											unsigned int pitch,
											unsigned int *d_cris,
											unsigned int seed,
											Real rcutsq,
											unsigned int coeff_width,					
											Real2* d_pr,
											unsigned int* ninit,
											unsigned int* d_init_group,
											unsigned int* d_is_init,
											unsigned int* d_maxcris,
											unsigned int* d_change_type)
	{
	extern __shared__ Real2 s_pr[];
	for (unsigned int cur_offset = 0; cur_offset < coeff_width*coeff_width*coeff_width; cur_offset += blockDim.x)
		{
		if (cur_offset + threadIdx.x < coeff_width*coeff_width*coeff_width)
			s_pr[cur_offset + threadIdx.x] = d_pr[cur_offset + threadIdx.x];
		}
	__syncthreads();
	
	unsigned int goup_idx = blockIdx.x * blockDim.x + threadIdx.x;	
	unsigned int num_init = ninit[0];
	if (goup_idx >= num_init)
		return;
    unsigned int tagi = d_init_group[goup_idx];
	unsigned int idx = d_rtag[tagi];
    unsigned int initi = 1;
	unsigned int n_neigh = d_n_neigh[idx];
	Real4 pos = GetPos(idx);
	int typi  = __real_as_int(pos.w);	
	unsigned int numi = d_n_tag_bond[tagi];
	
	for (int ni = 0; ni < n_neigh; ni++)
		d_react_list[ni*num_init + goup_idx] = d_nlist[nli(idx, ni)];		
			
	SaruGPU RNN(seed, tagi);
	unsigned int offset = (unsigned int)(RNN.f(0.0f, 1.0f)*float(n_neigh));	
	
	for (int ni = 0; ni < n_neigh; ni++)
		{
		unsigned int exch_ni = (unsigned int)(RNN.f<10>(0.0f, 1.0f)*float(n_neigh));
		unsigned int cur_neigh = d_react_list[ni*num_init + goup_idx];
		unsigned int exch_neigh = d_react_list[exch_ni*num_init + goup_idx];		
		d_react_list[ni*num_init + goup_idx] = exch_neigh;
		d_react_list[exch_ni*num_init + goup_idx] = cur_neigh;			
		}
	
    unsigned int cur_neigh = 0;
    unsigned int next_neigh = d_react_list[offset*num_init + goup_idx];
	
	for (int ni = 0; ni < n_neigh && initi ==1; ni++)
		{
		unsigned int neigh_idx = offset+ni+1;
		if(neigh_idx>=n_neigh)
			neigh_idx -= n_neigh;	
	
		cur_neigh = next_neigh;
		next_neigh = d_react_list[neigh_idx*num_init + goup_idx];	
		
		Real4 neigh_pos = GetPos(cur_neigh);
		int typj = __real_as_int(neigh_pos.w);
		Real dx = pos.x - neigh_pos.x;
		Real dy = pos.y - neigh_pos.y;
		Real dz = pos.z - neigh_pos.z;
			
		dx -= box.lx * rint(dx * box.lxinv);
		dy -= box.ly * rint(dy * box.lyinv);
		dz -= box.lz * rint(dz * box.lzinv);

		Real rsq = dx*dx + dy*dy + dz*dz;
        unsigned int tagj = d_tag[cur_neigh];
		unsigned int is_init_j = d_is_init[tagj]; 		
        unsigned int crisj = d_cris[tagj];		
		SaruGPU RNG(seed, tagi, tagj);
		unsigned int max_crisj = d_maxcris[typj];

		if (rsq < rcutsq && crisj < max_crisj && is_init_j==0)
			{
			Real Possib =0.0f;
			Real EffPossib=0.0f;
			bool bonded = false; 
			for( unsigned int b =0; b<numi; b++)
				{
				unsigned int temp_tagj = d_tag_bonds[b*pitch + tagi].x;
				if(temp_tagj==tagj)
					bonded=true;
				unsigned int temp_idxj = d_rtag[temp_tagj];
				unsigned int cris_tempj = d_cris[temp_tagj];				
				Real4 temp_posj = GetPos(temp_idxj);
				int temp_typj = __real_as_int(temp_posj.w);
				int typ_pair = typj * coeff_width* coeff_width + typi * coeff_width + temp_typj;	
				Real2 Pr = s_pr[typ_pair];
				Possib += Pr.x;
				if (cris_tempj>0)
					EffPossib += Pr.x;
				}
				
			Real ran = Rondom(0.0f,1.0f);
			if(ran < Possib&&EffPossib>0.0f&&!bonded)
				{
				unsigned int old_tagj = atomicMax(&d_cris[tagj],NO_INDEX);
				if(old_tagj < max_crisj)
					{
					bool success =false;
					for( unsigned int b =0; b<numi&&initi==1; b++)
						{
						unsigned int temp_tagj = d_tag_bonds[b*pitch + tagi].x;
						unsigned int temp_bond_type = d_tag_bonds[b*pitch + tagi].y;
						unsigned int temp_idxj = d_rtag[temp_tagj];					  
						Real4 temp_posj = GetPos(temp_idxj);
						int temp_typj = __real_as_int(temp_posj.w);
						int typ_pair = typj * coeff_width* coeff_width + typi * coeff_width + temp_typj;	
						Real2 Pr = s_pr[typ_pair];
						Real ranbf = ran;
						ran -= Pr.x;
						unsigned int cris_tempj = d_cris[temp_tagj];
						if(ran<=0.0f&&ranbf>0.0f&&cris_tempj>0&&cris_tempj!=NO_INDEX)
							{
							unsigned int old_temp_tagj = atomicMax(&d_cris[temp_tagj],NO_INDEX);
							if(old_temp_tagj!=NO_INDEX&&old_temp_tagj>0)
								{
//			printf("thread %d,%d,%d,%d,%d,%d\n",tagi,tagj,temp_tagj,typi,typj,temp_typj); 								
								d_tag_bonds[b*pitch + tagi].x = tagj;
								d_idx_bonds[b*pitch + idx].x = cur_neigh;
								unsigned int temp_numj = d_n_tag_bond[temp_tagj];
								unsigned int count =0;
								for(unsigned int j =0; j<temp_numj;j++)
									{
									uint2 temp_tag_bond = d_tag_bonds[j*pitch + temp_tagj];
									uint2 temp_idx_bond = d_idx_bonds[j*pitch + temp_idxj];
									if(temp_tag_bond.x!=tagi)
										{
										d_tag_bonds[count*pitch + temp_tagj] = temp_tag_bond;
										d_idx_bonds[count*pitch + temp_idxj] = temp_idx_bond;
										count +=1;
										}
									}
								d_n_tag_bond[temp_tagj] = count;
								d_n_idx_bond[temp_idxj] = count;
								d_cris[temp_tagj] =old_temp_tagj-1;
								initi = 0;
								unsigned int numj = d_n_tag_bond[tagj];
								d_tag_bonds[numj*pitch + tagj] = make_uint2(tagi, temp_bond_type);
								d_idx_bonds[numj*pitch + cur_neigh] = make_uint2(idx, temp_bond_type);
								d_n_tag_bond[tagj] = numj + 1;
								d_n_idx_bond[cur_neigh] = numj + 1;
								success = true;
								}
							else if(old_temp_tagj==0)
								d_cris[temp_tagj]=0;
							} 
						}
					if(success)
						d_cris[tagj] = old_tagj+1;
					else
						d_cris[tagj] = old_tagj;
					}
				else if(old_tagj !=NO_INDEX)
					{
					d_cris[tagj] = old_tagj;
					}
				}
		    }			
		}

	}

	
cudaError_t gpu_exchange_compute(Real4* d_pos,
								unsigned int* d_tag,
								unsigned int* d_rtag, 
								const gpu_boxsize &box, 
								const unsigned int *d_n_neigh,
								const unsigned int *d_nlist,
								const Index2D& nli,
								unsigned int* d_react_list,									
								unsigned int* d_n_tag_bond,
								uint2* d_tag_bonds,
								unsigned int* d_n_idx_bond,
								uint2* d_idx_bonds,
								unsigned int pitch,
								unsigned int *d_cris,
								unsigned int seed,
								Real rcutsq,	
								unsigned int coeff_width,
								Real2* d_pr,
								unsigned int* h_ninit,
								unsigned int* d_ninit,
								unsigned int* d_init_group,
								unsigned int* d_is_init,
								unsigned int* d_maxcris,
								unsigned int Np,
								unsigned int* d_change_type,								
								int blocksize)
	{
	
    pos_tex.normalized = false;
    pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pos_tex, d_pos, sizeof(float4) * Np);
    if (error != cudaSuccess)
        return error;
    dim3 grid( (int)ceil((Real)h_ninit[0] / (Real)blocksize), 1, 1);
    dim3 threads(blocksize, 1, 1);	
	gpu_compute_exchange_kernel<<< grid, threads, sizeof(Real2)*coeff_width*coeff_width*coeff_width>>>(d_pos,
																							d_tag,
																							d_rtag, 
																							box, 
																							d_n_neigh,
																							d_nlist, 
																							nli,
																							d_react_list,																							
																							d_n_tag_bond,
																							d_tag_bonds,
																							d_n_idx_bond,
																							d_idx_bonds,
																							pitch,
																							d_cris,
																							seed,
																							rcutsq,
																							coeff_width,
																							d_pr,
																							d_ninit,
																							d_init_group,
																							d_is_init,
																							d_maxcris,
																							d_change_type);	
	
																																	

    return cudaSuccess;
	}

__global__ void gpu_compute_SGAP_kernel(Real4* d_pos,
								unsigned int* d_tag,
								unsigned int* d_rtag,  
								gpu_boxsize box, 
								const unsigned int *d_n_neigh,
								const unsigned int *d_nlist, 
								Index2D nli, 
								unsigned int* d_react_list,								
								unsigned int* d_n_tag_bond,
								uint2* d_tag_bonds,
								unsigned int* d_n_idx_bond,
								uint2* d_idx_bonds,
								unsigned int pitch,
								unsigned int *d_cris,
								unsigned int seed,
								Real rcutsq,
								unsigned int coeff_width,
								Real2* d_pr,					
								unsigned int* ninit,
								unsigned int* d_init_group,
								unsigned int* d_is_init,					
								unsigned int* d_maxcris,
								unsigned int new_bond_type,
								unsigned int* d_change_type)
	{
	extern __shared__ Real2 s_pr[];
	for (unsigned int cur_offset = 0; cur_offset < coeff_width*coeff_width; cur_offset += blockDim.x)
		{
		if (cur_offset + threadIdx.x < coeff_width*coeff_width)
			s_pr[cur_offset + threadIdx.x] = d_pr[cur_offset + threadIdx.x];
		}
	__syncthreads();
	
	unsigned int goup_idx = blockIdx.x * blockDim.x + threadIdx.x;	
	unsigned int num_init = ninit[0];
	if (goup_idx >= num_init)
		return;
    unsigned int tagi = d_init_group[goup_idx];
	unsigned int idx = d_rtag[tagi];
    unsigned int initi = d_is_init[tagi];
	unsigned int numi = d_n_tag_bond[tagi];	
	unsigned int n_neigh = d_n_neigh[idx];
	Real4 pos = GetPos(idx);
	unsigned int typi = __real_as_int(pos.w);
	
	unsigned int max_crisi = d_maxcris[typi];
    unsigned int crisi = d_cris[tagi];
	if(crisi>=max_crisi||initi==0)
		{
		return;
		}

	for (int ni = 0; ni < n_neigh; ni++)
		d_react_list[ni*num_init + goup_idx] = d_nlist[nli(idx, ni)];		
			
	SaruGPU RNN(seed, tagi);
	unsigned int offset = (unsigned int)(RNN.f(0.0f, 1.0f)*float(n_neigh));	
	
	for (int ni = 0; ni < n_neigh; ni++)
		{
		unsigned int exch_ni = (unsigned int)(RNN.f<10>(0.0f, 1.0f)*float(n_neigh));
		unsigned int cur_neigh = d_react_list[ni*num_init + goup_idx];
		unsigned int exch_neigh = d_react_list[exch_ni*num_init + goup_idx];		
		d_react_list[ni*num_init + goup_idx] = exch_neigh;
		d_react_list[exch_ni*num_init + goup_idx] = cur_neigh;			
		}
	
    unsigned int cur_neigh = 0;
    unsigned int next_neigh = d_react_list[offset*num_init + goup_idx];
	
	for (int ni = 0; ni < n_neigh && initi ==1; ni++)
		{
		unsigned int neigh_idx = offset+ni+1;
		if(neigh_idx>=n_neigh)
			neigh_idx -= n_neigh;	
	
		cur_neigh = next_neigh;
		next_neigh = d_react_list[neigh_idx*num_init + goup_idx];
		
		Real4 neigh_pos = GetPos(cur_neigh);
		
		Real dx = pos.x - neigh_pos.x;
		Real dy = pos.y - neigh_pos.y;
		Real dz = pos.z - neigh_pos.z;
		unsigned int typj = __real_as_int(neigh_pos.w);
		dx -= box.lx * rint(dx * box.lxinv);
		dy -= box.ly * rint(dy * box.lyinv);
		dz -= box.lz * rint(dz * box.lzinv);

		Real rsq = dx*dx + dy*dy + dz*dz;
        unsigned int tagj = d_tag[cur_neigh];
        unsigned int crisj = d_cris[tagj];
		unsigned int max_crisj = d_maxcris[typj];
		unsigned int is_init_j = d_is_init[tagj];	
		SaruGPU RNG(seed, tagi, tagj);
		
		if (rsq < rcutsq && crisj < max_crisj && is_init_j==0 )
			{
			bool bonded = false; 
			for( unsigned int b =0; b<numi; b++)
				{
				unsigned int temp_tagj = d_tag_bonds[b*pitch + tagi].x;
				if(temp_tagj==tagj)
					bonded=true;
				}			
			int typ_pair = typi* coeff_width + typj;	
			Real2 Pr = s_pr[typ_pair];
		    Real ran = Rondom(0.0f,1.0f);
            Real factor = powf(Pr.y, crisi);
			if(ran < Pr.x*factor && !bonded)
				{
				unsigned int old = atomicMax(&d_cris[tagj],crisj+1);
				if(old == crisj)
					{	
					initi = 0;
					unsigned int numj = d_n_tag_bond[tagj];
					unsigned int type = new_bond_type;
					d_cris[tagi] +=1;
					
					unsigned int type_changedi = d_change_type[typi];
					if(typi!=type_changedi)
						d_pos[idx].w = __int_as_real(type_changedi);

					unsigned int type_changedj = d_change_type[typj];
					if(typj!=type_changedj)
						d_pos[cur_neigh].w = __int_as_real(type_changedj);					

					d_tag_bonds[numi*pitch + tagi] = make_uint2(tagj, type);
					d_tag_bonds[numj*pitch + tagj] = make_uint2(tagi, type);
					d_n_tag_bond[tagi] = numi + 1;
					d_n_tag_bond[tagj] = numj + 1;					

					d_idx_bonds[numi*pitch + idx] = make_uint2(cur_neigh, type);
					d_idx_bonds[numj*pitch + cur_neigh] = make_uint2(idx, type);
					d_n_idx_bond[idx] = numi + 1;
					d_n_idx_bond[cur_neigh] = numj + 1; 	
					}
				}
		    }			
		}
	}
	
cudaError_t gpu_SGAP_compute(Real4* d_pos,
							unsigned int* d_tag,
							unsigned int* d_rtag,
							const gpu_boxsize &box, 
							const unsigned int *d_n_neigh,
							const unsigned int *d_nlist,
							const Index2D& nli,
							unsigned int* d_react_list,								
							unsigned int* d_n_tag_bond,
							uint2* d_tag_bonds,
							unsigned int* d_n_idx_bond,
							uint2* d_idx_bonds,
							unsigned int pitch,
							unsigned int *d_cris,
							unsigned int seed,
							Real rcutsq,	
							unsigned int coeff_width,
							Real2* d_pr,
							unsigned int* h_ninit,
							unsigned int* d_ninit,
							unsigned int* d_init_group,
							unsigned int* d_is_init,					
							unsigned int* d_maxcris,
							unsigned int Np,
							unsigned int new_bond_type,
							unsigned int* d_change_type,							
							int blocksize)
	{
    pos_tex.normalized = false;
    pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pos_tex, d_pos, sizeof(float4) * Np);
    if (error != cudaSuccess)
        return error;
    dim3 grid( (int)ceil((Real)h_ninit[0] / (Real)blocksize), 1, 1);
    dim3 threads(blocksize, 1, 1);	
	   gpu_compute_SGAP_kernel<<< grid, threads, sizeof(Real2)*coeff_width*coeff_width>>>(d_pos,
																							d_tag,
																							d_rtag, 
																							box, 
																							d_n_neigh,
																							d_nlist, 
																							nli,
																							d_react_list,																							
																							d_n_tag_bond,
																							d_tag_bonds,
																							d_n_idx_bond,
																							d_idx_bonds,
																							pitch, 
																							d_cris,
																							seed,
																							rcutsq,
																							coeff_width,
																							d_pr,
																							d_ninit,
																							d_init_group,
																							d_is_init,
																							d_maxcris,
																							new_bond_type,
																							d_change_type);

																																	

    return cudaSuccess;
	}	

__global__ void gpu_compute_insertion_kernel(Real4* d_pos,
											unsigned int* d_tag,
											unsigned int* d_rtag, 
											gpu_boxsize box, 
											const unsigned int *d_n_neigh,
											const unsigned int *d_nlist, 
											Index2D nli, 
											unsigned int* d_react_list,												
											unsigned int* d_n_tag_bond,
											uint2* d_tag_bonds,
											unsigned int* d_n_idx_bond,
											uint2* d_idx_bonds,
											unsigned int pitch,
											unsigned int *d_cris,
											unsigned int seed,
											Real rcutsq,
											unsigned int coeff_width,					
											Real2* d_pr,
											unsigned int* ninit,
											unsigned int* d_init_group,
											unsigned int* d_is_init,
											unsigned int* d_maxcris,
											unsigned int new_bond_type,
											unsigned int* d_change_type)
	{
	extern __shared__ Real2 s_pr[];
	for (unsigned int cur_offset = 0; cur_offset < coeff_width*coeff_width*coeff_width; cur_offset += blockDim.x)
		{
		if (cur_offset + threadIdx.x < coeff_width*coeff_width*coeff_width)
			s_pr[cur_offset + threadIdx.x] = d_pr[cur_offset + threadIdx.x];
		}
	__syncthreads();
	
	unsigned int goup_idx = blockIdx.x * blockDim.x + threadIdx.x;	
	unsigned int num_init = ninit[0];
	if (goup_idx >= num_init)
		return;
    unsigned int tagi = d_init_group[goup_idx];
	unsigned int idx = d_rtag[tagi];
    unsigned int initi = 1;
	unsigned int n_neigh = d_n_neigh[idx];
	Real4 pos = GetPos(idx);
	int typi  = __real_as_int(pos.w);	
	unsigned int numi = d_n_tag_bond[tagi];
	
	for (int ni = 0; ni < n_neigh; ni++)
		d_react_list[ni*num_init + goup_idx] = d_nlist[nli(idx, ni)];		
			
	SaruGPU RNN(seed, tagi);
	unsigned int offset = (unsigned int)(RNN.f(0.0f, 1.0f)*float(n_neigh));	
	
	for (int ni = 0; ni < n_neigh; ni++)
		{
		unsigned int exch_ni = (unsigned int)(RNN.f<10>(0.0f, 1.0f)*float(n_neigh));
		unsigned int cur_neigh = d_react_list[ni*num_init + goup_idx];
		unsigned int exch_neigh = d_react_list[exch_ni*num_init + goup_idx];		
		d_react_list[ni*num_init + goup_idx] = exch_neigh;
		d_react_list[exch_ni*num_init + goup_idx] = cur_neigh;			
		}
	
    unsigned int cur_neigh = 0;
    unsigned int next_neigh = d_react_list[offset*num_init + goup_idx];
	
	for (int ni = 0; ni < n_neigh && initi ==1; ni++)
		{
		unsigned int neigh_idx = offset+ni+1;
		if(neigh_idx>=n_neigh)
			neigh_idx -= n_neigh;	
	
		cur_neigh = next_neigh;
		next_neigh = d_react_list[neigh_idx*num_init + goup_idx];	
		
		Real4 neigh_pos = GetPos(cur_neigh);
		int typj = __real_as_int(neigh_pos.w);
		Real dx = pos.x - neigh_pos.x;
		Real dy = pos.y - neigh_pos.y;
		Real dz = pos.z - neigh_pos.z;
			
		dx -= box.lx * rint(dx * box.lxinv);
		dy -= box.ly * rint(dy * box.lyinv);
		dz -= box.lz * rint(dz * box.lzinv);

		Real rsq = dx*dx + dy*dy + dz*dz;
        unsigned int tagj = d_tag[cur_neigh];
		unsigned int is_init_j = d_is_init[tagj]; 		
        unsigned int crisj = d_cris[tagj];		
		SaruGPU RNG(seed, tagi, tagj);
		unsigned int max_crisj = d_maxcris[typj];

		if (rsq < rcutsq && crisj < max_crisj && is_init_j==0)
			{
			Real Possib =0.0f;
			Real EffPossib=0.0f;
			bool bonded = false; 
			for( unsigned int b =0; b<numi; b++)
				{
				unsigned int temp_tagj = d_tag_bonds[b*pitch + tagi].x;
				if(temp_tagj==tagj)
					bonded=true;
				unsigned int temp_idxj = d_rtag[temp_tagj];
				unsigned int cris_tempj = d_cris[temp_tagj];				
				Real4 temp_posj = GetPos(temp_idxj);
				int temp_typj = __real_as_int(temp_posj.w);
				int typ_pair = typj * coeff_width* coeff_width + typi * coeff_width + temp_typj;	
				Real2 Pr = s_pr[typ_pair];
				Possib += Pr.x;
				if (cris_tempj>0)
					EffPossib += Pr.x;
				}
				
			Real ran = Rondom(0.0f,1.0f);
			if(ran < Possib&&EffPossib>0.0f&&!bonded)
				{
				unsigned int old_tagj = atomicMax(&d_cris[tagj],NO_INDEX);
				if(old_tagj < max_crisj)
					{
					bool success =false;
					for( unsigned int b =0; b<numi&&initi==1; b++)
						{
						unsigned int temp_tagj = d_tag_bonds[b*pitch + tagi].x;
						unsigned int temp_bond_type = d_tag_bonds[b*pitch + tagi].y;
						unsigned int temp_idxj = d_rtag[temp_tagj];					  
						Real4 temp_posj = GetPos(temp_idxj);
						int temp_typj = __real_as_int(temp_posj.w);
						int typ_pair = typj * coeff_width* coeff_width + typi * coeff_width + temp_typj;	
						Real2 Pr = s_pr[typ_pair];
						Real ranbf = ran;
						ran -= Pr.x;
						unsigned int cris_tempj = d_cris[temp_tagj];
						if(ran<=0.0f&&ranbf>0.0f&&cris_tempj>0&&cris_tempj!=NO_INDEX)
							{
							unsigned int old_temp_tagj = atomicMax(&d_cris[temp_tagj],NO_INDEX);
							if(old_temp_tagj!=NO_INDEX&&old_temp_tagj>0)
								{
//			printf("thread %d,%d,%d,%d,%d,%d\n",tagi,tagj,temp_tagj,typi,typj,temp_typj); 								
								d_tag_bonds[b*pitch + tagi].x = tagj;
								d_idx_bonds[b*pitch + idx].x = cur_neigh;
								unsigned int temp_numj = d_n_tag_bond[temp_tagj];
								for(unsigned int j =0; j<temp_numj;j++)
									{
									uint2 temp_tag_bond = d_tag_bonds[j*pitch + temp_tagj];
									if(temp_tag_bond.x==tagi)
										{
										d_tag_bonds[j*pitch + temp_tagj] = make_uint2(tagj, new_bond_type);
										d_idx_bonds[j*pitch + temp_idxj] = make_uint2(cur_neigh, new_bond_type);
										}
									}
								d_cris[temp_tagj] =old_temp_tagj;
								initi = 0;
								unsigned int numj = d_n_tag_bond[tagj];
								d_tag_bonds[numj*pitch + tagj] = make_uint2(tagi, temp_bond_type);
								d_idx_bonds[numj*pitch + cur_neigh] = make_uint2(idx, temp_bond_type);
								
								d_tag_bonds[(numj+1)*pitch + tagj] = make_uint2(temp_tagj, new_bond_type);
								d_idx_bonds[(numj+1)*pitch + cur_neigh] = make_uint2(temp_idxj, new_bond_type);
								
								d_n_tag_bond[tagj] = numj + 2;
								d_n_idx_bond[cur_neigh] = numj + 2;
								success = true;
								}
							else if(old_temp_tagj==0)
								d_cris[temp_tagj]=0;
							} 
						}
					if(success)
						d_cris[tagj] = old_tagj+1;
					else
						d_cris[tagj] = old_tagj;
					}
				else if(old_tagj !=NO_INDEX)
					{
					d_cris[tagj] = old_tagj;
					}
				}
		    }			
		}

	}

	
cudaError_t gpu_insertion_compute(Real4* d_pos,
								unsigned int* d_tag,
								unsigned int* d_rtag, 
								const gpu_boxsize &box, 
								const unsigned int *d_n_neigh,
								const unsigned int *d_nlist,
								const Index2D& nli,
								unsigned int* d_react_list,									
								unsigned int* d_n_tag_bond,
								uint2* d_tag_bonds,
								unsigned int* d_n_idx_bond,
								uint2* d_idx_bonds,
								unsigned int pitch,
								unsigned int *d_cris,
								unsigned int seed,
								Real rcutsq,	
								unsigned int coeff_width,
								Real2* d_pr,
								unsigned int* h_ninit,
								unsigned int* d_ninit,
								unsigned int* d_init_group,
								unsigned int* d_is_init,
								unsigned int* d_maxcris,
								unsigned int Np,
								unsigned int new_bond_type,
								unsigned int* d_change_type,								
								int blocksize)
	{
	
    pos_tex.normalized = false;
    pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pos_tex, d_pos, sizeof(float4) * Np);
    if (error != cudaSuccess)
        return error;
    dim3 grid( (int)ceil((Real)h_ninit[0] / (Real)blocksize), 1, 1);
    dim3 threads(blocksize, 1, 1);	
	gpu_compute_insertion_kernel<<< grid, threads, sizeof(Real2)*coeff_width*coeff_width*coeff_width>>>(d_pos,
																							d_tag,
																							d_rtag, 
																							box, 
																							d_n_neigh,
																							d_nlist, 
																							nli,
																							d_react_list,																							
																							d_n_tag_bond,
																							d_tag_bonds,
																							d_n_idx_bond,
																							d_idx_bonds,
																							pitch,
																							d_cris,
																							seed,
																							rcutsq,
																							coeff_width,
																							d_pr,
																							d_ninit,
																							d_init_group,
																							d_is_init,
																							d_maxcris,
																							new_bond_type,
																							d_change_type);	
	
																																	

    return cudaSuccess;
	}
