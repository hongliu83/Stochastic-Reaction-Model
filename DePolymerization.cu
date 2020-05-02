#include "DePolymerization.cuh"
#include <assert.h>

__global__ void gpu_depolymerization_compute_kernel(Real4* d_pos,
													unsigned int* d_rtag,
													unsigned int *d_cris,
													gpu_boxsize box,
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
													unsigned int* d_change_type)
	{
    int tag = blockIdx.x * blockDim.x + threadIdx.x;
    if (tag >= Np)
        return;
		
	unsigned int idx = d_rtag[tag];        
    int n_bond = d_n_tag_bond[tag];
	
    Real4 pos = d_pos[idx];
	unsigned int typi = __real_as_int(pos.w);	
	unsigned int count =0;
    for (int bond_num = 0; bond_num < n_bond; bond_num++)
        {
        uint2 bond_tag = d_tag_bonds[bond_num*pitch + tag];
        uint2 bond_idx = d_idx_bonds[bond_num*pitch + idx];		

        int cur_bond_tag = bond_tag.x;
        int cur_bond_type = bond_tag.y;

        int cur_bond_idx = d_rtag[cur_bond_tag];
        Real4 neigh_pos =  d_pos[cur_bond_idx];

        Real dx = pos.x - neigh_pos.x;
        Real dy = pos.y - neigh_pos.y;
        Real dz = pos.z - neigh_pos.z;
        
        dx -= box.lx * rint(dx * box.lxinv);
        dy -= box.ly * rint(dy * box.lyinv);
        dz -= box.lz * rint(dz * box.lzinv);

        Real4 params0 = d_params[cur_bond_type];
		Real4 params1 = d_params[cur_bond_type+coeff_width];
		
        Real K = params0.x;
        Real r_0 = params0.y;
        Real b_0 = params0.z;
        Real energy0 = params0.w;
		
		Real Pr = params1.x;
		Real func = params1.y;
	    int func_id = int(func);
		
        Real rsq = dx*dx + dy*dy + dz*dz;
		bool disconnection = false;
		Real possi = Pr;
		
		Real energy = 0.0;
		if(func_id==0)
			{
			energy = energy0;
			}		
		else if(func_id==1)
			{
			energy = -0.5f * K * r_0*r_0*log(1.0 - rsq/(r_0*r_0));
			}
		else if(func_id==2)
			{
			Real r = sqrt(rsq);
			energy = 0.5f * K *(r - b_0)* (r - b_0);
			}
		if(energy<energy0)
			{
			possi = Pr*expf((energy-energy0)/T);
			}

		unsigned int seed2, seed3;
		if (tag < cur_bond_tag)
			{
			seed2=tag;
			seed3=cur_bond_tag;
			}
		else
			{
			seed2=cur_bond_tag;
			seed3=tag;
			}		
		SaruGPU RNG(seed, seed2, seed3);
		if(Rondom(0.0f,1.0f)<possi)
			{
			disconnection=true;
			
			unsigned int type_changedi = d_change_type[typi];
			if(typi!=type_changedi)
				d_pos[idx].w = __int_as_real(type_changedi);
			//printf("thread depoly %d, %d, %d, %d, %f, %f, %f, %f\n",tag, cur_bond_tag, func_id, seed-16361, energy, energy0, possi, sqrt(rsq));			
			}
		if(!disconnection)
			{
			d_tag_bonds[count*pitch + tag] = bond_tag; 
			d_idx_bonds[count*pitch + idx] = bond_idx; 
			count += 1;
			}
		else if(d_cris[tag]>0)
			{
			d_cris[tag] -= 1;
			}	
//printf("thread depoly %d, %d, %d, %d, %d, %f, %f, %f, %f\n",tag, cur_bond_tag, func_id, seed-16361, disconnection, energy, energy0, possi, sqrt(rsq));				
		}
	d_n_tag_bond[tag] = count;
	d_n_idx_bond[idx] = count;
	}
	
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
										int blocksize)
	{
    // setup the grid to run the kernel

    dim3 grid( (int)ceil((Real)Np / (Real)blocksize), 1, 1);
    dim3 threads(blocksize, 1, 1);	
	gpu_depolymerization_compute_kernel<<< grid, threads>>>(d_pos, 
															d_rtag,
															d_cris,
															box, 			
															d_n_tag_bond,
															d_tag_bonds,
															d_n_idx_bond,
															d_idx_bonds,
															pitch,
															d_params,
															T,
															seed,
															coeff_width,
															Np,
															d_change_type);	

    return cudaSuccess;
	}	
