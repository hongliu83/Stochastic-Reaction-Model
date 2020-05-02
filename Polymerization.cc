

#include "Polymerization.h"
#include<time.h> 
#include<stdlib.h> 

#include <boost/python.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
using namespace boost::filesystem;
using namespace boost::python;
using namespace std;


Polymerization::Polymerization(boost::shared_ptr<AllInfo> all_info,
				  boost::shared_ptr<NeighborList> nlist, 
				  Real r_cut, 
				  unsigned int seed) 
	: Chare(all_info), m_nlist(nlist), m_r_cut(r_cut), m_seed(seed)
	{
	initData();
	Statistic();
	}

Polymerization::Polymerization(boost::shared_ptr<AllInfo> all_info,
				  const std::string &type,
				  Real percent,
				  boost::shared_ptr<NeighborList> nlist, 
				  Real r_cut, 
				  unsigned int seed) 
	: Chare(all_info),m_nlist(nlist), m_r_cut(r_cut), m_seed(seed)
	{
	initData();
	creatInitor(type, percent);
	Statistic();
	}	

void Polymerization::initData()
	{
	m_all_info->initBondInfo();
	m_bond_info = m_all_info->getBondInfo();
	
	Real nlistRcut = m_nlist->getRcut();
// initiate random generator	
	srand((int)time(0));
	if (m_r_cut < 0.0 || m_r_cut > nlistRcut)
		{
		cerr << endl << "***Error! Negative r_cut or r_cut larger than nlist rcut" << endl << endl;
		throw runtime_error("Error Polymerization::initData");
		}	
	m_block_size = 192;	
    unsigned int nparticles = m_basic_info->getN();
	m_nkinds = m_basic_info->getNKinds();  
	m_pr = boost::shared_ptr<Array<Real2> >(new Array<Real2>(m_nkinds*m_nkinds*m_nkinds, location::host));
	m_maxcris = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>(m_nkinds, location::host));
	m_init_group = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>(nparticles, location::host));
	m_reaction_times = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>(nparticles, location::host));
	m_ninit = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>(4, location::host));
	m_react_list = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>());
	m_change_type = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>(m_nkinds, location::host));
	
	boost::shared_ptr<Array<unsigned int> > init = m_basic_info->getInit();
	if(init->isempty())
		init->resize(nparticles);
	boost::shared_ptr<Array<unsigned int> > cris = m_basic_info->getCris();
	if(cris->isempty())
		cris->resize(nparticles);
		
	Real2* h_pr = m_pr->getArray(location::host, access::readwrite);
	for(unsigned int i =0; i<m_nkinds*m_nkinds*m_nkinds;i++)
			h_pr[i].y =1.0;
			
	unsigned int* h_maxcris = m_maxcris->getArray(location::host, access::readwrite);
	for(unsigned int i =0; i<m_nkinds;i++)
		h_maxcris[i] = 1;	

	unsigned int* h_change_type = m_change_type->getArray(location::host, access::readwrite);
	for(unsigned int i =0; i<m_nkinds;i++)
		h_change_type[i] = i;	

	m_setVariantT = false;
	m_T = 1.0;
	m_new_bond_type = 0;		
    m_period = 1;
	m_max_added_bonds = 2;
	m_init_ex_point = false;
	m_min_dis_rule = false;
	m_func_rule = false;
	m_first_step = true;
	m_reaction_mode = FRP;
	m_set_max_cris = false;
	m_set_exch_pr = false;
	m_setPr = false;
	m_set_insert_pr = false;	
	m_mode_set = false;	
	m_ObjectName = "Polymerization";
	cout << "INFO : Polymerization object has been build up !" << endl; 
	}
	
Polymerization::~Polymerization()
	{
	}
	
void Polymerization::setChangeTypeInReaction(const std::string &name_origin, const std::string &name_new)
	{
	unsigned int* h_change_type = m_change_type->getArray(location::host, access::readwrite);
	unsigned int type_origin = m_basic_info->switchNameToIndex(name_origin);
	unsigned int type_new = m_basic_info->switchNameToIndex(name_new);	
	h_change_type[type_origin] = type_new;
	}	
	
void Polymerization::setNewBondType(const std::string &name)
	{
	m_new_bond_type= m_bond_info->switchNameToIndex(name);	
	}	

void Polymerization::creatInitor(const std::string &name, Real percent)
	{
	unsigned int type= m_basic_info -> switchNameToIndex(name);	
	unsigned int nparticles = m_basic_info->getN();		
	Real4* h_pos = m_basic_info->getPos()->getArray(location::host, access::read);
	unsigned int* h_rtag = m_basic_info->getRtag()->getArray(location::host, access::read);	
	unsigned int* h_init= m_basic_info->getInit()->getArray(location::host, access::write);

	unsigned int count =0;
	for(unsigned int tag=0;tag<nparticles;tag++)
		{
		unsigned int idx = h_rtag[tag];
		Realint fi;
		fi.f = h_pos[idx].w;			
		unsigned int typi = fi.i;
		if(typi == type)
			{
			int ran = rand();
			Real j=(Real)ran/(Real)RAND_MAX; 
			if(j<percent)
				{
				h_init[tag] = 1;
				count += 1;
				}
			}
		}
    cout<<"INFO : There are "<<count<<" initors randomly created in type "<<name<<"!"<<endl;
	}

void Polymerization::setFrpMode()
	{
	m_reaction_mode = FRP;
	m_mode_set = true;	
	}

void Polymerization::setSgapMode()
	{
	m_reaction_mode = SGAP;	
	m_mode_set = true;		
	}

void Polymerization::setExchMode()
	{
	m_reaction_mode = EXCH;		
	m_mode_set = true;		
	}
	
void Polymerization::setInsertionMode()
	{
	m_reaction_mode = INSERT;		
	m_mode_set = true;		
	}
	
void Polymerization::setMinDisReactRule(bool mindis_rule)
	{
	m_min_dis_rule = mindis_rule;
	if(m_min_dis_rule)
		m_func_rule=false;
	}

void Polymerization::setFuncReactRule(bool func_rule, Real K, Real r_0, Real b_0, Real epsilon0, Func function)
	{
	m_func_rule = func_rule;
	if(m_func_rule)
		m_min_dis_rule = false;

	Real offset=0;
	if(function == FENE)
		{
		if (b_0 >= r_0)
			{
			cerr << endl << "***Error! Trying to set b_0 = "<<b_0<<" greater than r_0 = "<<r_0<<"!"<<endl << endl;
			throw runtime_error("Polymerization::setParams argument error"); 
			}
		if (b_0<0)
			{
			cerr << endl << "***Error! Trying to set b_0 = "<<b_0<<" less than 0"<<"!"<<endl << endl;
			throw runtime_error("Polymerization::setParams argument error"); 
			}				
		offset = -0.5 * K * r_0*r_0*log(1.0 - (b_0*b_0)/(r_0*r_0));		
		offset += epsilon0;
		}
	else if(function == harmonic)
		{
		if (b_0<0)
			{
			cerr << endl << "***Error! Trying to set b_0 = "<<b_0<<" less than 0"<<"!"<<endl << endl;
			throw runtime_error("Polymerization::setParams argument error"); 
			}				
		offset = epsilon0;
		}

	m_func_params = ToReal4(K, r_0, b_0, offset);
	m_func_id = int(function);	
	}
	
void Polymerization::setExchangePr(const std::string &name1, const std::string &name2, const std::string &name3, Real exchange)
	{
	unsigned int typ1= m_basic_info -> switchNameToIndex(name1);
	unsigned int typ2= m_basic_info -> switchNameToIndex(name2);
	unsigned int typ3= m_basic_info -> switchNameToIndex(name3);
	
	if (typ1 >= m_nkinds || typ2 >= m_nkinds|| typ3 >= m_nkinds)
		{
		cerr << endl << "***Error! Trying to set exchange for an non existent type! " << typ1 << "," << typ2 << "," << typ2 << endl << endl;
		throw runtime_error("Polymerization::stExchangePr argument error");
		}
    if(exchange<0.0)
		{
		cerr << endl << "***Error! Trying to set exchange little than zero! " << exchange<< endl << endl;
		throw runtime_error("Polymerization::stExchangePr argument error");
		}
		
	Real2* h_pr = m_pr->getArray(location::host, access::readwrite);
	h_pr[typ1*m_nkinds*m_nkinds + typ2*m_nkinds +typ3].x = exchange;

	m_set_exch_pr=true;
	}
	
	
void Polymerization::setInsertionPr(const std::string &name1, const std::string &name2, const std::string &name3, Real insertion)
	{
	unsigned int typ1= m_basic_info -> switchNameToIndex(name1);
	unsigned int typ2= m_basic_info -> switchNameToIndex(name2);
	unsigned int typ3= m_basic_info -> switchNameToIndex(name3);
	
	if (typ1 >= m_nkinds || typ2 >= m_nkinds|| typ3 >= m_nkinds)
		{
		cerr << endl << "***Error! Trying to set insertion for an non existent type! " << typ1 << "," << typ2 << "," << typ2 << endl << endl;
		throw runtime_error("Polymerization::setInsertionPr argument error");
		}
    if(insertion<0.0)
		{
		cerr << endl << "***Error! Trying to set insertion pr little than zero! " << insertion<< endl << endl;
		throw runtime_error("Polymerization::setInsertionPr error");
		}
		
	Real2* h_pr = m_pr->getArray(location::host, access::readwrite);
	h_pr[typ1*m_nkinds*m_nkinds + typ2*m_nkinds +typ3].x = insertion;

	m_set_insert_pr=true;
	}
		
	
void Polymerization::setMaxCris(const std::string &name, unsigned int maxcris)
	{
	unsigned int typ= m_basic_info -> switchNameToIndex(name);
	
	if (typ >= m_nkinds)
		{
		cerr << endl << "***Error! Trying to set maximum cris for an non existent type ! " << name << endl << endl;
		throw runtime_error("Polymerization::setMaxCris argument error");
		}
	if (maxcris>20)
		{
		cerr << endl << "***Error! Trying to set maximum cris larger than the limited 20! " <<maxcris<< endl << endl;
		throw runtime_error("Polymerization::setMaxCris argument error");
		}
	unsigned int* h_maxcris = m_maxcris->getArray(location::host, access::readwrite);
	h_maxcris[typ] = maxcris;
	
	if(maxcris>m_max_added_bonds)
		m_max_added_bonds = maxcris;

	m_set_max_cris=true;
	}
	
void Polymerization::setPr(const std::string &name1, const std::string &name2, Real Pr)
	{
	
	unsigned int typ1= m_basic_info -> switchNameToIndex(name1);
	unsigned int typ2= m_basic_info -> switchNameToIndex(name2);

	if (typ1 >= m_nkinds || typ2 >= m_nkinds)
		{
		cerr << endl << "***Error! Trying to set pr for an non existent type! " << typ1 << "," << typ2 << endl << endl;
		throw runtime_error("Polymerization::setPr argument error");
		}
	Real2* h_pr = m_pr->getArray(location::host, access::readwrite);
	h_pr[typ1*m_nkinds + typ2].x = Pr;

	m_setPr = true;
	}
	
void Polymerization::setPr(Real Pr)
	{
	Real2* h_pr = m_pr->getArray(location::host, access::readwrite);
	for(unsigned int i =0; i<m_nkinds*m_nkinds*m_nkinds;i++)
			h_pr[i].x =Pr;

	m_setPr = true;
	}
	
void Polymerization::setPrFactor(const std::string &name1, const std::string &name2, Real Factor)
	{
	unsigned int typ1= m_basic_info -> switchNameToIndex(name1);
	unsigned int typ2= m_basic_info -> switchNameToIndex(name2);

	if (typ1 >= m_nkinds || typ2 >= m_nkinds)
		{
		cerr << endl << "***Error! Trying to set pr for an non existent type! " << typ1 << "," << typ2 << endl << endl;
		throw runtime_error("Polymerization::setPrFactor argument error");
		}
	Real2* h_pr = m_pr->getArray(location::host, access::readwrite);
	h_pr[typ1*m_nkinds + typ2].y = Factor;
	}
	
void Polymerization::setPrFactor(Real Factor)
	{
	Real2* h_pr = m_pr->getArray(location::host, access::readwrite);
	for(unsigned int i =0; i<m_nkinds*m_nkinds*m_nkinds;i++)
			h_pr[i].y =Factor;
	}	

void Polymerization::initExPoint()
	{
	m_init_ex_point = true;
	}
	
void Polymerization::initiateExchPoint()
	{
	if (!m_set_exch_pr)
		{
		cerr << endl << "***Error! Please first set ligand exchange probability! "<< endl << endl;
		throw runtime_error("Polymerization::initExPoint error");
		}	
	unsigned int nparticles = m_basic_info->getN();
	unsigned int* h_init_group = m_init_group->getArray(location::host, access::readwrite);
	unsigned int* h_is_init = m_basic_info->getInit()->getArray(location::host, access::readwrite);		
	unsigned int* h_ninit = m_ninit->getArray(location::host, access::readwrite);
	Real2* h_pr = m_pr->getArray(location::host, access::read);

    unsigned int* h_n_bond = m_bond_info->getBondNumTagArray()->getArray(location::host, access::read); 
	uint2* h_bonds = m_bond_info->getBondTagArray()->getArray(location::host, access::read);
	unsigned int pitch= m_bond_info->getBondTagArray()->getPitch();
	
	Real4* h_pos = m_basic_info->getPos()->getArray(location::host, access::read);	
	unsigned int* h_rtag = m_basic_info->getRtag()->getArray(location::host, access::read); 

	unsigned int new_ex_point =0;
	unsigned int count = h_ninit[0];
	for (unsigned int taga = 0; taga < nparticles; taga++)
		{
		unsigned int nbonds = h_n_bond[taga];
		unsigned int idxa= h_rtag[taga];
		Realint fi;
		fi.f = h_pos[idxa].w;			
		unsigned int typi = fi.i;
		bool anchor = false;
		Real Possi=0.0;
		bool neighbor_init = false;
		for(unsigned int j =0; j< nbonds; j++)
			{
			uint2 bond = h_bonds[j*pitch + taga];
			unsigned int tagb = bond.x;
			neighbor_init = neighbor_init||h_is_init[tagb];
			unsigned int idxb= h_rtag[tagb];				
			Realint fj;
			fj.f = h_pos[idxb].w;			
			unsigned int typj = fj.i;
			for(unsigned int typ1=0; typ1<m_nkinds; typ1++)
				{
				unsigned int pair = typ1*m_nkinds*m_nkinds + typi*m_nkinds + typj;
				Possi += h_pr[pair].x;
				}
			}
		if(Possi>0&&!neighbor_init)
			anchor = true;			
		if(anchor)
			{
//			cout<<taga<<" "<<typi<<endl;
			bool exist =false; 
			for(unsigned int i=0; i<new_ex_point+count; i++)
				{
				if(h_init_group[i]==taga)
					exist = true;
				}
			if(!exist)
				{
				h_init_group[new_ex_point+count]=taga;
				h_is_init[taga]= 1;				
				new_ex_point +=1;
				}
			}	
		}
	h_ninit[0] = new_ex_point+count;
	cout<<"INFO : There are "<<new_ex_point<<" exchange active points newly added!"<<endl;
	m_init_ex_point = false;	
	}

void Polymerization::Statistic()
	{
	unsigned int Numinit = 0;
	unsigned int Numcris = 0;
	
	unsigned int Np = m_basic_info->getN();
	unsigned int* h_init_group = m_init_group->getArray(location::host, access::write);
	unsigned int* h_init= m_basic_info->getInit()->getArray(location::host, access::read);
	unsigned int* h_cris= m_basic_info->getCris()->getArray(location::host, access::read);	
	unsigned int* h_ninit = m_ninit->getArray(location::host, access::readwrite);
	
	for (unsigned int i=0; i < Np; i++)
		{
		unsigned int value = h_init[i];	
		if (value == 1)
			{
			h_init_group[Numinit] = i;			
			Numinit +=1;
			}
		}
	
	for (unsigned int i=0; i < Np; i++)
		{
		unsigned int value = h_cris[i];	
		if (value == 0)
			Numcris +=1;	
		}

	h_ninit[0] = Numinit;
	h_ninit[3] = 0xffffffff;
	cout << "INFO : Polymerization statistics, " <<Numinit<<" initiators"<< endl;				
	cout << "INFO : Polymerization statistics, " <<Numcris<<" free monomers"<< endl;
	if (Numinit==0)
		{
		cerr << endl << "***Error! No initiators "<< endl << endl;
		throw runtime_error("Polymerization::Statistic error");	
		}
	}
	
void Polymerization::checkLiEx()
	{
	unsigned int nparticles = m_basic_info->getN();
	unsigned int* h_is_init = m_basic_info->getInit()->getArray(location::host, access::readwrite);	
	Real2* h_pr = m_pr->getArray(location::host, access::read);

    unsigned int* h_n_bond = m_bond_info->getBondNumTagArray()->getArray(location::host, access::read); 
	uint2* h_bonds = m_bond_info->getBondTagArray()->getArray(location::host, access::read);
	unsigned int pitch= m_bond_info->getBondTagArray()->getPitch();
	
	Real4* h_pos = m_basic_info->getPos()->getArray(location::host, access::read);	
	unsigned int* h_rtag = m_basic_info->getRtag()->getArray(location::host, access::read); 	
	
	for (unsigned int taga = 0; taga < nparticles; taga++)
		{
		if(h_is_init[taga]==1)
			{
			unsigned int idxa= h_rtag[taga];
			Realint fi;
			fi.f = h_pos[idxa].w;			
			unsigned int typi = fi.i;
			
			unsigned int nbonds = h_n_bond[taga];			
			for(unsigned int j =0; j< nbonds; j++)
				{
				uint2 bond = h_bonds[j*pitch + taga];
				unsigned int tagb = bond.x;
				if(h_is_init[tagb]==1)
					{
					unsigned int idxb= h_rtag[tagb];				
					Realint fj;
					fj.f = h_pos[idxb].w;			
					unsigned int typj = fj.i;					
					Real Possi1=0.0;
					Real Possi2=0.0;					
					for(unsigned int typ1=0; typ1<m_nkinds; typ1++)
						{
						unsigned int pair1 = typ1*m_nkinds*m_nkinds + typi*m_nkinds + typj;
						unsigned int pair2 = typ1*m_nkinds*m_nkinds + typj*m_nkinds + typi;						
						Possi1 += h_pr[pair1].x;
						Possi2 += h_pr[pair2].x;						
						}						
					if(Possi1>0&&Possi2>0)
						{
						cerr << endl << "***Error! Exchange  or Insertion mode check, two bonded active points, "<<taga<<" "<<tagb<< endl << endl;
						throw runtime_error("Polymerization::checkLiEx error");
						}
					}
				}
			}
		}

	}
	
void Polymerization::computeChare(unsigned int timestep)
	{
    if (m_first_step)
        {;
        m_bond_info->growBondArrayHeight(m_max_added_bonds);
	    CHECK_CUDA_ERROR();
		if(!m_mode_set)
			{
			if(m_set_exch_pr)
				m_reaction_mode = EXCH;
			else if (m_set_max_cris&&m_setPr)
				m_reaction_mode = SGAP;
			else if(m_setPr)
				m_reaction_mode = FRP;
			else if(m_set_insert_pr)
				m_reaction_mode = INSERT;				
			else
				{
				cerr << endl << "***Error! Polymerization can not parse the reaction mode by the parameter set!" << endl << endl;
				throw runtime_error("Error computeChare in Polymerization");
				}
			m_mode_set = true;
			}
        m_first_step = false;
        }
		
	if(m_init_ex_point)
		initiateExchPoint();
		
	m_nlist->compute(timestep);
	   
	if(m_mode_set)
		{
		if(m_reaction_mode == FRP)
			{
			cout<<"INFO : Free Radical Polymerization Mode!"<<endl;
			}
		else if(m_reaction_mode == SGAP)
			{			
			cout<<"INFO : Step Growth Addition Polymerization mode!"<<endl;
			}
		else if(m_reaction_mode == EXCH)
			{
			checkLiEx();
			cout<<"INFO : Exchange Reaction Mode!"<<endl;
			}	
		else if(m_reaction_mode == INSERT)
			{
			checkLiEx();
			cout<<"INFO : Insertion Reaction Mode!"<<endl;
			}				
		else
			{
			cerr << endl << "***Error! Polymerization have not chosen a mode" << endl << endl;
			throw runtime_error("Error computeChare in Polymerization");		
			}
		m_mode_set=false;
		}

		
    unsigned int* d_n_tag_bond = m_bond_info->getBondNumTagArray()->getArray(location::device, access::readwrite); 
	uint2* d_tag_bonds = m_bond_info->getBondTagArray()->getArray(location::device, access::readwrite);
    unsigned int* d_n_idx_bond = m_bond_info->getBondNumIdxArray()->getArray(location::device, access::readwrite); 
	uint2* d_idx_bonds = m_bond_info->getBondIdxArray()->getArray(location::device, access::readwrite);
	unsigned int pitch= m_bond_info->getBondTagArray()->getPitch();

	
	Real4* d_pos = m_basic_info->getPos()->getArray(location::device, access::read);
	unsigned int* d_tag = m_basic_info->getTag()->getArray(location::device, access::read);	
	unsigned int* d_rtag = m_basic_info->getRtag()->getArray(location::device, access::read); 
	unsigned int Np = m_basic_info->getN();
	 
	const gpu_boxsize& box = m_basic_info->getBoxGPU();
	Real2* d_pr = m_pr->getArray(location::device, access::read);
	unsigned int* d_init_group = m_init_group->getArray(location::device, access::readwrite);
	unsigned int* d_cris = m_basic_info->getCris()->getArray(location::device, access::readwrite);
	unsigned int* d_is_init = m_basic_info->getInit()->getArray(location::device, access::readwrite);
	unsigned int* h_ninit = m_ninit->getArray(location::host, access::read);
	
	unsigned int* d_ninit = m_ninit->getArray(location::device, access::read);	 
	unsigned int* d_maxcris = m_maxcris->getArray(location::device, access::readwrite);
	unsigned int* d_reaction_times = m_reaction_times->getArray(location::device, access::readwrite);
	unsigned int* d_change_type = m_change_type->getArray(location::device, access::read);
	
	unsigned int nmax = m_nlist->getNListIndexer().getH();
	if(m_react_list->getHeight()!=nmax+1)
		m_react_list->resize(h_ninit[0], nmax+1);
	unsigned int* d_react_list = m_react_list->getArray(location::device, access::readwrite);
	 // unsigned int* h_init_group = m_init_group->getArray(location::host, access::readwrite);	 
	// cout<<"pitch"<<pitch<<endl;
		// for(unsigned int i =0; i<h_ninit[0];i++)
			// cout<<h_init_group[i]<<endl;
	 
	if(m_reaction_mode == FRP)
		{
		if(m_min_dis_rule)
			{
		   gpu_FRP_Dis_compute(d_pos,
						  d_tag,
						  d_rtag,
                          box,
						  m_nlist-> getGpuNNeigh(),
						  m_nlist-> getGpuNList(),
						  m_nlist-> getNListIndexer(),						  
						  d_n_tag_bond,
						  d_tag_bonds,
						  d_n_idx_bond,
						  d_idx_bonds,
						  pitch,
						  d_cris,
						  m_seed+timestep,
						  m_r_cut*m_r_cut,
						  m_nkinds,
						  d_pr,
						  h_ninit,
						  d_ninit,
						  d_init_group,
						  d_reaction_times,
						  d_is_init,
						  Np,
						  m_new_bond_type,
						  d_change_type,
						  m_block_size);
			}
		else if (m_func_rule)
			{
			if(m_setVariantT)
				m_T = (Real) m_vT->getValue(timestep);
		    gpu_FRP_Func_compute(d_pos,
						  d_tag,
						  d_rtag,
                          box,
						  m_nlist-> getGpuNNeigh(),
						  m_nlist-> getGpuNList(),
						  m_nlist-> getNListIndexer(),
						  d_react_list,							  
						  d_n_tag_bond,
						  d_tag_bonds,
						  d_n_idx_bond,
						  d_idx_bonds,
						  pitch,
						  d_cris,
						  m_seed+timestep,
						  m_r_cut*m_r_cut,
						  m_nkinds,
						  d_pr,
						  h_ninit,
						  d_ninit,
						  d_init_group,
						  d_reaction_times,
						  d_is_init,
						  Np,
						  m_new_bond_type,
						  m_func_params,
						  m_func_id,
						  m_T,
						  d_change_type,						  
						  m_block_size);
			}
		else
			{
		   gpu_FRP_compute(d_pos,
						  d_tag,
						  d_rtag,
                          box,
						  m_nlist-> getGpuNNeigh(),
						  m_nlist-> getGpuNList(),
						  m_nlist-> getNListIndexer(),	
						  d_react_list,							  
						  d_n_tag_bond,
						  d_tag_bonds,
						  d_n_idx_bond,
						  d_idx_bonds,
						  pitch,
						  d_cris,
						  m_seed+timestep,
						  m_r_cut*m_r_cut,
						  m_nkinds,
						  d_pr,
						  h_ninit,
						  d_ninit,
						  d_init_group,
						  d_reaction_times,
						  d_is_init,
						  Np,
						  m_new_bond_type,
						  d_change_type,						  
						  m_block_size);
			}
		}
	else if(m_reaction_mode == EXCH)
		{
		if(m_min_dis_rule)
			{
			gpu_exchange_Dis_compute(d_pos,
							  d_tag,
							  d_rtag,
							  box,
							  m_nlist-> getGpuNNeigh(),
							  m_nlist-> getGpuNList(),
							  m_nlist-> getNListIndexer(),						  
							  d_n_tag_bond,
							  d_tag_bonds,
							  d_n_idx_bond,
							  d_idx_bonds,
							  pitch,
							  d_cris,
							  m_seed+timestep,
							  m_r_cut*m_r_cut,
							  m_nkinds,
							  d_pr,
							  h_ninit,
							  d_ninit,
							  d_init_group,
							  d_is_init,
							  d_maxcris,
							  Np,
							  d_change_type,
							  m_block_size);
			
			}
		else if (m_func_rule)
			{
			if(m_setVariantT)
				m_T = (Real) m_vT->getValue(timestep);
			gpu_exchange_Func_compute(d_pos,
							  d_tag,
							  d_rtag,
							  box,
							  m_nlist-> getGpuNNeigh(),
							  m_nlist-> getGpuNList(),
							  m_nlist-> getNListIndexer(),
							  d_react_list,								  
							  d_n_tag_bond,
							  d_tag_bonds,
							  d_n_idx_bond,
							  d_idx_bonds,
							  pitch,
							  d_cris,
							  m_seed+timestep,
							  m_r_cut*m_r_cut,
							  m_nkinds,
							  d_pr,
							  h_ninit,
							  d_ninit,
							  d_init_group,
							  d_is_init,
							  d_maxcris,
							  Np,
							  m_func_params,
							  m_func_id,
							  m_T,
							  d_change_type,							  
							  m_block_size);				
			}
		else
			{		
			gpu_exchange_compute(d_pos,
							  d_tag,
							  d_rtag,
							  box,
							  m_nlist-> getGpuNNeigh(),
							  m_nlist-> getGpuNList(),
							  m_nlist-> getNListIndexer(),
							  d_react_list,								  
							  d_n_tag_bond,
							  d_tag_bonds,
							  d_n_idx_bond,
							  d_idx_bonds,
							  pitch,
							  d_cris,
							  m_seed+timestep,
							  m_r_cut*m_r_cut,
							  m_nkinds,
							  d_pr,
							  h_ninit,
							  d_ninit,
							  d_init_group,
							  d_is_init,
							  d_maxcris,
							  Np,
							  d_change_type,							  
							  m_block_size);
			}
		}
	else if(m_reaction_mode == SGAP)
		{
		if(m_min_dis_rule)
			{
			 gpu_SGAP_Dis_compute(d_pos,
							  d_tag,
							  d_rtag,
							  box,
							  m_nlist-> getGpuNNeigh(),
							  m_nlist-> getGpuNList(),
							  m_nlist-> getNListIndexer(),						  
							  d_n_tag_bond,
							  d_tag_bonds,
							  d_n_idx_bond,
							  d_idx_bonds,
							  pitch,
							  d_cris,
							  m_seed+timestep,
							  m_r_cut*m_r_cut,
							  m_nkinds,
							  d_pr,
							  h_ninit,
							  d_ninit,
							  d_init_group,
							  d_is_init,						  
							  d_maxcris,
							  Np,
							  m_new_bond_type,
							  d_change_type,							  
							  m_block_size);
			
			}
		else if (m_func_rule)
			{
			if(m_setVariantT)
				m_T = (Real) m_vT->getValue(timestep);			
			 gpu_SGAP_Func_compute(d_pos,
							  d_tag,
							  d_rtag,
							  box,
							  m_nlist-> getGpuNNeigh(),
							  m_nlist-> getGpuNList(),
							  m_nlist-> getNListIndexer(),
							  d_react_list,								  
							  d_n_tag_bond,
							  d_tag_bonds,
							  d_n_idx_bond,
							  d_idx_bonds,
							  pitch,
							  d_cris,
							  m_seed+timestep,
							  m_r_cut*m_r_cut,
							  m_nkinds,
							  d_pr,
							  h_ninit,
							  d_ninit,
							  d_init_group,
							  d_is_init,						  
							  d_maxcris,
							  Np,
							  m_new_bond_type,
							  m_func_params,
							  m_func_id,
							  m_T,
							  d_change_type,							  
							  m_block_size);
			}
		else
			{
			 gpu_SGAP_compute(d_pos,
							  d_tag,
							  d_rtag,
							  box,
							  m_nlist-> getGpuNNeigh(),
							  m_nlist-> getGpuNList(),
							  m_nlist-> getNListIndexer(),
							  d_react_list,								  
							  d_n_tag_bond,
							  d_tag_bonds,
							  d_n_idx_bond,
							  d_idx_bonds,
							  pitch,
							  d_cris,
							  m_seed+timestep,
							  m_r_cut*m_r_cut,
							  m_nkinds,
							  d_pr,
							  h_ninit,
							  d_ninit,
							  d_init_group,
							  d_is_init,						  
							  d_maxcris,
							  Np,	
							  m_new_bond_type,
							  d_change_type,							  
							  m_block_size);	
			}
	
		}
	else if(m_reaction_mode == INSERT)
		{
	
			gpu_insertion_compute(d_pos,
							  d_tag,
							  d_rtag,
							  box,
							  m_nlist-> getGpuNNeigh(),
							  m_nlist-> getGpuNList(),
							  m_nlist-> getNListIndexer(),
							  d_react_list,								  
							  d_n_tag_bond,
							  d_tag_bonds,
							  d_n_idx_bond,
							  d_idx_bonds,
							  pitch,
							  d_cris,
							  m_seed+timestep,
							  m_r_cut*m_r_cut,
							  m_nkinds,
							  d_pr,
							  h_ninit,
							  d_ninit,
							  d_init_group,
							  d_is_init,
							  d_maxcris,
							  Np,
							  m_new_bond_type,							  
							  d_change_type,							  
							  m_block_size);
		}		
	else
		{
		cerr << endl << "***Error! Polymerization have not been set with a mode" << endl << endl;
		throw runtime_error("Error computeChare in Polymerization");
		}

	CHECK_CUDA_ERROR();

	}

void export_Polymerization()
	{
	scope in_Polymerization_Chare = class_<Polymerization, boost::shared_ptr<Polymerization>, bases<Chare>, boost::noncopyable >
		("Polymerization", init< boost::shared_ptr<AllInfo>, boost::shared_ptr<NeighborList>, Real, unsigned int >())
		.def(init<boost::shared_ptr<AllInfo>, std::string, Real, boost::shared_ptr<NeighborList>, Real, unsigned int >())		
		.def("setPr", static_cast< void (Polymerization::*)(Real) >(&Polymerization::setPr)) 
		.def("setPr", static_cast< void (Polymerization::*)(const std::string &, const std::string &, Real)>(&Polymerization::setPr))
		.def("setPrFactor", static_cast< void (Polymerization::*)(Real) >(&Polymerization::setPrFactor)) 
		.def("setPrFactor", static_cast< void (Polymerization::*)(const std::string &, const std::string &, Real)>(&Polymerization::setPrFactor)) 
		.def("setExchangePr", &Polymerization::setExchangePr)
		.def("setInsertionPr", &Polymerization::setInsertionPr)			
		.def("initExPoint", &Polymerization::initExPoint)
		.def("setMaxCris", &Polymerization::setMaxCris)
		.def("setFrpMode", &Polymerization::setFrpMode)
		.def("setExchMode", &Polymerization::setExchMode)
		.def("setSgapMode", &Polymerization::setSgapMode)
		.def("setInsertionMode", &Polymerization::setInsertionMode)			
		.def("setNewBondType", &Polymerization::setNewBondType)	
		.def("setMinDisReactRule", &Polymerization::setMinDisReactRule)
		.def("setFuncReactRule", &Polymerization::setFuncReactRule)
		.def("setT", static_cast< void (Polymerization::*)(Real) >(&Polymerization::setT)) 
		.def("setT", static_cast< void (Polymerization::*)(boost::shared_ptr<Variant>)>(&Polymerization::setT))
		.def("setReactionTimes", &Polymerization::setReactionTimes)
		.def("setChangeTypeInReaction", &Polymerization::setChangeTypeInReaction)			
		;
    enum_<Polymerization::Func>("Func")
		.value("NoFunc",Polymerization::NoFunc)	
		.value("FENE",Polymerization::FENE)
		.value("harmonic",Polymerization::harmonic)
		;
	}


