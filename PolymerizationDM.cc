

#include "PolymerizationDM.h"
#include<time.h> 
#include<stdlib.h> 

#include <boost/python.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
using namespace boost::filesystem;
using namespace boost::python;
using namespace std;


PolymerizationDM::PolymerizationDM(boost::shared_ptr<AllInfo> all_info,
				  boost::shared_ptr<NeighborList> nlist, 
				  Real r_cut, 
				  unsigned int seed) 
	: Chare(all_info), m_nlist(nlist), m_r_cut(r_cut), m_seed(seed)
	{
	initData();
	Statistic();
	}

PolymerizationDM::PolymerizationDM(boost::shared_ptr<AllInfo> all_info,
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

void PolymerizationDM::initData()
	{
	m_all_info->initBondInfo();
	m_bond_info = m_all_info->getBondInfo();

	m_all_info->initAngleInfo();
	m_angle_info = m_all_info->getAngleInfo();
	
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
	m_pr = boost::shared_ptr<Array<Real3> >(new Array<Real3>(m_nkinds*m_nkinds*m_nkinds, location::host));
	m_maxcris = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>(m_nkinds, location::host));
	m_init_group = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>(nparticles, location::host));
	m_ninit = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>(4, location::host));
	m_change_type = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>(m_nkinds, location::host));
	
	m_bond_type_table = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>(m_nkinds*m_nkinds, location::host));
	m_angle_type_table = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>(m_nkinds*m_nkinds*m_nkinds, location::host));

	boost::shared_ptr<Array<unsigned int> > init = m_basic_info->getInit();
	if(init->isempty())
		init->resize(nparticles);
	boost::shared_ptr<Array<unsigned int> > cris = m_basic_info->getCris();
	if(cris->isempty())
		cris->resize(nparticles);
	boost::shared_ptr<Array<uint2> > bond_state = m_bond_info->getBondStateArray();
	if(bond_state->isempty())
		bond_state->resize(nparticles);
		
	Real3* h_pr = m_pr->getArray(location::host, access::readwrite);
	for(unsigned int i =0; i<m_nkinds*m_nkinds*m_nkinds;i++)
			h_pr[i].y =1.0;	
		
	unsigned int* h_maxcris = m_maxcris->getArray(location::host, access::readwrite);
	for(unsigned int i =0; i<m_nkinds;i++)
		h_maxcris[i] = 1;

	unsigned int* h_change_type = m_change_type->getArray(location::host, access::readwrite);
	for(unsigned int i =0; i<m_nkinds;i++)
		h_change_type[i] = i;		
		
	m_new_bond_type = 0;
	m_new_angle_type = 0;	
    m_period = 1;
	m_period_R = 1;
	m_max_added_bonds = 2;
	m_nm = 0;
	m_angle_limit = 1.0;
	m_init_ex_point = false;
	m_first_step = true;
	m_reaction_mode = FRP;
	m_set_max_cris = false;
	m_set_exch_pr = false;
	m_setPr = false;
	m_mode_set = false;
	m_generate_angle = false;
	m_bond_type_by_pair = false;
	m_angle_type_by_pair = false;	
	m_ObjectName = "PolymerizationDM";
	cout << "INFO : PolymerizationDM object has been build up !" << endl; 
	}

PolymerizationDM::~PolymerizationDM()
	{
	}
	
void PolymerizationDM::setNewBondType(const std::string &name)
	{
	m_new_bond_type= m_bond_info->switchNameToIndex(name);	
	}	

void PolymerizationDM::setNewAngleType(const std::string &name)
	{
	m_new_angle_type= m_angle_info->switchNameToIndex(name);	
	}
	
void PolymerizationDM::setChangeTypeInReaction(const std::string &name_origin, const std::string &name_new)
	{
	unsigned int* h_change_type = m_change_type->getArray(location::host, access::readwrite);
	unsigned int type_origin = m_basic_info->switchNameToIndex(name_origin);
	unsigned int type_new = m_basic_info->switchNameToIndex(name_new);	
	h_change_type[type_origin] = type_new;
	}

void PolymerizationDM::setNewBondTypeByPairs()
	{
	unsigned int* h_bond_type_table = m_bond_type_table->getArray(location::host, access::write);
	for(unsigned int i=0; i< m_nkinds; i++)
		{
		string namei = m_basic_info->switchIndexToName(i);
		for(unsigned int j=i; j< m_nkinds; j++)
			{
			string namej = m_basic_info->switchIndexToName(j);			
			string new_bond_type = namei+"-"+namej;
			unsigned int newbondid = m_bond_info->switchNameToIndex(new_bond_type);
			h_bond_type_table[i*m_nkinds+j] = newbondid;
			h_bond_type_table[j*m_nkinds+i] = newbondid;
			}		
		}
	m_bond_type_by_pair = true;	
	}	

void PolymerizationDM::setNewAngleTypeByPairs()
	{
	unsigned int* h_angle_type_table = m_angle_type_table->getArray(location::host, access::write);
	for(unsigned int i=0; i< m_nkinds; i++)
		{
		string namei = m_basic_info->switchIndexToName(i);
		for(unsigned int j=0; j< m_nkinds; j++)
			{
			string namej = m_basic_info->switchIndexToName(j);
			for(unsigned int k=i; k< m_nkinds; k++)
				{			
				string namek = m_basic_info->switchIndexToName(k);			
				string new_angle_type = namei+"-"+namej+"-"+namek;
				unsigned int newangleid = m_angle_info->switchNameToIndex(new_angle_type);
				h_angle_type_table[i*m_nkinds*m_nkinds+j*m_nkinds+k] = newangleid;
				h_angle_type_table[k*m_nkinds*m_nkinds+j*m_nkinds+i] = newangleid;
				}
			}			
		}
	m_angle_type_by_pair = true;		
	}	

void PolymerizationDM::creatInitor(const std::string &name, Real percent)
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

void PolymerizationDM::setFrpMode()
	{
	m_reaction_mode = FRP;
	m_mode_set = true;	
	}

void PolymerizationDM::setSgapMode()
	{
	m_reaction_mode = SGAP;	
	m_mode_set = true;		
	}

void PolymerizationDM::setExchMode()
	{
	m_reaction_mode = EXCH;		
	m_mode_set = true;		
	}
	
void PolymerizationDM::generateAngle(bool generate_angle)
	{
	m_generate_angle = generate_angle;
	}

void PolymerizationDM::setExchangePr(const std::string &name1, const std::string &name2, const std::string &name3, Real exchange)
	{
	unsigned int typ1= m_basic_info -> switchNameToIndex(name1);
	unsigned int typ2= m_basic_info -> switchNameToIndex(name2);
	unsigned int typ3= m_basic_info -> switchNameToIndex(name3);
	
	if (typ1 >= m_nkinds || typ2 >= m_nkinds|| typ3 >= m_nkinds)
		{
		cerr << endl << "***Error! Trying to set exchange for an non existent type! " << typ1 << "," << typ2 << "," << typ2 << endl << endl;
		throw runtime_error("PolymerizationDM::stExchange argument error");
		}
    if(exchange<0.0)
		{
		cerr << endl << "***Error! Trying to set exchange little than zero! " << exchange<< endl << endl;
		throw runtime_error("PolymerizationDM::stExchangeargument error");
		}
		
	Real3* h_pr = m_pr->getArray(location::host, access::readwrite);
	h_pr[typ1*m_nkinds*m_nkinds + typ2*m_nkinds +typ3].x = exchange;

	m_set_exch_pr=true;
	}
	
void PolymerizationDM::setMaxCris(const std::string &name, unsigned int maxcris)
	{
	unsigned int typ= m_basic_info -> switchNameToIndex(name);
	
	if (typ >= m_nkinds)
		{
		cerr << endl << "***Error! Trying to set maximum cris for an non existent type ! " << name << endl << endl;
		throw runtime_error("PolymerizationDM::setMaxCris argument error");
		}
	if (maxcris>20)
		{
		cerr << endl << "***Error! Trying to set maximum cris larger than the limited 20! " <<maxcris<< endl << endl;
		throw runtime_error("PolymerizationDM::setMaxCris argument error");
		}
	unsigned int* h_maxcris = m_maxcris->getArray(location::host, access::readwrite);
	h_maxcris[typ] = maxcris;
	
	if(maxcris>m_max_added_bonds)
		m_max_added_bonds = maxcris;

	m_set_max_cris=true;
	}
	
void PolymerizationDM::setPr(const std::string &name1, const std::string &name2, Real Pr)
	{
	
	unsigned int typ1= m_basic_info -> switchNameToIndex(name1);
	unsigned int typ2= m_basic_info -> switchNameToIndex(name2);

	if (typ1 >= m_nkinds || typ2 >= m_nkinds)
		{
		cerr << endl << "***Error! Trying to set pr for an non existent type! " << typ1 << "," << typ2 << endl << endl;
		throw runtime_error("PolymerizationDM::setPr argument error");
		}
	Real3* h_pr = m_pr->getArray(location::host, access::readwrite);
	h_pr[typ1*m_nkinds + typ2].x = Pr;

	m_setPr = true;
	}
	
void PolymerizationDM::setPr(Real Pr)
	{
	Real3* h_pr = m_pr->getArray(location::host, access::readwrite);
	for(unsigned int i =0; i<m_nkinds*m_nkinds*m_nkinds;i++)
			h_pr[i].x =Pr;

	m_setPr = true;
	}
	
void PolymerizationDM::setPrFactor(const std::string &name1, const std::string &name2, Real Factor)
	{
	unsigned int typ1= m_basic_info -> switchNameToIndex(name1);
	unsigned int typ2= m_basic_info -> switchNameToIndex(name2);

	if (typ1 >= m_nkinds || typ2 >= m_nkinds)
		{
		cerr << endl << "***Error! Trying to set pr for an non existent type! " << typ1 << "," << typ2 << endl << endl;
		throw runtime_error("PolymerizationDM::setPrFactor argument error");
		}
	Real3* h_pr = m_pr->getArray(location::host, access::readwrite);
	h_pr[typ1*m_nkinds + typ2].y = Factor;
	}
	
void PolymerizationDM::setPrFactor(Real Factor)
	{
	Real3* h_pr = m_pr->getArray(location::host, access::readwrite);
	for(unsigned int i =0; i<m_nkinds*m_nkinds*m_nkinds;i++)
			h_pr[i].y =Factor;
	}
	
void PolymerizationDM::setAngleLowerLimitDegree(Real angle_limit)
	{
	m_angle_limit = cos(M_PI*angle_limit/180.0);
	}

void PolymerizationDM::initExPoint()
	{
	m_init_ex_point = true;
	}
	
void PolymerizationDM::initiateExchPoint()
	{
	if (!m_set_exch_pr)
		{
		cerr << endl << "***Error! Please first set ligand exchange probability! "<< endl << endl;
		throw runtime_error("PolymerizationDM::initExPoint error");
		}	
	unsigned int nparticles = m_basic_info->getN();
	unsigned int* h_init_group = m_init_group->getArray(location::host, access::readwrite);
	unsigned int* h_is_init = m_basic_info->getInit()->getArray(location::host, access::readwrite);		
	unsigned int* h_ninit = m_ninit->getArray(location::host, access::readwrite);
	Real3* h_pr = m_pr->getArray(location::host, access::read);

    unsigned int* h_n_bond = m_bond_info->getBondNumTagArray()->getArray(location::host, access::read); 
	uint2* h_bonds = m_bond_info->getBondTagArray()->getArray(location::host, access::read);
	unsigned int bond_pitch= m_bond_info->getBondTagArray()->getPitch();
	
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
			uint2 bond = h_bonds[j*bond_pitch + taga];
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

void PolymerizationDM::Statistic()
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
	cout << "INFO : PolymerizationDM statistics, " <<Numinit<<" initiators"<< endl;				
	cout << "INFO : PolymerizationDM statistics, " <<Numcris<<" free monomers"<< endl;
	if (Numinit==0)
		{
		cerr << endl << "***Error! No initiators "<< endl << endl;
		throw runtime_error("PolymerizationDM::Statistic error");	
		}
	}
	
void PolymerizationDM::checkLiEx()
	{
	unsigned int nparticles = m_basic_info->getN();
	unsigned int* h_init = m_basic_info->getInit()->getArray(location::host, access::read);	
	Real3* h_pr = m_pr->getArray(location::host, access::read);

    unsigned int* h_n_bond = m_bond_info->getBondNumTagArray()->getArray(location::host, access::read); 
	uint2* h_bonds = m_bond_info->getBondTagArray()->getArray(location::host, access::read);
	unsigned int bond_pitch= m_bond_info->getBondTagArray()->getPitch();
	
	Real4* h_pos = m_basic_info->getPos()->getArray(location::host, access::read);	
	unsigned int* h_rtag = m_basic_info->getRtag()->getArray(location::host, access::read); 	
	
	for (unsigned int taga = 0; taga < nparticles; taga++)
		{
		if(h_init[taga]==1)
			{
			unsigned int idxa= h_rtag[taga];
			Realint fi;
			fi.f = h_pos[idxa].w;			
			unsigned int typi = fi.i;
			
			unsigned int nbonds = h_n_bond[taga];			
			for(unsigned int j =0; j< nbonds; j++)
				{
				uint2 bond = h_bonds[j*bond_pitch + taga];
				unsigned int tagb = bond.x;
				if(h_init[tagb]==1)
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
						cerr << endl << "***Error! Exchange mode check, a bonded exchange active point, "<<taga<<" "<<tagb<< endl << endl;
						throw runtime_error("PolymerizationDM::checkLiEx error");
						}
					}
				}
			}
		}
	}

void PolymerizationDM::checkFRP()
	{
	unsigned int nparticles = m_basic_info->getN();
	unsigned int* h_init = m_basic_info->getInit()->getArray(location::host, access::read);
	unsigned int* h_cris = m_basic_info->getCris()->getArray(location::host, access::read);
	Real3* h_pr = m_pr->getArray(location::host, access::read);
	
	Real4* h_pos = m_basic_info->getPos()->getArray(location::host, access::read);	
	unsigned int* h_rtag = m_basic_info->getRtag()->getArray(location::host, access::read); 		
	
	for (unsigned int taga = 0; taga < nparticles; taga++)
		{
		if(h_init[taga]==0&&h_cris[taga]==0)
			{
			unsigned int idxa = h_rtag[taga];
			Realint fi;
			fi.f = h_pos[idxa].w;			
			unsigned int typi = fi.i;
			Real possi = 0.0;				
			for(unsigned int typ1=0; typ1<m_nkinds; typ1++)
				{
				unsigned int pair1 = typ1*m_nkinds + typi;				
				possi += h_pr[pair1].x;						
				}						
			if(possi>0)
				{
				m_nm += 1;
				}
			}
		}
	}

void PolymerizationDM::checkSGAP()
	{
	unsigned int nparticles = m_basic_info->getN();
	unsigned int* h_init = m_basic_info->getInit()->getArray(location::host, access::read);
	unsigned int* h_cris = m_basic_info->getCris()->getArray(location::host, access::read);
	Real3* h_pr = m_pr->getArray(location::host, access::read);
	unsigned int* h_maxcris = m_maxcris->getArray(location::host, access::read);	
	Real4* h_pos = m_basic_info->getPos()->getArray(location::host, access::read);	
	unsigned int* h_rtag = m_basic_info->getRtag()->getArray(location::host, access::read); 		
	
	for (unsigned int taga = 0; taga < nparticles; taga++)
		{
		unsigned int idxa = h_rtag[taga];
		Realint fi;
		fi.f = h_pos[idxa].w;			
		unsigned int typi = fi.i;
		if(h_init[taga]==0&&h_cris[taga]<h_maxcris[typi])
			{

			Real possi = 0.0;				
			for(unsigned int typ1=0; typ1<m_nkinds; typ1++)
				{
				unsigned int pair1 = typ1*m_nkinds + typi;				
				possi += h_pr[pair1].x;						
				}						
			if(possi>0)
				{
				m_nm += 1;
				}
			}
		}
	}

void PolymerizationDM::computeChare(unsigned int timestep)
	{
    if (m_first_step)
        {
		if(!m_mode_set)
			{
			if(m_set_exch_pr)
				m_reaction_mode = EXCH;
			else if (m_set_max_cris&&m_setPr)
				m_reaction_mode = SGAP;
			else if(m_setPr)
				m_reaction_mode = FRP;
			else
				{
				cerr << endl << "***Error! PolymerizationDM can not parse the reaction mode by the parameter set!" << endl << endl;
				throw runtime_error("Error computeChare in PolymerizationDM");
				}
			m_mode_set = true;
			}
			
		unsigned int n_max_bonded = m_bond_info->getBondTagArray()->getHeight();
		unsigned int n_max_added_angles = 3;
		unsigned int n_max_added_angles_exclusion = 2;
		unsigned int n_all_bond = n_max_bonded + m_max_added_bonds;		
		if(m_reaction_mode == FRP)
			{
			n_max_added_angles = n_all_bond*(n_all_bond-1) + n_all_bond*(n_all_bond-1)/2;
			n_max_added_angles_exclusion = n_all_bond*(n_all_bond-1);
			}
		else if(m_reaction_mode == SGAP)
			{
			n_max_added_angles = n_all_bond*(n_all_bond-1) + n_all_bond*(n_all_bond-1)/2;
			n_max_added_angles_exclusion = n_all_bond*(n_all_bond-1);			
			}
			
//		cout<<" 617 "<<n_max_added_angles<<" "<<n_max_added_angles_exclusion<<endl;
		
        m_bond_info->growBondArrayHeight(m_max_added_bonds);
		if(m_generate_angle)
			m_angle_info->growAngleArrayHeight(n_max_added_angles);
		if(m_nlist->checkBondExclusions())
			m_nlist->growExclusionList(m_max_added_bonds);	
		if(m_nlist->checkAngleExclusions())
			m_nlist->growExclusionList(n_max_added_angles_exclusion);
        m_first_step = false;
        }
		
	if(m_init_ex_point)
		initiateExchPoint();
		
	m_nlist->compute(timestep);
	   
	if(m_mode_set)
		{
		if(m_reaction_mode == FRP)
			{
			checkFRP();
			cout<<"INFO : Free Radical PolymerizationDM Mode!"<<endl;
			}
		else if(m_reaction_mode == SGAP)
			{
			checkSGAP();			
			cout<<"INFO : Step Growth Addition PolymerizationDM mode!"<<endl;
			}
		else if(m_reaction_mode == EXCH)
			{
			checkLiEx();
			cout<<"INFO : Exchange Reaction Mode!"<<endl;
			}		
		else
			{
			cerr << endl << "***Error! PolymerizationDM have not chosen a mode" << endl << endl;
			throw runtime_error("Error computeChare in PolymerizationDM");		
			}
		m_mode_set=false;
		}

    unsigned int* d_n_tag_bond = m_bond_info->getBondNumTagArray()->getArray(location::device, access::readwrite); 
	uint2* d_tag_bonds = m_bond_info->getBondTagArray()->getArray(location::device, access::readwrite);
    unsigned int* d_n_idx_bond = m_bond_info->getBondNumIdxArray()->getArray(location::device, access::readwrite); 
	uint2* d_idx_bonds = m_bond_info->getBondIdxArray()->getArray(location::device, access::readwrite);
	unsigned int bond_pitch= m_bond_info->getBondTagArray()->getPitch();
    uint2* d_bond_state = m_bond_info->getBondStateArray()->getArray(location::device, access::readwrite); 

    unsigned int* d_n_tag_angle = m_angle_info->getAngleNumTagArray()->getArray(location::device, access::readwrite); 
	uint4* d_tag_angles = m_angle_info->getAngleTagArray()->getArray(location::device, access::readwrite);
    unsigned int* d_n_idx_angle = m_angle_info->getAngleNumIdxArray()->getArray(location::device, access::readwrite); 
	uint4* d_idx_angles = m_angle_info->getAngleIdxArray()->getArray(location::device, access::readwrite);
	unsigned int angle_pitch= m_angle_info->getAngleTagArray()->getPitch();
	unsigned int* d_change_type = m_change_type->getArray(location::device, access::read);
	
	Reaction_Data reaction_data;
	reaction_data.d_n_tag_bond = d_n_tag_bond;
	reaction_data.d_tag_bonds = d_tag_bonds;
	reaction_data.d_n_idx_bond = d_n_idx_bond;
	reaction_data.d_idx_bonds = d_idx_bonds;
	reaction_data.d_bond_state = d_bond_state;
	reaction_data.bond_pitch = bond_pitch;
	reaction_data.bond_exclusions = m_nlist->checkBondExclusions();
	
	reaction_data.d_n_tag_angle = d_n_tag_angle;
	reaction_data.d_tag_angles = d_tag_angles;
	reaction_data.d_n_idx_angle = d_n_idx_angle;
	reaction_data.d_idx_angles = d_idx_angles;
	reaction_data.angle_pitch = angle_pitch;
	reaction_data.angle_generate = m_generate_angle;
	reaction_data.angle_exclusions = m_nlist->checkAngleExclusions();
	if(m_nlist->getExclusionsSet())
		{
		reaction_data.d_n_ex_tag = m_nlist->getNExTag()->getArray(location::device, access::readwrite);
		reaction_data.d_ex_list_tag = m_nlist->getExListTag()->getArray(location::device, access::readwrite);
		reaction_data.d_n_ex_idx = m_nlist->getNExIdx()->getArray(location::device, access::readwrite); 
		reaction_data.d_ex_list_idx = m_nlist->getExListIdx()->getArray(location::device, access::readwrite);
		reaction_data.ex_list_indexer = m_nlist->getExListIndexer();
		}
	reaction_data.d_bond_type_table = m_bond_type_table->getArray(location::device, access::read);
	reaction_data.d_angle_type_table = m_angle_type_table->getArray(location::device, access::read);
	reaction_data.bond_type_by_pair = m_bond_type_by_pair;
	reaction_data.angle_type_by_pair = m_angle_type_by_pair;
	reaction_data.d_change_type = d_change_type;
	reaction_data.angle_limit = m_angle_limit;	

	Real4* d_pos = m_basic_info->getPos()->getArray(location::device, access::readwrite);
	unsigned int* d_tag = m_basic_info->getTag()->getArray(location::device, access::read);	
	unsigned int* d_rtag = m_basic_info->getRtag()->getArray(location::device, access::read); 
	unsigned int Np = m_basic_info->getN();
	 
	const gpu_boxsize& box = m_basic_info->getBoxGPU();
	unsigned int* d_init_group = m_init_group->getArray(location::device, access::readwrite);
	unsigned int* d_cris = m_basic_info->getCris()->getArray(location::device, access::readwrite);
	unsigned int* d_init = m_basic_info->getInit()->getArray(location::device, access::readwrite);

	unsigned int* h_ninit = m_ninit->getArray(location::host, access::read);
	Real cr = Real(m_nm - h_ninit[3])/Real(m_nm);
	Real3* h_pr = m_pr->getArray(location::host, access::readwrite);
	for(unsigned int i =0; i<m_nkinds*m_nkinds*m_nkinds;i++)
			h_pr[i].z = h_pr[i].x * cr;
// if(timestep%1000==0)			
//	 cout<<m_nm<<" "<<h_ninit[3]<<" "<<cr<<endl;
			
	unsigned int* d_ninit = m_ninit->getArray(location::device, access::readwrite);
	Real3* d_pr = m_pr->getArray(location::device, access::read);	 
	unsigned int* d_maxcris = m_maxcris->getArray(location::device, access::readwrite);	
	 // unsigned int* h_init_group = m_init_group->getArray(location::host, access::readwrite);	 
	// cout<<"bond_pitch"<<bond_pitch<<endl;
		// for(unsigned int i =0; i<h_ninit[0];i++)
			// cout<<h_init_group[i]<<endl;
 
	 
	if(m_reaction_mode == FRP)
		{
		gpu_FRP_DM_compute(d_pos,
						  d_tag,
						  d_rtag,
                          box,
						  m_nlist-> getGpuNNeighChange(),
						  m_nlist-> getGpuNListChange(),
						  m_nlist-> getNListIndexer(),						  
						  reaction_data,
						  d_cris,
						  m_seed+timestep,
						  m_r_cut*m_r_cut,
						  m_nkinds,
						  d_pr,
						  h_ninit,
						  d_ninit,
						  d_init_group,
						  d_init,
						  Np,
						  m_new_bond_type,
						  m_new_angle_type,
						  m_period_R,
						  m_block_size);
		}
	else if(m_reaction_mode == EXCH)
		{
		gpu_exchange_DM_compute(d_pos,
						  d_tag,
						  d_rtag,
						  box,
						  m_nlist-> getGpuNNeigh(),
						  m_nlist-> getGpuNList(),
						  m_nlist-> getNListIndexer(),						  
						  reaction_data,
						  d_cris,
						  m_seed+timestep,
						  m_r_cut*m_r_cut,
						  m_nkinds,
						  d_pr,
						  h_ninit,
						  d_ninit,
						  d_init_group,
						  d_init,
						  d_maxcris,
						  Np,
						  m_period_R,
						  m_block_size);
		}
	else if(m_reaction_mode == SGAP)
		{
		gpu_SGAP_DM_compute(d_pos,
						  d_tag,
						  d_rtag,
						  box,
						  m_nlist-> getGpuNNeighChange(),
						  m_nlist-> getGpuNListChange(),
						  m_nlist-> getNListIndexer(),						  
						  reaction_data,
						  d_cris,
						  m_seed+timestep,
						  m_r_cut*m_r_cut,
						  m_nkinds,
						  d_pr,
						  h_ninit,
						  d_ninit,
						  d_init_group,
						  d_init,						  
						  d_maxcris,
						  Np,
						  m_new_bond_type,
						  m_new_angle_type,						  
						  m_period_R,						  
						  m_block_size);
		}	
	else
		{
		cerr << endl << "***Error! Polymerization have not been set with a mode" << endl << endl;
		throw runtime_error("Error computeChare in Polymerization");
		}

	CHECK_CUDA_ERROR();
// update reaction possibility
	}

void export_PolymerizationDM()
	{
	scope in_PolymerizationDM_Chare = class_<PolymerizationDM, boost::shared_ptr<PolymerizationDM>, bases<Chare>, boost::noncopyable >
		("PolymerizationDM", init< boost::shared_ptr<AllInfo>, boost::shared_ptr<NeighborList>, Real, unsigned int >())
		.def(init<boost::shared_ptr<AllInfo>, std::string, Real, boost::shared_ptr<NeighborList>, Real, unsigned int >())		
		.def("setPr", static_cast< void (PolymerizationDM::*)(Real) >(&PolymerizationDM::setPr)) 
		.def("setPr", static_cast< void (PolymerizationDM::*)(const std::string &, const std::string &, Real)>(&PolymerizationDM::setPr))
		.def("setPrFactor", static_cast< void (PolymerizationDM::*)(Real) >(&PolymerizationDM::setPrFactor)) 
		.def("setPrFactor", static_cast< void (PolymerizationDM::*)(const std::string &, const std::string &, Real)>(&PolymerizationDM::setPrFactor)) 
		.def("setExchangePr", &PolymerizationDM::setExchangePr)
		.def("initExPoint", &PolymerizationDM::initExPoint)
		.def("setMaxCris", &PolymerizationDM::setMaxCris)
		.def("setFrpMode", &PolymerizationDM::setFrpMode)
		.def("setExchMode", &PolymerizationDM::setExchMode)
		.def("setSgapMode", &PolymerizationDM::setSgapMode)
		.def("setNewBondType", &PolymerizationDM::setNewBondType)
		.def("setNewAngleType", &PolymerizationDM::setNewAngleType)
		.def("setNewBondTypeByPairs", &PolymerizationDM::setNewBondTypeByPairs)
		.def("setNewAngleTypeByPairs", &PolymerizationDM::setNewAngleTypeByPairs)			
		.def("generateAngle", &PolymerizationDM::generateAngle)
		.def("setChangeTypeInReaction", &PolymerizationDM::setChangeTypeInReaction)
		.def("setAngleLowerLimitDegree", &PolymerizationDM::setAngleLowerLimitDegree)
		;
	}


