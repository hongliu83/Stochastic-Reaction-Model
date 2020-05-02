#include "DePolymerization.h"
#include<time.h> 
#include<stdlib.h> 

#include <boost/python.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
using namespace boost::filesystem;
using namespace boost::python;
using namespace std;


DePolymerization::DePolymerization(boost::shared_ptr<AllInfo> all_info, 
									Real T, 
									unsigned int seed) 
	: Chare(all_info), m_T(T), m_seed(seed)
	{
	m_all_info->initBondInfo();
	m_bond_info = m_all_info->getBondInfo();
	
	m_NBondKinds = m_bond_info->getNBondKinds();
	m_setVariantT = false;	
    if (m_NBondKinds == 0)
        throw runtime_error("Error building DePolymerization, no bond types!");
	m_params = boost::shared_ptr<Array<Real4> >(new Array<Real4>(m_NBondKinds*2, location::host));
	
	m_change_type = boost::shared_ptr<Array<unsigned int> >(new Array<unsigned int>(m_basic_info->getNKinds(), location::host));
	unsigned int* h_change_type = m_change_type->getArray(location::host, access::readwrite);
	for(unsigned int i =0; i<m_basic_info->getNKinds();i++)
		h_change_type[i] = i;
	
    unsigned int nparticles = m_basic_info->getN();		
	boost::shared_ptr<Array<unsigned int> > cris = m_basic_info->getCris();
	if(cris->isempty())
		cris->resize(nparticles);	
	m_ObjectName = "DePolymerization";
	cout << "INFO : DePolymerization object has been build up !" << endl;
	}
	
DePolymerization::~DePolymerization()
	{
	}
	
void DePolymerization::setChangeTypeInReaction(const std::string &name_origin, const std::string &name_new)
	{
	unsigned int* h_change_type = m_change_type->getArray(location::host, access::readwrite);
	unsigned int type_origin = m_basic_info->switchNameToIndex(name_origin);
	unsigned int type_new = m_basic_info->switchNameToIndex(name_new);	
	h_change_type[type_origin] = type_new;
	}
	
void DePolymerization::setParams(const std::string &bondname, Real K, Real r_0, Real b_0, Real epsilon0, Real Pr, Func function)
	{
	unsigned int type= m_bond_info -> switchNameToIndex(bondname);
	Real4* h_params = m_params->getArray(location::host, access::readwrite);
	Real offset=0;
	if(function == FENE)
		{
		if (b_0 >= r_0)
			{
			cerr << endl << "***Error! Trying to set b_0 = "<<b_0<<" greater than r_0 = "<<r_0<<"!"<<endl << endl;
			throw runtime_error("DePolymerization::setParams argument error"); 
			}
		if (b_0<0)
			{
			cerr << endl << "***Error! Trying to set b_0 = "<<b_0<<" less than 0"<<"!"<<endl << endl;
			throw runtime_error("DePolymerization::setParams argument error"); 
			}			
		offset = -0.5f * K * r_0*r_0*log(1.0 - (b_0*b_0)/(r_0*r_0));		
		offset += epsilon0;
		}
	else if(function == harmonic)
		{
		if (b_0<0)
			{
			cerr << endl << "***Error! Trying to set b_0 = "<<b_0<<" less than 0"<<"!"<<endl << endl;
			throw runtime_error("DePolymerization::setParams argument error"); 
			}				
		offset = epsilon0;
		}

	h_params[type] = ToReal4(K, r_0, b_0, offset);
	h_params[type+m_NBondKinds] = ToReal4(Pr, Real(function), 0.0f, 0.0f);
	}

void DePolymerization::computeChare(unsigned int timestep)
	{
    unsigned int* d_n_tag_bond = m_bond_info->getBondNumTagArray()->getArray(location::device, access::readwrite); 
	uint2* d_tag_bonds = m_bond_info->getBondTagArray()->getArray(location::device, access::readwrite);
    unsigned int* d_n_idx_bond = m_bond_info->getBondNumIdxArray()->getArray(location::device, access::readwrite); 
	uint2* d_idx_bonds = m_bond_info->getBondIdxArray()->getArray(location::device, access::readwrite);
	unsigned int pitch= m_bond_info->getBondTagArray()->getPitch();

	Real4* d_pos = m_basic_info->getPos()->getArray(location::device, access::read);
	unsigned int* d_rtag = m_basic_info->getRtag()->getArray(location::device, access::read); 	
	unsigned int Np = m_basic_info->getN();
	Real4* d_params = m_params->getArray(location::device, access::read);	 
	const gpu_boxsize& box = m_basic_info->getBoxGPU();
	unsigned int* d_cris = m_basic_info->getCris()->getArray(location::device, access::readwrite);
	unsigned int* d_change_type = m_change_type->getArray(location::device, access::read);	
	if(m_setVariantT)
		m_T = (Real) m_vT->getValue(timestep);

	gpu_depolymerization_compute(d_pos,
								d_rtag,
								d_cris,
								box,						  
							    d_n_tag_bond,
							    d_tag_bonds,
							    d_n_idx_bond,
							    d_idx_bonds,
								pitch,
								d_params,
								m_T,
								m_seed+timestep,
								m_NBondKinds,
								Np,	
								d_change_type,
								m_block_size);


        CHECK_CUDA_ERROR();

	}

void export_DePolymerization()
	{
    scope in_DePolymerization_Chare = class_<DePolymerization, boost::shared_ptr<DePolymerization>, bases<Chare>, boost::noncopyable >
		("DePolymerization", init< boost::shared_ptr<AllInfo>, Real, unsigned int >())
		.def("setParams", &DePolymerization::setParams)
		.def("setT", static_cast< void (DePolymerization::*)(Real) >(&DePolymerization::setT)) 
		.def("setT", static_cast< void (DePolymerization::*)(boost::shared_ptr<Variant>)>(&DePolymerization::setT))	
		.def("setChangeTypeInReaction", &DePolymerization::setChangeTypeInReaction)		
		;
    enum_<DePolymerization::Func>("Func")
		.value("NoFunc",DePolymerization::NoFunc)	
		.value("FENE",DePolymerization::FENE)
		.value("harmonic",DePolymerization::harmonic)
		;
	}
