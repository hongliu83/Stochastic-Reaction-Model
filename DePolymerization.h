

#include "Chare.h"
#include "DePolymerization.cuh"
#include "Index.h"
#include "NeighborList.h"

#include <string>
#include <boost/shared_ptr.hpp>
#include <fstream>
#include "Variant.h"

#ifndef __DE_POLYMERIZATION_H__
#define __DE_POLYMERIZATION_H__


class DePolymerization : public Chare
	{
	public:
    enum Func
        {
		NoFunc=0,
        FENE,
		harmonic,		
        };		
		DePolymerization(boost::shared_ptr<AllInfo> all_info, Real T, unsigned int seed);

		virtual ~DePolymerization();

        virtual void computeChare(unsigned int timestep);
        void setParams(const std::string &bondname, Real K, Real r_0, Real b_0, Real epsilon0, Real Pr, Func function);
		virtual void setT(boost::shared_ptr<Variant> T)
            {
            m_vT = T;
			m_setVariantT = true;
            }
		virtual void setT(Real T)
            {
            m_T = T;
			m_setVariantT = false;
            }
		void setChangeTypeInReaction(const std::string &name_origin, const std::string &name_new);			
	protected:         	
		boost::shared_ptr<Array<Real4> > m_params; 
        unsigned int m_NBondKinds;        
        boost::shared_ptr<BondInfo> m_bond_info;
        boost::shared_ptr<Variant> m_vT;
		Real m_T;
		bool m_setVariantT;	
		unsigned int m_seed;
		boost::shared_ptr<Array<unsigned int> > m_change_type;		
	};
	

void export_DePolymerization();

#endif
