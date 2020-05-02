

#include "Chare.h"
#include "Polymerization.cuh"
#include "PolymerizationDis.cuh"
#include "Index.h"
#include "NeighborList.h"

#include <string>
#include <boost/shared_ptr.hpp>
#include <fstream>
#include "Variant.h"

#ifndef __POLYMERIZATION_H__
#define __POLYMERIZATION_H__


class Polymerization : public Chare
	{
	public:
    enum Func
        {
		NoFunc=0,
        FENE,
		harmonic,		
        };

    enum ReactionMode
        {
		FRP=0,
        SGAP,
		EXCH,
		INSERT,		
        };
		Polymerization(boost::shared_ptr<AllInfo> all_info,
		        boost::shared_ptr<NeighborList> nlist,
		        Real r_cut, 
		        unsigned int seed);
				
		Polymerization(boost::shared_ptr<AllInfo> all_info,	
                const std::string &type,
				Real percent,
		        boost::shared_ptr<NeighborList> nlist,
		        Real r_cut, 
		        unsigned int seed);	
		virtual ~Polymerization();
		void initData();
        virtual void computeChare(unsigned int timestep);	
        void setPr(Real Pr);
        void setPrFactor(Real Factor);		
        void setPr(const std::string &name1, const std::string &name2, Real Pr);
        void setPrFactor(const std::string &name1, const std::string &name2, Real Factor);
		void setExchangePr(const std::string &name1,const std::string &name2, const std::string &name3, Real exchange);
		void setInsertionPr(const std::string &name1,const std::string &name2, const std::string &name3, Real insertion);		
        void setMaxCris(const std::string &name, unsigned int maxcris);		
        void initExPoint();
        void initiateExchPoint();
		void creatInitor(const std::string &name, Real percent);		
		void Statistic();
		void setFrpMode();
		void setSgapMode();
		void setExchMode();
		void setInsertionMode();		
		void setMinDisReactRule(bool mindis_rule);	
		void setFuncReactRule(bool func_rule, Real K, Real r_0, Real b_0, Real epsilon0, Func function);
		void setNewBondType(const std::string &name);
		void checkLiEx();
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
		void setReactionTimes(unsigned int rtimes)
            {
			unsigned int* h_ninit = m_ninit->getArray(location::host, access::readwrite);
			h_ninit[3]=rtimes;
            }
		void setChangeTypeInReaction(const std::string &name_origin, const std::string &name_new);			
	protected:
        std::string m_fname;           	
		boost::shared_ptr<NeighborList> m_nlist;	
		Real m_r_cut;			
		unsigned int m_seed;
        boost::shared_ptr<BondInfo> m_bond_info;
		bool m_first_step;
		boost::shared_ptr<Array<Real2> > m_pr;	
		boost::shared_ptr<Array<unsigned int> > m_maxcris;		
		boost::shared_ptr<Array<unsigned int> > m_init_group;	
		boost::shared_ptr<Array<unsigned int> > m_reaction_times;
		boost::shared_ptr<Array<unsigned int> > m_react_list;
		
		ReactionMode m_reaction_mode;
		bool m_set_max_cris;
		bool m_set_exch_pr;
		bool m_setPr;
		bool m_set_insert_pr;		
		bool m_init_ex_point;
		bool m_init_cris;
		bool m_mode_set;
		bool m_min_dis_rule;
		bool m_func_rule;
		boost::shared_ptr<Array<unsigned int> > m_ninit;   // m_ninit[0] the number of initiator or anchor; m_ninit[3] the number of reaction times
		unsigned int m_nkinds;
		unsigned int m_max_added_bonds;
		unsigned int m_new_bond_type;
        boost::shared_ptr<Variant> m_vT;
		Real m_T;
		Real4 m_func_params;
		int m_func_id;
		bool m_setVariantT;	
		boost::shared_ptr<Array<unsigned int> > m_change_type;		
	};
	

void export_Polymerization();

#endif
