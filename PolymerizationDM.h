
#include "Chare.h"
#include "PolymerizationDM.cuh"
#include "Index.h"
#include "NeighborList.h"

#include <string>
#include <boost/shared_ptr.hpp>
#include <fstream>
#include "Variant.h"

#ifndef __POLYMERIZATION_DM_H__
#define __POLYMERIZATION_DM_H__


class PolymerizationDM : public Chare
	{
	public:
    enum ReactionMode
        {
		FRP=0,
        SGAP,
		EXCH,		
        };
		PolymerizationDM(boost::shared_ptr<AllInfo> all_info,
		        boost::shared_ptr<NeighborList> nlist,
		        Real r_cut,
		        unsigned int seed);
				
		PolymerizationDM(boost::shared_ptr<AllInfo> all_info,	
                const std::string &type,
				Real percent,
		        boost::shared_ptr<NeighborList> nlist,
		        Real r_cut,
		        unsigned int seed);
		virtual ~PolymerizationDM();
		void initData();
        virtual void computeChare(unsigned int timestep);	
        virtual void setPeriod(unsigned int period)
			{
	        m_period_R = period;
			}
        void setPr(Real Pr);
        void setPrFactor(Real Factor);		
        void setPr(const std::string &name1, const std::string &name2, Real Pr);
        void setPrFactor(const std::string &name1, const std::string &name2, Real Factor);
		void setExchangePr(const std::string &name1,const std::string &name2, const std::string &name3, Real exchange);		
        void setMaxCris(const std::string &name, unsigned int maxcris);		
        void initExPoint();
        void initiateExchPoint();
		void creatInitor(const std::string &name, Real percent);		
		void Statistic();
		void setFrpMode();
		void setSgapMode();
		void setExchMode();
		void setNewBondType(const std::string &name);
		void setNewAngleType(const std::string &name);
		void setNewBondTypeByPairs();
		void setNewAngleTypeByPairs();
		void checkLiEx();
		void checkFRP();
		void checkSGAP();
		void generateAngle(bool generate_angle);
		void setChangeTypeInReaction(const std::string &name_origin, const std::string &name_new);
		void setAngleLowerLimitDegree(Real angle_limit);
	protected:
        std::string m_fname;           	
		boost::shared_ptr<NeighborList> m_nlist;	
		Real m_r_cut;			
		unsigned int m_seed;
        boost::shared_ptr<BondInfo> m_bond_info;
        boost::shared_ptr<AngleInfo> m_angle_info;
		bool m_first_step;
		boost::shared_ptr<Array<Real3> > m_pr;	
		boost::shared_ptr<Array<unsigned int> > m_maxcris;		
		boost::shared_ptr<Array<unsigned int> > m_init_group;	

		ReactionMode m_reaction_mode;
		bool m_set_max_cris;
		bool m_set_exch_pr;
		bool m_setPr;		
		bool m_init_ex_point;
		bool m_init_cris;
		bool m_mode_set;
		unsigned int m_period_R;
		unsigned int m_nm;
		boost::shared_ptr<Array<unsigned int> > m_ninit;   // m_ninit[0]-num of initiator or anchor;
		unsigned int m_nkinds;
		unsigned int m_max_added_bonds;
		unsigned int m_new_bond_type;
		unsigned int m_new_angle_type;
		bool m_bond_type_by_pair;
		bool m_angle_type_by_pair;
		boost::shared_ptr<Array<unsigned int> > m_bond_type_table;
		boost::shared_ptr<Array<unsigned int> > m_angle_type_table;
		unsigned int m_generate_angle;
		boost::shared_ptr<Array<unsigned int> > m_change_type;
		Real m_angle_limit;
	};

void export_PolymerizationDM();

#endif

