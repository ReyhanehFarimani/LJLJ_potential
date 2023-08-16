/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Contributing author: Axel Kohlmeyer, Temple University, akohlmey@gmail.com

#include "pair_lj_lj.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJLJ::PairLJLJ(LAMMPS *lmp) : Pair(lmp) 
{
  writedata = 1;
}

/* ---------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
   desoroy all arrays
------------------------------------------------------------------------- */

PairLJLJ::~PairLJLJ()
{
  if (allocated) 
  {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(cut_sq);
    memory->destroy(cut_inner);
    memory->destroy(cut_inner_sq); //cut_inner[i][j]*cut_inner[i][j]
    memory->destroy(epsilon);
    memory->destroy(sigma);

    memory->destroy(lambda_p);

    memory->destroy(lj1); //48.0 * epsilon[i][j] * pow(sigma[i][j],12.0)
    memory->destroy(lj2); //24.0 * epsilon[i][j] * pow(sigma[i][j],6.0)
    memory->destroy(lj3); //4.0 * epsilon[i][j] * pow(sigma[i][j],12.0)
    memory->destroy(lj4); //4.0 * epsilon[i][j] * pow(sigma[i][j],6.0)

  }
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */
void PairLJLJ::allocate()
{
    allocated = 1;
    int n = atom->ntypes;
    int i, j; //iteration dummy variable
    memory->create(setflag, n+1, n+1, "pair:setflag");
    for (i = 0; i < n+1 ; i++)
    {
        for (j = 0; j< n+1; j++)
            setflag[i][j] = 0;
    }

    memory->create(cut, n+1, n+1, "pair:cut");
    memory->create(cut_sq, n+1, n+1, "pair:cut_sq");
    memory->create(cut_inner, n+1, n+1, "pair:cut_inner");
    memory->create(cut_inner_sq, n+1, n+1, "pair:cut_inner_sq");

    memory->create(epsilon, n+1, n+1, "pair:epsilon");
    memory->create(sigma, n+1, n+1, "pair:sigma");

    memory->create(lj1, n+1, n+1, "pair:lj1");
    memory->create(lj2, n+1, n+1, "pair:lj2");
    memory->create(lj3, n+1, n+1, "pair:lj3");
    memory->create(lj4, n+1, n+1, "pair:lj4");

    memory->create(lambda_p, n+1, n+1, "pair:lambda_p");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJLJ::settings(int narg, char **arg)
{
    //check for the number of parsed arguments
    if (narg != 3) 
        error->all(FLERR, "Illegal pair_style command");

    // getting the required arguments
    cut_inner_global = utils::numeric(FLERR,arg[0],false,lmp);
    cut_global = utils::numeric(FLERR,arg[1],false,lmp);
    lambda_param = utils::numeric(FLERR,arg[2],false,lmp);

    //check for the correct amount of arguments:
    if (cut_inner_global <= 0.0 || cut_inner_global > cut_global)
        error->all(FLERR,"Illegal pair_style command");

    // reset cutoffs that have been explicitly set
    if (allocated)
    {
        int i, j;
        for (i = 1; i <= atom->ntypes; i++)
        {
            for (j = 1; j <= atom->ntypes; j++)
            {
                if(setflag[i][j])
                {
                    cut_inner[i][j] = cut_inner_global;
                    cut[i][j] = cut_global;
                }
            }
        }
    }
}
/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJLJ::coeff(int narg, char **arg)
{
    if (narg < 4 || narg > 5) 
        error->all(FLERR, "Incorrect args for pair coefficients");
    if (!allocated) 
        allocate();

    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
    utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

    double epsilon_one = utils::numeric(FLERR,arg[2],false,lmp);
    double sigma_one = utils::numeric(FLERR,arg[3],false,lmp);
    double lambda_p_one = utils::numeric(FLERR, arg[4], false, lmp);

    double cut_inner_one = cut_inner_global;
    double cut_one = cut_global;

    if (narg == 7)
    {
        cut_inner_one = utils::numeric(FLERR, arg[5], false, lmp);
        cut_one = utils::numeric(FLERR, arg[6], false, lmp);
    }

    if (cut_inner_global <= 0.0 || cut_inner_global > cut_global)
        error->all(FLERR,"Illegal pair_style command");

    int count = 0;
    for (int i = ilo; i <= ihi; i++) 
    {
        for (int j = MAX(jlo,i); j <= jhi; j++) 
        {
            epsilon[i][j] = epsilon_one;
            sigma[i][j] = sigma_one;
            lambda_p[i][j] = lambda_p_one;
            cut_inner[i][j] = cut_inner_one;
            cut[i][j] = cut_one;
            setflag[i][j] = 1;
            count++;
        }
    }

    if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");    
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
double PairLJLJ::init_one(int i, int j)
{
    if (setflag[i][j] == 0)
    {
        epsilon[i][j] = mix_energy(epsilon[i][i], epsilon[j][i],
                        sigma[i][i], sigma[j][i]);
        sigma[i][j] = mix_distance(sigma[i][i], sigma[j][j]);
        cut_inner[i][j] = mix_distance(cut_inner[i][i], cut_inner[j][j]);
        cut[i][j] = mix_distance(cut[i][i], cut[j][j]);

    }

    cut_inner_sq[i][j] = cut_inner[i][j]*cut_inner[i][j];
    cut_sq[i][j] = cut[i][j]*cut[i][j];
    lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
    lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
    lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
    lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

    cut[j][i] = cut[i][j];
    cut_inner[j][i] = cut_inner[i][j];
    cut_inner_sq[j][i] = cut_inner_sq[i][j];
    cut_sq[j][i] = cut_sq[i][j];
    lj1[j][i] = lj1[i][j];
    lj2[j][i] = lj2[i][j];
    lj3[j][i] = lj3[i][j];
    lj4[j][i] = lj4[i][j];

    return cut[i][j];    
}   


/* ----------------------------------------------------------------------
 	
force/r and energy of a single pairwise interaction between pair i,j and corresponding j,i
------------------------------------------------------------------------- */

// double PairLJLJ::single(int /*i*/, int /*j*/, int itype, int jtype,
//                              double rsq,
//                              double /*factor_coul*/, double factor_lj,
//                              double &fforce)
// {
//     double r2inv, r6inv, florcelj, philj;
//     double rr, dp, d, tt, dt, dd;

//     r2inv = 1.0/rsq;
//     r6inv = r2inv * r2inv * r2inv;


// }
/* ---------------------------------------------------------------------- */

void PairLJLJ::compute(int eflag, int vflag)
{
    int i, j, ii, jj, inum, jnum, itype, jtype;
    double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
    double rsq, r2inv, r3inv, r6inv, forcelj, factor_lj;
    int *ilist, *jlist, *numneigh, **firstneigh;

    evdwl = 0.0;
    ev_init(eflag, vflag);

    double **xx = atom->x;
    double **ff = atom->f;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    double *special_lj = force->special_lj;
    int newton_pair = force->newton_pair;

    double rr, d, dd, tt, dt, dp, philj;

    inum = list->inum; //number of atoms
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh= list->firstneigh;

    //loop over neighboers of the simulation atoms:
    for (ii = 0; ii < inum; ii++)
    {
        i = ilist[ii];
        xtmp = xx[i][0];
        ytmp = xx[i][1];
        ztmp = xx[i][2];
        itype = type[i];
        jlist = firstneigh[i]; // fist neighbor list of atom "i"
        jnum = numneigh[i];

        // loop over the neighborhood
        for (jj = 0; jj < jnum; jj++)
        {
            j = jlist[jj];
            factor_lj = special_lj[sbmask(j)];
            j &= NEIGHMASK; //Due to the additional bits, the value of j would be out of range 
                            // when accessing data from per-atom arrays, 
                            // so we apply the NEIGHMASK constant 
                            // with a bit-wise and operation to mask them out. This step must be done, even if a pair style
                            // does not use special bond scaling of forces and energies to avoid segmentation faults.

            //computing the dist:
            delx = xtmp - xx[j][0];
            dely = ytmp - xx[j][1];
            delz = ztmp - xx[j][2];
            rsq = delx*delx + dely*dely + delz*delz;
            jtype = type[j];

            if (rsq < cut_inner_sq[itype][jtype])
            {
                r2inv = 1.0/rsq;
                r6inv = r2inv*r2inv*r2inv;
                forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
                fpair = factor_lj * forcelj * r2inv;

                ff[i][0] += delx * fpair;
                ff[i][1] += dely * fpair;
                ff[i][2] += delz * fpair;
                if (newton_pair || j < nlocal) 
                {
                    ff[j][0] -= delx * fpair;
                    ff[j][1] -= dely * fpair;
                    ff[j][2] -= delz * fpair;
                }

                    if (eflag) 
                    {
                        evdwl = r6inv * (lj3[itype][jtype]*r6inv - lj4[itype][jtype]) + epsilon[itype][jtype] * (1 - lambda_p[itype][jtype]);
                        evdwl *= factor_lj;

                        if (evflag) ev_tally(i,j,nlocal,newton_pair,evdwl,0.0,fpair,delx,dely,delz);
                    }
            }

            else if (rsq < cut_sq[itype][jtype])
            {
                r2inv = 1.0/rsq;
                r6inv = r2inv*r2inv*r2inv;
                forcelj = lambda_p[itype][jtype] * r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
                fpair = factor_lj * forcelj * r2inv;

                ff[i][0] += delx * fpair;
                ff[i][1] += dely * fpair;
                ff[i][2] += delz * fpair;

                if (newton_pair || j < nlocal) 
                {
                    ff[j][0] -= delx * fpair;
                    ff[j][1] -= dely * fpair;
                    ff[j][2] -= delz * fpair;
                }

                    if (eflag) 
                    {
                        evdwl = r6inv * (lj3[itype][jtype]*r6inv - lj4[itype][jtype]) * lambda_p[itype][jtype];
                        evdwl *= factor_lj;
                        if (evflag) ev_tally(i,j,nlocal,newton_pair,evdwl,0.0,fpair,delx,dely,delz);
                    }
            }
            

        }
    }
}

/*---------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
--------------------------------------------------------------------------*/


//-------- Writing and Reading Restrat files-----------//

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJLJ::write_restart(FILE *fp)
{
    write_restart_settings(fp);

    int i, j;
    for(i = 1 ; i <atom->ntypes; i++)
    {
        for (j = 1; j<atom->ntypes; j++)
        {
            fwrite(&setflag[i][j], sizeof(int), 1, fp);
            if (setflag[i][j])
            {
                fwrite(&epsilon[i][j], sizeof(double), 1, fp);
                fwrite(&sigma[i][j], sizeof(double), 1, fp);
                fwrite(&cut_inner[i][j], sizeof(double), 1, fp);
                fwrite(&cut[i][j], sizeof(double), 1, fp);
                fwrite(&lambda_p[i][j], sizeof(double), 1, fp);
            }
        }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJLJ::read_restart(FILE *fp)
{
    read_restart_settings(fp);
    allocate();

    int i, j;
    int me = comm->me;

    for (i = 0 ; i<= atom->ntypes; i++)
    {
        for (j = 0 ; j<= atom->ntypes; j++)
        {
            if (me == 0)
                utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
            MPI_Bcast(&setflag[i][j] ,1 , MPI_INT, 0 , world);
            if (setflag[i][j])
            {
                if (me ==0)
                {
                    utils::sfread(FLERR,&epsilon[i][j],sizeof(double),1,fp,nullptr,error);
                    utils::sfread(FLERR,&sigma[i][j],sizeof(double),1,fp,nullptr,error);
                    utils::sfread(FLERR,&cut_inner[i][j],sizeof(double),1,fp,nullptr,error);
                    utils::sfread(FLERR,&cut[i][j],sizeof(double),1,fp,nullptr,error);
                    utils::sfread(FLERR,&lambda_p[i][j],sizeof(double),1,fp,nullptr,error);
                }

                MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
                MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
                MPI_Bcast(&cut_inner[i][j],1,MPI_DOUBLE,0,world);
                MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
                MPI_Bcast(&lambda_p[i][j],1,MPI_DOUBLE,0,world);
            }
        } 
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJLJ::write_restart_settings(FILE *fp)
{
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJLJ::read_restart_settings(FILE *fp)
{
    int me = comm->me;
    if (me == 0)
    {
        utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
    }
    MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLJLJ::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,epsilon[i][i],sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLJLJ::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g\n",
              i,j,epsilon[i][j],sigma[i][j],
              cut_inner[i][j],cut[i][j], lambda_p[i][j]);
}


void *PairLJLJ::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  return nullptr;
}
