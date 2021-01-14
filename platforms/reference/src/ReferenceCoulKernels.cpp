#include "ReferenceCoulKernels.h"
#include "CoulForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include "ReferenceForce.h"
#include <cmath>

using namespace OpenMM;
using namespace std;
using namespace CoulPlugin;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->positions);
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->forces);
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

ReferenceCalcCoulForceKernel::~ReferenceCalcCoulForceKernel() {
}

static double getEwaldParamValue(int kmax, double width, double alpha){
    double temp = kmax * M_PI / (width * alpha);
    return 0.05 * sqrt(width * alpha) * kmax * exp(- temp * temp);
}

void ReferenceCalcCoulForceKernel::initialize(const System& system, const CoulForce& force) {
    int numParticles = system.getNumParticles();
    charges.resize(numParticles);
    for(int i=0;i<numParticles;i++){
        charges[i] = force.getParticleCharge(i);
    }
    for(int i=0;i<force.getNumExceptions();i++){
        int p1, p2;
        force.getExceptionParameters(i, p1, p2);
        pair<int,int> expair(p1, p2);
        exclusions.push_back(expair);
    }
    cutoff = force.getCutoffDistance();
    ewaldTol = force.getEwaldErrorTolerance();
    ifPBC = force.usesPeriodicBoundaryConditions();
    if (ifPBC){
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        alpha = (1.0/cutoff)*sqrt(-log(2.0*ewaldTol));
        kmaxx = 0;
        while (getEwaldParamValue(kmaxx, boxVectors[0][0], alpha) > ewaldTol){
            kmaxx += 1;
        }
        kmaxy = 0;
        while (getEwaldParamValue(kmaxy, boxVectors[1][1], alpha) > ewaldTol){
            kmaxy += 1;
        }
        kmaxz = 0;
        while (getEwaldParamValue(kmaxz, boxVectors[2][2], alpha) > ewaldTol){
            kmaxz += 1;
        }
        if (kmaxx%2 == 0)
            kmaxx += 1;
        if (kmaxy%2 == 0)
            kmaxy += 1;
        if (kmaxz%2 == 0)
            kmaxz += 1;
    }
}

double ReferenceCalcCoulForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& forces = extractForces(context);
    Vec3* box = extractBoxVectors(context);
    int numParticles = charges.size();
    double energy = 0.0;    
    double dEdR;
    vector<double> deltaR;
    deltaR.resize(5);
    if (!ifPBC){
        // noPBC, calcall
        for(int ii=0;ii<numParticles;ii++){
            for(int jj=ii+1;jj<numParticles;jj++){
                ReferenceForce::getDeltaR(pos[ii], pos[jj], &deltaR[0]);
                double inverseR = 1.0 / deltaR[4];
                if (includeEnergy) {
                    energy += ONE_4PI_EPS0*charges[ii]*charges[jj]*inverseR;
                }
                if (includeForecs) {
                    dEdR = ONE_4PI_EPS0*charges[ii]*charges[jj]*inverseR*inverseR*inverseR;
                    for(int dd=0;dd<3;dd++){
                        forces[ii][dd] += dEdR*deltaR[dd];
                        forces[jj][dd] -= dEdR*deltaR[dd];
                    }
                }
            }
        }
        // calc exclusions
        for(int ii=0;ii<exclusions.size();ii++){
            int idx1 = exclusions[ii].first;
            int idx2 = exclusions[ii].second;
            ReferenceForce::getDeltaR(pos[idx1], pos[idx2], &deltaR[0]);
            double inverseR = 1.0 / deltaR[4];
            if (includeEnergy) {
                energy -= ONE_4PI_EPS0*charges[idx1]*charges[idx2]*inverseR;
            }
            if (includeForces) {
                dEdR = ONE_4PI_EPS0*charges[idx1]*charges[idx2]*inverseR*inverseR*inverseR;
                for(int dd=0;dd<3;dd++){
                    forces[idx1][dd] -= dEdR*deltaR[dd];
                    forces[idx2][dd] += dEdR*deltaR[dd];
                }
            }
        }
    } else {
        // PBC
        // calc reciprocal part

        // calc bonded part

        // calc exclusion part
    }
    return energy;
}