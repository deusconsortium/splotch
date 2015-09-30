#include <iostream>
#include <vector>
#include <string>
#include <cassert>
//#include <unordered_map>

#include "omp.h"
#include "cxxsupport/arr.h"
#include "cxxsupport/cxxutils.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"
#include "reader/FOFReaderLib/FOFReaderLib/FOFReaderLib.h"

using namespace std;

namespace
{

    void fofcube_reader_finish(vector<particle_sim> &points)
    {
        float minr = 1e30;
        float minx = 1e30;
        float miny = 1e30;
        float minz = 1e30;
        float maxr = -1e30;
        float maxx = -1e30;
        float maxy = -1e30;
        float maxz = -1e30;
        for (tsize i = 0; i < points.size(); ++i) {
            //points[i].active = 1;
            points[i].type = 0;
            minr = min(minr, points[i].r);
            maxr = max(maxr, points[i].r);
            minx = min(minx, points[i].x);
            maxx = max(maxx, points[i].x);
            miny = min(miny, points[i].y);
            maxy = max(maxy, points[i].y);
            minz = min(minz, points[i].z);
            maxz = max(maxz, points[i].z);
        }
        mpiMgr.allreduce(maxr, MPI_Manager::Max);
        mpiMgr.allreduce(minr, MPI_Manager::Min);
        mpiMgr.allreduce(maxx, MPI_Manager::Max);
        mpiMgr.allreduce(minx, MPI_Manager::Min);
        mpiMgr.allreduce(maxy, MPI_Manager::Max);
        mpiMgr.allreduce(miny, MPI_Manager::Min);
        mpiMgr.allreduce(maxz, MPI_Manager::Max);
        mpiMgr.allreduce(minz, MPI_Manager::Min);
        //#ifdef DEBUG
        cout << "MIN, MAX --> " << minr << " " << maxr << endl;
        cout << "MIN, MAX --> " << minx << " " << maxx << endl;
        cout << "MIN, MAX --> " << miny << " " << maxy << endl;
        cout << "MIN, MAX --> " << minz << " " << maxz << endl;
        //#endif
    }

    void normalize(vector<particle_sim> &points)
    {
        // normalization
        float maxExtent, maxExtentX, maxExtentY, maxExtentZ;
        float decX, decY, decZ;
        float echelle;

        float minX = 10.0, maxX = -10.0, minY = 10.0, maxY = -10.0, minZ = 10.0, maxZ = -10.0;

        for (unsigned long i = 0; i < points.size(); i++) {
            minX = min(minX, points[i].x);
            maxX = max(maxX, points[i].x);
            minY = min(minY, points[i].y);
            maxY = max(maxY, points[i].y);
            minZ = min(minZ, points[i].z);
            maxZ = max(maxZ, points[i].z);
        }

        maxExtentX = maxX - minX;
        maxExtentY = maxY - minY;
        maxExtentZ = maxZ - minZ;
        maxExtent = std::max(maxExtentX, std::max(maxExtentY, maxExtentZ));

        echelle = 1.0f / maxExtent;

        std::cout << "Echelle = " << echelle << std::endl;

        decX = (maxExtent - maxExtentX) / 2;
        decY = (maxExtent - maxExtentY) / 2;
        decZ = (maxExtent - maxExtentZ) / 2;

        std::cout
                << "minX=" << minX << ", maxX=" << maxX
                << " | minY=" << minY << ", maxY=" << maxY
                << " | minZ=" << minZ << ", maxZ=" << maxZ
                << std::endl;

        float minX2 = 10.0, maxX2 = -10.0, minY2 = 10.0, maxY2 = -10.0, minZ2 = 10.0, maxZ2 = -10.0;

        // convert
        for (unsigned long i = 0; i < points.size(); i++) {
            points[i].x = echelle * (points[i].x - minX + decX) - 0.5f;
            minX2 = min(minX2, points[i].x);
            maxX2 = max(maxX2, points[i].x);
            points[i].y = echelle * (points[i].y - minY + decY) - 0.5f;
            minY2 = min(minY2, points[i].y);
            maxY2 = max(maxY2, points[i].y);
            points[i].z = echelle * (points[i].z - minZ + decZ) - 0.5f;
            minZ2 = min(minZ2, points[i].z);
            maxZ2 = max(maxZ2, points[i].z);
        }
        std::cout
                << "minX=" << minX2 << ", maxX=" << maxX2
                << " | minY=" << minY2 << ", maxY=" << maxY2
                << " | minZ=" << minZ2 << ", maxZ=" << maxZ2
                << std::endl;
        //std::cout << "Nb part avant = " << _npart << " | size = " << sizeof( Particle ) << std::endl;
    }
}

void fofcube_reader(paramfile &params, std::vector<particle_sim> &points, double &boxsize)
{
    unsigned long npart = 0;
    float *boundaries;
    string rootfile;

    boxsize = 1.0f;

    boundaries = new float[6];

    float minH = 50.0f, maxH = -50.0f;
    float minX = 50.0f, maxX = -50.0f;
    float minY = 50.0f, maxY = -50.0f;
    float minZ = 50.0f, maxZ = -50.0f;

    npart = 0;
    long int curr = 0;

    std::cout << "Max theads = " << omp_get_max_threads() << std::endl;
            
    
    // clear the points
    points.clear();

    std::cout << "Halos path = " << params.find<string>("infile", "") << std::endl
            << "Cubes path = " << params.find<string>("infile_halos", "") << std::endl;

    DEUSSimulationSingleton *mySimulation = DEUSSimulationSingleton::getInstance(
                                                                                 params.find<string>("infile", ""),
                                                                                 params.find<string>("infile_halos", ""));

    std::cout << "Simulation initialised" << std::endl;

    DEUSArea *area;

    if (params.find<float>("maxDOF", 0) > 0) {
        area = new DEUSArea(
                            params.find<float>("camera_x", 0.5),
                            params.find<float>("camera_y", 0.5),
                            params.find<float>("camera_z", 1.5),
                            params.find<float>("lookat_x", 0.5),
                            params.find<float>("lookat_y", 0.5),
                            params.find<float>("lookat_z", 0.5),
                            params.find<float>("fov", 40),
                            params.find<float>("maxDOF", 0),
                            1.0
                            );
    }
    else {
        area = new DEUSArea(
                            params.find<float>("minX", 0),
                            params.find<float>("maxX", 1),
                            params.find<float>("minY", 0),
                            params.find<float>("maxY", 1),
                            params.find<float>("minZ", 0),
                            params.find<float>("maxZ", 1),
                            1.0f);
    }

    int resolution = params.find<float>("resolution", 1024.0);

    std::cout << "area= "
            << "(" << area->coords(0, 0, 0) << "," << area->coords(0, 1, 0) << "," << area->coords(0, 2, 0) << ") "
            << "to (" << area->coords(0, 0, 1) << "," << area->coords(0, 1, 1) << "," << area->coords(0, 2, 1) << ") "
            << std::endl << " and "
            << "(" << area->coords(1, 0, 0) << "," << area->coords(1, 1, 0) << "," << area->coords(1, 2, 0) << ") "
            << "to (" << area->coords(1, 0, 1) << "," << area->coords(1, 1, 1) << "," << area->coords(1, 2, 1) << ") "
            << std::endl;

    std::map<long long, char> haloparticles; // = {};
    
    // Associated struct ?

    int nbHalos = 0;

    if (false) { //mySimulation->halos() > 0) {

        //std::cout << "loading " << rootfile << std::endl;
        DEUSHalos *halos = mySimulation->halos();

        std::vector<long long int> *idparticlesperhalo = new std::vector<long long int>[halos->nHalos()];
        
        //#pragma omp parallel for num_threads(64)
        for (int n = 0; n < halos->nHalos(); n++) {

            float x = halos->halos(n)->x();
            float y = halos->halos(n)->y();
            float z = halos->halos(n)->z();

            if (area->particuleIsInside(x, y, z)) {

                //std::cout << "loading particles from " << halos.filename(halos.halos(n)->fileindex()) << " to get halo= " << n << std::endl;

                halos->loadParticles(n, FOFParticles::READ_IDS);
                FOFParticles *halo = halos->halos(n)->particles();
                
                //idparticlesperhalo[n] = halos->halos(n)->particles()->id();
                 
                for (int j = 0; j < halo->npart(); j++) {
                    haloparticles.insert(std::pair<long long, char>(halo->id(j), 1));
                }                
                delete halo;
                nbHalos++;
            }
        }
        
        /*std::cout << "Merging..." << std::endl;
        for (int n = 0; n < halos->nHalos(); n++) {
            for (int j = 0; j < idparticlesperhalo[n].size(); j++) {
                haloparticles.insert(std::pair<long long, char>(idparticlesperhalo[n][j], 1));
            }
        }*/

        // END READ STRUCT
        std::cout << "Ok, read " << nbHalos << " halos" << std::endl;

    }

    ///////////////////////

    if (mySimulation->cubes() > 0) {

        //std::cout << "loading " << rootfile << std::endl;

        FOFMultiCube *multi = mySimulation->cubes();
        
        std::vector<particle_sim> *partperfile = new std::vector<particle_sim>[multi->nCubes()];
        
        //long long int total_npart = 0;
        float radius = 1.0f / resolution / 2.0f;

        int nCubes = multi->nCubes();
        
        int curr2 = 0;//npart;
        int n = 0;
        
        #pragma omp parallel for schedule(dynamic,1) num_threads(64) private(n,curr2)
        for (n = 0; n < nCubes; n++) { // 704

            if (area->cubeIntersect(multi->cubes(n)->boundaries())) {

                if (multi->cubes(n)->position().size() == 0) {
                    std::cout << "reading " << n << std::endl;
                    multi->cubes(n)->readParticles(FOFParticles::READ_ALL);
                    std::cout << "Ok, read " << multi->cubes(n)->npart() << " particles" << std::endl;
                }
                else {
                    std::cout << "get from cache " <<  n << " with " << multi->cubes(n)->npart() << " particles" << std::endl;
                }
                
                curr2 = 0;
                
                //npart += multi->cubes(n)->npart();
                //points.resize(npart);
                
                partperfile[n].reserve(multi->cubes(n)->npart());
                
                for (long int j = 0; j < multi->cubes(n)->npart(); j++) {

                    float x = multi->cubes(n)->posX(j);
                    float y = multi->cubes(n)->posY(j);
                    float z = multi->cubes(n)->posZ(j);

                    if (area->particuleIsInside(x, y, z)) {

                        bool isHalo = true;// haloparticles.find(multi->cubes(n)->id(j)) != haloparticles.end(); //(haloparticles[multi->cubes(n)->id(j)] == 1);
                        
                        partperfile[n][curr2].x = x;
                        partperfile[n][curr2].y = y;
                        partperfile[n][curr2].z = z;
                        partperfile[n][curr2].I = 1.0f;
                        partperfile[n][curr2].active = true;
                        partperfile[n][curr2].e.r = 1000000.0f * multi->cubes(n)->velNorm(j);
                        partperfile[n][curr2].e.g = NULL;
                        partperfile[n][curr2].e.b = NULL;
                        partperfile[n][curr2].r = isHalo ? radius / 4.0f : radius;
                        partperfile[n][curr2].type = isHalo ? 1 : 0;
                        if (partperfile[n][curr2].x < minX) {
                            minX = partperfile[n][curr2].x;
                        }
                        if (partperfile[n][curr2].x > maxX) {
                            maxX = partperfile[n][curr2].x;
                        }
                        if (partperfile[n][curr2].y < minY) {
                            minY = partperfile[n][curr2].y;
                        }
                        if (partperfile[n][curr2].y > maxY) {
                            maxY = partperfile[n][curr2].y;
                        }
                        if (partperfile[n][curr2].z < minZ) {
                            minZ = partperfile[n][curr2].z;
                        }
                        if (partperfile[n][curr2].z > maxZ) {
                            maxZ = partperfile[n][curr2].z;
                        }
                        if (partperfile[n][curr2].e.r < minH) {
                            minH = partperfile[n][curr2].e.r;
                        }
                        if (partperfile[n][curr2].e.r > maxH) {
                            maxH = partperfile[n][curr2].e.r;
                        }
                        curr2++;
                    }
                }
                npart = curr2;
                
                //points.resize(npart);
                partperfile[n].resize(curr2);
                
                //total_npart += curr2;
                
                std::cout << "Cube " << n << " done (" << curr2 << ") parts" << std::endl;
            }
            else {
                if (multi->cubes(n)->position().size() > 0) {
                    std::cout << "releasing " << n << std::endl;
                    multi->cubes(n)->releaseParticles();
                }
            }
        }
        std::cout << "Merging..." << std::endl;
        // Merge results
        //points.reserve(total_npart);
        for (int i = 0; i < multi->nCubes(); i++) {            
            points.insert(points.end(),partperfile[i].begin(),partperfile[i].end());
        }

        // END READ CUBE

        std::cout << "Ok, read " << points.size() << "/" << npart << " particles." << std::endl;
        std::cout << "minH= " << minH << " maxH= " << maxH << std::endl;
        std::cout
                << "minX=" << minX << ", maxX=" << maxX
                << " | minY=" << minY << ", maxY=" << maxY
                << " | minZ=" << minZ << ", maxZ=" << maxZ
                << std::endl;
    }
}

//normalize(points);




