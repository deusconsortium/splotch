#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cassert>
#include <unordered_map>

//#include "mpi.h"
#include "cxxsupport/arr.h"
#include "cxxsupport/cxxutils.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"
#include "reader/FOFReaderLib/FOFReaderLib/FOFReaderLib.h"
#include "FOFReaderLib/FOFReaderLib/FOFFiles/FOFCube.h"

using namespace std;


namespace
{
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
            
            points[i].r *= echelle;
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

    float scale = params.find<float>("scale", 1);
    
    std::cout << "IT'S THE RIGHT FILE ! " << std::endl;
        
    // clear the points
    points.clear();

    std::cout << "Cube path = " << params.find<string>("infile", "") << std::endl
            << "Halo path = " << params.find<string>("infile_halos", "") << std::endl;

    DEUSSimulationSingleton *mySimulation = DEUSSimulationSingleton::getInstance(
                                                                                 params.find<string>("infile", ""),
                                                                                 params.find<string>("infile_halos", ""));

    std::cout << "Simulation initialised" << std::endl;

    DEUSArea *area;

    if (params.find<float>("maxDOF", 0) > 0) {
        area = new DEUSArea(
                            params.find<float>("camera_x", 50) / scale,
                            params.find<float>("camera_y", 50) / scale,
                            params.find<float>("camera_z", 150) / scale,
                            params.find<float>("lookat_x", 50) / scale,
                            params.find<float>("lookat_y", 50) / scale,
                            params.find<float>("lookat_z", 50) / scale,
                            params.find<float>("fov", 40),
                            params.find<float>("maxDOF", 0) / scale,
                            1.0
                            );
    }
    else {
        area = new DEUSArea(
                            params.find<float>("minX", 0) / scale,
                            params.find<float>("maxX", scale) / scale,
                            params.find<float>("minY", 0) / scale,
                            params.find<float>("maxY", scale) / scale,
                            params.find<float>("minZ", 0) / scale,
                            params.find<float>("maxZ", scale) / scale,
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
    
    std::cout << "shift= "
            << "(" << area->shift(0, 0) << "," << area->shift(0, 1) << "," << area->shift(0, 2) << ") "            
            << std::endl << " and "
            << "(" << area->shift(1, 0) << "," << area->shift(1, 1) << "," << area->shift(1, 2) << ") "            
            << std::endl;

    std::unordered_map<long long, char> haloparticles = {};

    // Associated struct ?

    int nbHalos = 0;

    if (mySimulation->halos() > 0) {
        
        std::cout << "Generating halos hashmap..." << std::endl;

        //std::cout << "loading " << rootfile << std::endl;
        DEUSHalos *halos = mySimulation->halos();

        std::vector<long long int> *idparticlesperhalo = new std::vector<long long int>[halos->nHalos()];
        
        #pragma omp parallel for num_threads(64) schedule(dynamic,1)
        for (int n = 0; n < halos->nHalos(); n++) {

            float x = halos->halos(n)->x();
            float y = halos->halos(n)->y();
            float z = halos->halos(n)->z();

            if (area->particuleIsInside(x, y, z)) {

                //std::cout << "loading particles from " << halos.filename(halos.halos(n)->fileindex()) << " to get halo= " << n << std::endl;

                halos->loadParticles(n, FOFParticles::READ_IDS);
                FOFParticles *halo = halos->halos(n)->particles();
                
                idparticlesperhalo[n] = halos->halos(n)->particles()->id();
                 
                /*#pragma omp critical(hashinsert)
                for (int j = 0; j < halo->npart(); j++) {
                    haloparticles.insert(std::pair<long long, char>(halo->id(j), 1));
                    //haloparticles.emplace (halo->id(j), 1);
                } */              
                delete halo;
                nbHalos++;
            }
        }
        
        std::cout << "Merging..." << std::endl;
        int hashsize = 0;
        for (int n = 0; n < halos->nHalos(); n++) {
            hashsize += idparticlesperhalo[n].size();            
        }
        haloparticles.rehash(hashsize); // Reserve the needed space*/
        for (int n = 0; n < halos->nHalos(); n++) {
            for (int j = 0; j < idparticlesperhalo[n].size(); j++) {
                haloparticles.insert(std::pair<long long, char>(idparticlesperhalo[n][j], 1));
            }
        }
        delete[] idparticlesperhalo;

        // END READ STRUCT
        std::cout << "Ok, read " << nbHalos << " halos" << std::endl;

    }
    
    // Management for "special" particles
    for(int n=2; n<99; n++) {
        char specialfilename[100];
        sprintf(specialfilename,"infile_special%d",n);
        if(params.find<string>(specialfilename, "") != "") {
            std::cout << "Ok, read " << params.find<string>(specialfilename, "") << " special ids for " << n << std::endl;
            std::ifstream infile(params.find<string>(specialfilename, ""));
            long long id;
            while (infile >> id)
            {
              haloparticles.insert(std::pair<long long, char>(id, n));
            }
        }
        
    }
    
    
    ///////////////////////

    if (mySimulation->cubes() > 0) {

        //std::cout << "loading " << rootfile << std::endl;
        
        float radius = params.find<float>("force_radius", 0.0);
        if(radius == 0.0) {
            radius = scale / resolution / 2.0f; // / 2.0f; // / 2.0f; /// (2.0f);// / (2.0f * 2.0f * 2.0f);  // = scale/res / refine_moy
        }

        FOFMultiCube *multi = mySimulation->cubes();
        
        std::vector<particle_sim> *partperfile = new std::vector<particle_sim>[multi->nCubes()];
        
        #pragma omp parallel for num_threads(64) schedule(dynamic,1)
        for (int n = 0; n < multi->nCubes(); n++) { // 704

            if (area->cubeIntersect(multi->cubes(n)->boundaries())) {
                
                if (multi->cubes(n)->position().size() == 0) {
                    std::cout << "reading " << n << std::endl;
                    multi->cubes(n)->readParticles(FOFParticles::READ_ALL);
                    std::cout << "Ok, read " << multi->cubes(n)->npart() << " particles" << std::endl;
                }
                else {
                    std::cout << "get from cache " << n << std::endl;
                }
                
                //std::cout << "Y (" << multi->cubes(n)->minY() << "," << multi->cubes(n)->maxY() << ")" << std::endl;
                
                int curr2 = 0;//npart;
                //npart += multi->cubes(n)->npart();
                //points.resize(npart);
                
                partperfile[n].resize(multi->cubes(n)->npart());
                
                std::unordered_map<long long, char>::const_iterator specialParticleIterator;
                
                // #pragma omp parallel for
                for (long int j = 0; j < multi->cubes(n)->npart(); j++) {

                    float x = multi->cubes(n)->posX(j);
                    float y = multi->cubes(n)->posY(j);
                    float z = multi->cubes(n)->posZ(j);

                    if (area->particuleIsInside(x, y, z)) {

                        //bool isHalo = //haloparticles.find(multi->cubes(n)->id(j)) != haloparticles.end(); //false; //(haloparticles[multi->cubes(n)->id(j)] == 1);
                        int type = 0;
                        specialParticleIterator = haloparticles.find(multi->cubes(n)->id(j));
                        if(specialParticleIterator != haloparticles.end()) {
                            type = specialParticleIterator->second;
                        }
                        bool isHalo = (type == 1);
                        
                        partperfile[n][curr2].x = x * scale;
                        partperfile[n][curr2].y = y * scale;
                        partperfile[n][curr2].z = z * scale;
                        partperfile[n][curr2].I = 1.0f;
                        partperfile[n][curr2].active = true;
                        partperfile[n][curr2].e.r = 1000000.0f * multi->cubes(n)->velNorm(j);
                        partperfile[n][curr2].e.g = NULL;
                        partperfile[n][curr2].e.b = NULL;
                        partperfile[n][curr2].r = isHalo ? radius / 4.0f : radius;
                        partperfile[n][curr2].type = type;
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
        for (int i = 0; i < multi->nCubes(); i++) {            
            points.insert(points.end(),partperfile[i].begin(),partperfile[i].end());
        }
        delete[] partperfile;
        delete[] boundaries;
        delete area;
        

        // END READ CUBE

        std::cout << "Ok, read " << points.size() << "/" << npart << " particles." << std::endl;
        std::cout << "minH= " << minH << " maxH= " << maxH << std::endl;
        std::cout
                << "minX=" << minX << ", maxX=" << maxX
                << " | minY=" << minY << ", maxY=" << maxY
                << " | minZ=" << minZ << ", maxZ=" << maxZ
                << std::endl;

        if(params.find<bool>("normalize_coords", false)) {
            normalize(points);
        }
        
    }    
    
    
}






