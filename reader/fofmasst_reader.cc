#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include "cxxsupport/arr.h"
#include "cxxsupport/cxxutils.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"
#include "reader/FOFReaderLib/FOFReaderLib/FOFReaderLib.h"

using namespace std;

namespace {
    
  void fofmasst_reader_finish (vector<particle_sim> &points)
  {
  float minr=1e30;
  float minx=1e30;
  float miny=1e30;
  float minz=1e30;
  float maxr=-1e30;
  float maxx=-1e30;
  float maxy=-1e30;
  float maxz=-1e30;
  for (tsize i=0; i<points.size(); ++i)
    {
    //points[i].active = 1;
    points[i].type=0;
    minr = min(minr,points[i].r);
    maxr = max(maxr,points[i].r);
    minx = min(minx,points[i].x);
    maxx = max(maxx,points[i].x);
    miny = min(miny,points[i].y);
    maxy = max(maxy,points[i].y);
    minz = min(minz,points[i].z);
    maxz = max(maxz,points[i].z);
    }
  mpiMgr.allreduce(maxr,MPI_Manager::Max);
  mpiMgr.allreduce(minr,MPI_Manager::Min);
  mpiMgr.allreduce(maxx,MPI_Manager::Max);
  mpiMgr.allreduce(minx,MPI_Manager::Min);
  mpiMgr.allreduce(maxy,MPI_Manager::Max);
  mpiMgr.allreduce(miny,MPI_Manager::Min);
  mpiMgr.allreduce(maxz,MPI_Manager::Max);
  mpiMgr.allreduce(minz,MPI_Manager::Min);
//#ifdef DEBUG
  cout << "MIN, MAX --> " << minr << " " << maxr << endl;
  cout << "MIN, MAX --> " << minx << " " << maxx << endl;
  cout << "MIN, MAX --> " << miny << " " << maxy << endl;
  cout << "MIN, MAX --> " << minz << " " << maxz << endl;
//#endif
  }
  
  float normalize(vector<particle_sim> &points)
  {
    // normalization
    float maxExtent, maxExtentX, maxExtentY, maxExtentZ;
    float decX, decY, decZ;
    float echelle;
    
    float minX = 10.0, maxX=-10.0,minY=10.0,maxY=-10.0,minZ=10.0,maxZ=-10.0;

    for (long i = 0; i < points.size(); i++) {        
        minX = min(minX,points[i].x);
        maxX = max(maxX,points[i].x);        
        minY = min(minY,points[i].y);
        maxY = max(maxY,points[i].y);        
        minZ = min(minZ,points[i].z);
        maxZ = max(maxZ,points[i].z);
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

     float minX2 = 10.0, maxX2=-10.0,minY2=10.0,maxY2=-10.0,minZ2=10.0,maxZ2=-10.0;
        
   // convert
    for (long i = 0; i < points.size(); i++) {
        points[i].x = echelle * (points[i].x - minX + decX) - 0.5f;
        minX2 = min(minX2,points[i].x);
        maxX2 = max(maxX2,points[i].x);
        points[i].y = echelle * (points[i].y - minY + decY) - 0.5f;
        minY2 = min(minY2,points[i].y);
        maxY2 = max(maxY2,points[i].y);
        points[i].z = echelle * (points[i].z - minZ + decZ) - 0.5f;
        minZ2 = min(minZ2,points[i].z);
        maxZ2 = max(maxZ2,points[i].z);
    }
        std::cout 
                << "minX=" << minX2 << ", maxX=" << maxX2
                << " | minY=" << minY2 << ", maxY=" << maxY2
                << " | minZ=" << minZ2 << ", maxZ=" << maxZ2
                << std::endl;
    //std::cout << "Nb part avant = " << _npart << " | size = " << sizeof( Particle ) << std::endl;
        return echelle;
  }
  
}

void fofmasst_reader(paramfile &params, std::vector<particle_sim> &points, double &boxsize) {
    
    string datafile = params.find<string>("infile");
    
    boxsize = 1.0f;
        
    FOFMasst halos(datafile);
    
    DEUSArea *area;
    
    float scale = params.find<float>("scale", 1);
    
    // clear the points
    points.clear();

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
    
    points.resize(halos.nHalos());
    long int curr = 0;
    
    for (int n = 0; n < halos.nHalos(); n++) {

        float x = halos.halos(n)->x();
        float y = halos.halos(n)->y();
        float z = halos.halos(n)->z();

        if (area->particuleIsInside(x, y, z)) {
            points[curr].x = halos.halos(n)->x();
            points[curr].y = halos.halos(n)->y();
            points[curr].z = halos.halos(n)->z();
            points[curr].I = 1000.0;//halos.halos(n)->mass();
            points[curr].active = true;
            points[curr].e.r = 1.0f;
            points[curr].e.g = 1.0f;
            points[curr].e.b = 1.0f;
            points[curr].r = 0.75/6192.0;
            points[curr].type=0;
            curr++;            
        }
    }
        
    points.resize(curr);

    // END READ CUBE

    std::cout << "Ok, read " << curr << " halos." << std::endl;
        
}




           