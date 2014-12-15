all: mseigs

mseigs: src/mseigs.cpp src/GraphClustering.h src/Lanczos.cpp src/Lanczos.h src/mseigs.h
	icc -O2 -xhost -openmp -std=c++0x -IEigen -Imetis/include -Lmetis/lib -o mseigs src/mseigs.cpp src/Lanczos.cpp -lmetis

