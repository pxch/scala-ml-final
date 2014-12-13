all: clustering

clustering: test.cpp GraphClustering.h Lanczos.cpp Lanczos.h
	icc -O2 -xhost -openmp -std=c++0x -I../Eigen -I../metis/include -L../metis/lib -o clustering test.cpp Lanczos.cpp -lmetis

