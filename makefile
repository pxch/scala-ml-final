CC=g++
CFLAGS=-c -O3 -Wall -fopenmp -IEigen

all: mseigs clean

mseigs: main.o Lanczos.o
	$(CC) -fopenmp main.o Lanczos.o -O3 -o mseigs

main.o: src/main.cpp
	$(CC) $(CFLAGS) src/main.cpp

Lanczos.o: src/Lanczos.cpp src/Lanczos.h
	$(CC) $(CFLAGS) src/Lanczos.cpp

clean:
	rm -rf *o

