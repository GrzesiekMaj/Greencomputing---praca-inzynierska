#!/bin/bash -l
## nazwa zlecenia
#SBATCH -J ADFtestjob
## Liczba alokowanych w�z��w
#SBATCH -N 1
## Liczba zada� per w�ze� (domy�lnie jest to liczba alokowanych rdzeni na w�le)
#SBATCH --ntasks-per-node=1
## Ilo�� pami�ci przypadaj�cej na jeden rdze� obliczeniowy (domy�lnie 5GB na rdze�)
#SBATCH --mem-per-cpu=5GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:10:00 
## Nazwa grantu do rozliczenia zu�ycia zasob�w
#SBATCH -A greencomputing2016
## Specyfikacja partycji
#SBATCH -p plgrid-gpu
## Plik ze standardowym wyj�ciem
#SBATCH --output="output.out"
## Plik ze standardowym wyj�ciem b��d�w
#SBATCH --error="error.err"
#SBATCH --partition=plgrid-gpu
#SBATCH --gres=gpu

module load plgrid/apps/cuda
ts=$(date +%s%N)
############ do ca�ki
g++ -I. -I$CUDADIR/include -L$CUDADIR/lib64 -lOpenCL -pthread -o program matrixmul_host-cpp.cpp
#nvidia-smi dmon -s pc -o DT &
./program
tt=$((($(date +%s%N) - $ts)))
echo "Time taken: $tt"
##gdb program core


############ do redukcji
#chmod 777 sumReductionGPU.c
#gcc -I. -I$CUDADIR/include -L$CUDADIR/lib64 -lOpenCL -pthread -o sumReductionGPU sumReductionGPU.c
#chmod 777 run_performance_sumGPU
#./run_performance_sumGPU

##cat ./performances.txt
#./program