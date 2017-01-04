#!/bin/bash -l
## nazwa zlecenia
#SBATCH -J ADFtestjob
## Liczba alokowanych wêz³ów
#SBATCH -N 1
## Liczba zadañ per wêze³ (domyœlnie jest to liczba alokowanych rdzeni na wêŸle)
#SBATCH --ntasks-per-node=1
## Iloœæ pamiêci przypadaj¹cej na jeden rdzeñ obliczeniowy (domyœlnie 5GB na rdzeñ)
#SBATCH --mem-per-cpu=5GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:10:00 
## Nazwa grantu do rozliczenia zu¿ycia zasobów
#SBATCH -A greencomputing2016
## Specyfikacja partycji
#SBATCH -p plgrid-gpu
## Plik ze standardowym wyjœciem
#SBATCH --output="output.out"
## Plik ze standardowym wyjœciem b³êdów
#SBATCH --error="error.err"
#SBATCH --partition=plgrid-gpu
#SBATCH --gres=gpu

module load plgrid/apps/cuda
ts=$(date +%s%N)
############ do ca³ki
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