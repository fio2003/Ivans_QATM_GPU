/*
 ============================================================================
 Name        : project_6_v1.cu
 Author      : Ivan Syzonenko
 Version     :
 Copyright   : You can use this code only if your first name is Ivan
 Description : CUDA QTAM program
 ============================================================================
 */

//#include <iostream>
//#include <numeric>
//#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "sphere_lebedev_rule.h"


#define _READ_STR_LEN 256
#define _SIZE_OF_CHAR sizeof(char)

#define PRIM_FUNC 0
#define GRAD 1
#define HESS 2
#define FULL_CALCULATION 2

struct pointD {
	double x;
	double y;
	double z;
};

struct pointI {
	int x;
	int y;
	int z;
};

__global__ void getDistanceComp_cuda( const double* cur, const double* origin, double* out, double* r)
{
	extern __shared__ double dcopy[];
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

	dcopy[threadIdx.x] = cur[threadIdx.x] - origin[idx];
	out[idx] = dcopy[threadIdx.x];
	dcopy[threadIdx.x] *= dcopy[threadIdx.x];

	__syncthreads();

	if (threadIdx.x == 0)
	{
		r[blockIdx.x] = sqrt(dcopy[0] + dcopy[1] + dcopy[2]);
	}
}


//getPsi_cuda<<<numOfNucl/*BLOCK_COUNT*/, numOfPrimFunc/*BLOCK_SIZE*/, numOfPrimFunc>>> (molOrbitals_cuda, primFunc_cuda, psi_cuda);
__global__ void getPsi_cuda(const double* molOrbitals, const double* primFunc, double* psi)
{
	extern __shared__ double dcopy[];
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	dcopy[threadIdx.x] = molOrbitals[idx] * primFunc[threadIdx.x];
//	printf("GPU_1: molOrbitals[%d] = %5.16f\n", idx, molOrbitals[idx]);
//	printf("GPU idx = %d and value is : %f\n", idx, dcopy[threadIdx.x]);
//	__syncthreads();
//	if (threadIdx.x == 0)
//	{
//		double temp = 0;
//		for(int i= 0; i<blockDim.x; ++i)
//		{
//			temp += dcopy[i];
//		}
//		printf("GPU FINAL STUPID block= %d and value is : %f\n", blockIdx.x, temp);
//	}
	__syncthreads();
	for (int tc = blockDim.x, stepSize = 1; tc > 0; tc >>= 1, stepSize <<= 1)
	{
	// thread must be allowed to write
		int pa = threadIdx.x * stepSize;
		int pb = pa + stepSize;
		if ( pb < blockDim.x)
		{
//			printf("ADDING %d and %d\n",pa,pb);
			dcopy[pa] += dcopy[pb];
		}
	}
//	__syncthreads();
	if (threadIdx.x == 0)
	{
		psi[blockIdx.x] = dcopy[0];
//		printf("GPU FINAL block= %d and value is : %f\n", blockIdx.x, psi[blockIdx.x]);
	}
}

//getDPsi_cuda<<<dim3(numOfNucl,3,1), numOfPrimFunc, numOfPrimFunc*sizeof(double)>>> (molOrbitals_cuda, dermodvar_cuda, dpsi_cuda);
__global__ void getDPsi_cuda(const double* molOrbitals, const double* dermodvar, double* dpsi)
{
	extern __shared__ double dcopy[];
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
//	int threadId = blockId * blockDim.x + threadIdx.x;
//	if (threadIdx.x == 0)
//	printf("My blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d, total = %d\n",blockIdx.x, blockIdx.y, blockIdx.z, blockId);
	dcopy[threadIdx.x] = molOrbitals[idx] * dermodvar[threadIdx.x*3 + blockIdx.y];
//	printf("My blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d, total = %d, thread: %d = %f * %f = %f\n",blockIdx.x, blockIdx.y, blockIdx.z, blockId, threadIdx.x, molOrbitals[idx],dermodvar[blockIdx.x *blockDim.x + blockDim.y],dcopy[threadIdx.x]);
//	printf("GPU: dpsi[%d][%d][%d] = %f * %f = %f\n", blockIdx.x, blockIdx.y,idx, molOrbitals[idx],dermodvar[threadIdx.x*3 + blockIdx.y],dcopy[threadIdx.x]);
//	printf("GPU: dermodvar[%d][%d][%d] = %f \n", threadIdx.x, blockIdx.y,threadIdx.x*3 + blockIdx.y,dermodvar[threadIdx.x*3 + blockIdx.y]);

	__syncthreads();
	for (int tc = blockDim.x, stepSize = 1; tc > 0; tc >>= 1, stepSize <<= 1)
	{
		int pa = threadIdx.x * stepSize;
		int pb = pa + stepSize;
		if ( pb < blockDim.x)
			dcopy[pa] += dcopy[pb];
	}
	__syncthreads();
	if (threadIdx.x == 0)
	{
		printf("IN GPU: I am [%d][%d] value = %f\n", blockIdx.x, blockIdx.y, dcopy[0]);
		dpsi[blockIdx.x*3 + blockIdx.y] = dcopy[0];
	}
}
//getDRho_cuda<<<3/*BLOCK_COUNT*/, numOfNucl/*BLOCK_SIZE*/, numOfNucl*sizeof(double)>>> (psi_cuda, dpsi_cuda, occNo_cuda, drho_cuda);
__global__ void getDRho_cuda(const double* psi, const double* dpsi, const double* occNo, double* drho)
{
	extern __shared__ double dcopy[];
	unsigned int idx = blockIdx.x + gridDim.x*threadIdx.x;
	dcopy[threadIdx.x] = 2 * occNo[threadIdx.x] * psi[threadIdx.x] * dpsi[idx];
	printf("GPU drho[%d] = 2 * %f * %f * %f = %f\n", blockIdx.x, occNo[threadIdx.x], psi[threadIdx.x], dpsi[idx], dcopy[threadIdx.x] );
//	printf("Inside GPU[t %d][tot %d]: %f\n",threadIdx.x, idx, dcopy[threadIdx.x]);
	__syncthreads();
	for (int tc = blockDim.x, stepSize = 1; tc > 0; tc >>= 1, stepSize <<= 1)
	{
		int pa = threadIdx.x * stepSize;
		int pb = pa + stepSize;
		if ( pb < blockDim.x)
			dcopy[pa] += dcopy[pb];
	}
	__syncthreads();
	if (threadIdx.x == 0)
	{
		drho[blockIdx.x] = dcopy[0];
	}
}

__global__ void getRho_cuda(const double* psi, const double* occNo, double* rho)
{
	extern __shared__ double dcopy[];
	dcopy[threadIdx.x] = occNo[threadIdx.x] * psi[threadIdx.x] * psi[threadIdx.x];
	__syncthreads();
	for (int tc = blockDim.x, stepSize = 1; tc > 0; tc >>= 1, stepSize <<= 1)
	{
	// thread must be allowed to write
		int pa = threadIdx.x * stepSize;
		int pb = pa + stepSize;
		if ( pb < blockDim.x)
		{

			dcopy[pa] += dcopy[pb];
		}
	}

	if (threadIdx.x == 0)
	{
		*rho = dcopy[0];
	}
}







void fixScientificNotation(char* str_to_read, int length);
int parseWFNfile (const char* filename, int* numOfMolOrb, int* numOfPrimFunc, char*** molName, struct pointD** origin,
					double** charge, int** types, int** molTypeFunc, int** molPowType, double** exponents, double** occNo,
					double** orbEnergy, double*** molOrbitals, int* numOfNucl, struct pointI** powcoef);
// int parseWFNfile (const char* filename, int& numOfMolOrb, int& numOfPrimFunc, char**& molName, pointD*& origin,
// 					double*& charge, int*& types, int*& molTypeFunc, int*& molPowType, double*& exponents, double*& occNo,
// 					double*& orbEnergy, double**& molOrbitals, int& numOfNucl, pointI*& powcoef);
// int releaseMemory (const char* filename, int& numOfMolOrb, int& numOfPrimFunc, char**& molName, pointD*& origin,
// 					double*& charge, int*& types, int*& molTypeFunc, int*& molPowType, double*& exponents, double*& occNo,
// 					double*& orbEnergy, double**& molOrbitals, int& numOfNucl, pointI*& powcoef);

void fillExpTypes(const int* molPowType, const int numOfPrimFunc, struct pointI* powcoef);

void getDistanceComp(const double cur_x, const double cur_y, const double cur_z, const struct pointD* origin, const int numOfNucl, struct pointD* out, const char debug );

void getDistanceAbs(const struct pointD* calcPoint, const int numOfNucl, double* r_dist, const char debug);

void getGRHO(const int numOfPrimFunc, const int* molTypeFunc, const double* r_dist, const double* exponents, const struct pointD* calcPoint, const struct pointI* powcoef, double* primFunc, double** dermodvar, double*** hesmodvar, const char selection, const char debug);

// void getHesPSI(const int numOfPrimFunc, const int numOfNucl, const double* const* molOrbitals, const double*  const * const * hesmodvar, double*** hespsi, const char debug);
void getHesPSI(const int numOfPrimFunc, const int numOfNucl, const double** molOrbitals, const double*** hesmodvar, double*** hespsi, const char debug);

void convcoord(const double* x, const double* y, const double* z, double* r, double* thet, double* phi );

void backconvcoord(const double* r, const double* thet, const double* phi, double* x, double* y, double* z );

void getPsi (const int numOfPrimFunc, const int numOfNucl, const double** molOrbitals, const double* primFunc, double* psi, const char debug);
// void getPsi (const int numOfPrimFunc, const int numOfNucl, const double* const* molOrbitals, const double* primFunc, double* psi, const char debug);

void getDPsi (const int numOfPrimFunc, const int numOfNucl, const double** molOrbitals, const double** dermodvar, double** dpsi, const char debug);
// void getDPsi (const int numOfPrimFunc, const int numOfNucl, const double* const* molOrbitals, const double* const* dermodvar, double** dpsi, const char debug);

void getRho(const int numOfNucl, const double* psi, const double* occNo, double* rho, const char debug);

void getDRho(const int numOfNucl, const double* psi, const double** dpsi, const double* occNo, double* drho, const char debug);
// void getDRho(const int numOfNucl, const double* psi, const double* const* dpsi, const double* occNo, double* drho, const char debug);

double myNormS (const double a, const double b, const double c);

double myNormV (const double* arr);

void getHesRho(const int numOfNucl, const double* psi, const double** dpsi, const double* occNo, const double*** hespsi, double** hesrho, const char debug);
// void getHesRho(const int numOfNucl, const double* psi, const double*  const * dpsi, const double* occNo, const double*  const * const * hespsi, double** hesrho, const char debug);

void reduction(double a[][6], int size, int pivot, int col);

void matrINV(const double** in_matr, double** out_matr, const char debug);

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)



/**
 * Host function that copies the data and launches the work on GPU
 */
//float *gpuReciprocal(float *data, unsigned size)
//{
//	float *rc = new float[size];
//	float *gpuData;
//
//	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
//	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));
//
//	static const int BLOCK_SIZE = 256;
//	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
//	reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, size);
//
//	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
//	CUDA_CHECK_RETURN(cudaFree(gpuData));
//	return rc;
//}




/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	//std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	printf("%s returned %s (%d) at %s:%d\n", statement,cudaGetErrorString(err),err, file, line);
	exit (1);
}


int main()
{
	clock_t begin, end;
	const char filename[] = "LiH_sto-6g.wfn";
	int numOfNucl 		= 0;
	int numOfMolOrb 	= 0;
	int numOfPrimFunc 	= 0;
	char**	molName		= NULL;
	struct pointD* origin		= NULL;
	double*	charge 		= NULL;
	int* 	types 		= NULL;
	int* 	molTypeFunc = NULL;
	int* 	molPowType 	= NULL;
	double* exponents 	= NULL;
	double* occNo 		= NULL;
	double* orbEnergy 	= NULL;
	double** molOrbitals = NULL;
	struct pointI* powcoef 	= NULL;

	// END OF MAIN MEMORY ALLOCATION

	parseWFNfile (	filename, &numOfMolOrb, &numOfPrimFunc, &molName, &origin, &charge, &types, &molTypeFunc, &molPowType,
					&exponents, &occNo, &orbEnergy, &molOrbitals, &numOfNucl, &powcoef);

	double* r_dist = (double*)malloc( numOfNucl * sizeof(double) );

	double* primFunc = (double*)malloc( numOfPrimFunc * sizeof(double) );

	double** dermodvar = (double**)malloc( numOfPrimFunc * sizeof(double*) );
	double* dermodvar_I = (double*)malloc( 3 * numOfPrimFunc * sizeof(double*) );
	for (int i = 0; i < numOfPrimFunc; ++i)
		dermodvar[i] = dermodvar_I + 3*i;

	double*** hesmodvar = (double***)malloc( numOfPrimFunc * sizeof(double**) );
	for (int i = 0; i < numOfPrimFunc; ++i)
		hesmodvar[i] = (double**)malloc( 3 * sizeof(double*) );

	double* hesmodvar_I = (double*)malloc( 3 * numOfPrimFunc * sizeof(double**) );
	for (int i = 0; i < numOfPrimFunc; ++i)
		for (int j = 0; j < 3; ++j)
			hesmodvar[i][j] = hesmodvar_I + numOfNucl*i + 3*j;

	double*** hespsi = (double***)malloc( numOfNucl * sizeof(double**) );
	for (int i = 0; i < numOfNucl; ++i)
		hespsi[i] = (double**)malloc( 3 * sizeof(double*) );

	double* hespsi_I = (double*)malloc(numOfNucl * 3 * 3 * sizeof(double));
	for (int i = 0; i < numOfNucl; ++i)
		for (int j = 0; j < 3; ++j)
			hespsi[i][j] = hespsi_I + numOfNucl*i + 3*j;

	double* psi = (double*)malloc(numOfNucl * sizeof(double));

	double** dpsi = (double**)malloc(numOfNucl * sizeof(double*));
	double* dpsi_I = (double*)malloc(numOfNucl * 3 * sizeof(double));
	for (int i = 0; i < numOfNucl; ++i)
		dpsi[i] = dpsi_I + 3*i;

	double drho[3];

	double** hesrho = (double**)malloc(3 * sizeof(double*));
	double* hesrho_I = (double*)malloc(3 * 3 * sizeof(double));
	for (int i = 0; i < 3; ++i)
		hesrho[i] = hesrho_I + 3*i;


	double** hesrhoINV = (double**)malloc(3 * sizeof(double*));
	double* hesrhoINV_I = (double*)malloc(3 * 3 * sizeof(double));
	for (int i = 0; i < 3; ++i)
		hesrhoINV[i] = hesrhoINV_I + 3*i;
		// hesrhoINV[i] = (double*)malloc(3 * sizeof(double));

	// double* hesrhoINV_I = (double*)malloc(3 * 3 * sizeof(double));
	// double (*hesrhoINV)[3] = (double (*)[3]) hesrhoINV_I;

	int numOfLebedevPoints = 110;
	double* lw = ( double * ) malloc ( numOfLebedevPoints * sizeof ( double ) );
	double* lx = ( double * ) malloc ( numOfLebedevPoints * sizeof ( double ) );
	double* ly = ( double * ) malloc ( numOfLebedevPoints * sizeof ( double ) );
	double* lz = ( double * ) malloc ( numOfLebedevPoints * sizeof ( double ) );

	ld_by_order ( numOfLebedevPoints, lx, ly, lz, lw );
	// for(int i = 0; i < numOfLebedevPoints; i++)
	// 	printf("Point(%d)\t%f\t%f\t%f\t%f\n", i, lx[i], ly[i], lz[i], lw[i]);
	// FILE IS PARSED AND WE SHOULD BE READY TO START HAVING FUN
// this case for only two points, if we need more, next lines should be rewriten
	struct pointD* startPoint = (struct pointD *)malloc(sizeof(struct pointD));
	startPoint->x = ( origin[0].x + origin[1].x )/2;
	startPoint->y = ( origin[0].y + origin[1].y )/2;
	startPoint->z = ( origin[0].z + origin[1].z )/2;

	struct pointD* calcPoint = (struct pointD *)malloc(2*sizeof(struct pointD));


	getDistanceComp(startPoint->x, startPoint->y, startPoint->z, origin, numOfNucl, calcPoint, 0);

	for(int i = 0; i < numOfNucl; ++i)
	{
		printf("calcpoint[%d] = %f;\n", i*3,calcPoint[i].x);
		printf("calcpoint[%d] = %f;\n", i*3+1,calcPoint[i].y);
		printf("calcpoint[%d] = %f;\n\n", i*3+2,calcPoint[i].z);
	}

	getDistanceAbs(calcPoint, numOfNucl, r_dist, 1);


////////////////////////////////////////////                  BEGIN DISTANCE DEBUG                                /////////////////////////////////////



	double *cur_cuda, *origin_cuda, *out_cuda, *dist_cuda;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&cur_cuda, sizeof(double)*3));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&origin_cuda, sizeof(double)*3*numOfNucl));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&out_cuda, sizeof(double)*3*numOfNucl));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&dist_cuda, sizeof(double)*numOfNucl));

	double *temp_data = (double*)malloc(3*sizeof(double));
	temp_data[0] = startPoint->x;
	temp_data[1] = startPoint->y;
	temp_data[2] = startPoint->z;

	double *temp_origin = (double*)malloc(numOfNucl*3*sizeof(double));
	for(int i = 0; i < numOfNucl; ++i)
	{
		temp_origin[i*3+0] = origin[i].x;
		temp_origin[i*3+1] = origin[i].y;
		temp_origin[i*3+2] = origin[i].z;
	}

	double *temp_out = (double*)malloc(numOfNucl*3*sizeof(double));
	double *temp_out_r = (double*)malloc(numOfNucl*sizeof(double));

//	printf("BEFORE COPY:\n");
//	for(int i = 0; i < numOfNucl; ++i)
//	{
//		printf("Temp_origin %d  = %f %f %f;\n",i, temp_origin[i*3], temp_origin[3*i+1], temp_origin[3*i+2]);
//	}
//
//	for(int i = 0; i < 3; ++i)
//	{
//		printf("Temp_data %d = %f;\n",i, temp_data[i]);
//	}

	CUDA_CHECK_RETURN(cudaMemcpy(cur_cuda, temp_data, sizeof(double)*3, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(origin_cuda, temp_origin, sizeof(double)*3*numOfNucl, cudaMemcpyHostToDevice));

	getDistanceComp_cuda<<<numOfNucl/*BLOCK_COUNT*/, 3/*BLOCK_SIZE*/, 3*sizeof(double)>>> (cur_cuda, origin_cuda, out_cuda, dist_cuda);

	CUDA_CHECK_RETURN(cudaMemcpy(temp_out, out_cuda, 3*numOfNucl*sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(temp_out_r, dist_cuda, numOfNucl*sizeof(double), cudaMemcpyDeviceToHost));


	for(int i = 0; i < numOfNucl*3; ++i)
	{
		printf("GPU calcpoint[%d] = %f;\n",i, temp_out[i]);
	}
	printf("\n");

	for(int i = 0; i < numOfNucl; ++i)
	{
		printf("GPU distance[%d] = %f;\n",i, temp_out_r[i] );
	}
	printf("\n");

////////////////////////////////////////////                  END DISTANCE DEBUG                                /////////////////////////////////////



//	getDistanceAbs(calcPoint, numOfNucl, r_dist, 0);

	getGRHO(numOfPrimFunc, molTypeFunc, r_dist, exponents, calcPoint, powcoef, primFunc, dermodvar, hesmodvar, HESS, 0);

	getHesPSI(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, (const double ***)hesmodvar, hespsi, 0 );

	getPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, primFunc, psi, 1);


////////////////////////////////////////////                  BEGIN PSI DEBUG                                /////////////////////////////////////





	double *psi_cuda, *occNo_cuda, *rho_cuda, *psi_out, *temp_molOrbitals, *temp_primFunc, *primFunc_cuda, *molOrbitals_cuda;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&psi_cuda, sizeof(double) * numOfNucl));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&molOrbitals_cuda, sizeof(double) * numOfNucl * numOfPrimFunc));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&primFunc_cuda, sizeof(double) * numOfPrimFunc));

	temp_molOrbitals = (double*)malloc(sizeof(double)*numOfNucl*numOfPrimFunc);
	for(int j = 0; j < numOfNucl; ++j)
		for(int i = 1; i < numOfPrimFunc; ++i)
		{
			temp_molOrbitals[j*numOfPrimFunc + i] = molOrbitals[j][i];
//			printf("DEBUG print for molOrbitals i=%d j=%d (%d): %f\n", i, j, j*numOfPrimFunc + i,molOrbitals[j][i]);
		}

//	for(int i = 0; i < numOfNucl *numOfPrimFunc; ++i )
//		printf("DEBUG print (COMPR) for i=%d : %f\n", i, temp_molOrbitals[i]);

	temp_primFunc = (double*)malloc(sizeof(double) *numOfPrimFunc );
	for (int j = 1; j < numOfPrimFunc; ++j)
		temp_primFunc[j] = primFunc[j];

	psi_out = (double*)malloc(sizeof(double) * numOfNucl);

	CUDA_CHECK_RETURN(cudaMemcpy(molOrbitals_cuda, temp_molOrbitals, sizeof(double)*numOfNucl*numOfPrimFunc, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(primFunc_cuda, temp_primFunc, sizeof(double)*numOfPrimFunc, cudaMemcpyHostToDevice));

	getPsi_cuda<<<numOfNucl/*BLOCK_COUNT*/, numOfPrimFunc/*BLOCK_SIZE*/, numOfPrimFunc*sizeof(double)>>> (molOrbitals_cuda, primFunc_cuda, psi_cuda);

	CUDA_CHECK_RETURN(cudaMemcpy(psi_out, psi_cuda, sizeof(double) * numOfNucl, cudaMemcpyDeviceToHost));

	for (int i = 0; i < numOfNucl; ++i)
		printf("[GPU] PSI[%d] = %f\n",i, psi_out[i]);
	printf("\n");

////////////////////////////////////////////                  END PSI DEBUG                                /////////////////////////////////////


	getDPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, (const double **)dermodvar, dpsi, 1);


	double *dpsi_cuda,  *dermodvar_cuda, *dermodvar_temp, *temp_dpsi;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&dpsi_cuda, sizeof(double)*3*numOfNucl));

	CUDA_CHECK_RETURN(cudaMalloc((void **)&dermodvar_cuda, sizeof(double)*3*numOfPrimFunc));

	temp_dpsi = (double*)malloc(sizeof(double) * 3 * numOfNucl);

	dermodvar_temp = (double*)malloc(sizeof(double) * 3*numOfPrimFunc);

	for (int i = 0; i < numOfPrimFunc; ++i)
	{
		dermodvar_temp[3*i] = dermodvar[i][0];
		dermodvar_temp[3*i + 1] = dermodvar[i][1];
		dermodvar_temp[3*i + 2] = dermodvar[i][2];
//		printf("dermodvar[%d][%d] [0]=%f, [1]=%f, [2]=%f \n",3*i,i,dermodvar[i][0],dermodvar[i][1],dermodvar[i][2]);
	}

//	for (int i = 0; i < numOfPrimFunc*3; ++i)
//		printf("dermodvar[%d] = %f \n",i,dermodvar_temp[i]);

	CUDA_CHECK_RETURN(cudaMemcpy(dermodvar_cuda, dermodvar_temp, sizeof(double)*3*numOfPrimFunc, cudaMemcpyHostToDevice));

	getDPsi_cuda<<<dim3(numOfNucl,3,1), numOfPrimFunc, numOfPrimFunc*sizeof(double)>>> (molOrbitals_cuda, dermodvar_cuda, dpsi_cuda);


	CUDA_CHECK_RETURN(cudaMemcpy(temp_dpsi, dpsi_cuda, sizeof(double)*3*numOfNucl, cudaMemcpyDeviceToHost));

//	printf("Rho is : %f", temp_dpsi);
	for (int i = 0; i < numOfNucl; ++i)
		printf("GPU dpsi[%d]: \t[%d] %f,\t[%d] %f,\t[%d] %f\n",i, 3*i,temp_dpsi[3*i], 3*i+1,temp_dpsi[3*i+1],i*3+2, temp_dpsi[3*i+2]);


////////////////////////////////////////////                  BEGIN RHO DEBUG                                /////////////////////////////////////



	double /**temp_psi, */*temp_occNo, *rho_out;

	rho_out = (double*)malloc(sizeof(double));
//	temp_psi = (double*)malloc(sizeof(double) * numOfNucl);
	temp_occNo = (double*)malloc(sizeof(double) * numOfNucl);

	for(int i = 0; i < numOfNucl; ++i)
	{
//		temp_psi[i] = psi[i];
		temp_occNo[i] = occNo[i];
	}

	CUDA_CHECK_RETURN(cudaMalloc((void **)&occNo_cuda, sizeof(double) * numOfNucl));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&rho_cuda, sizeof(double)));

//	CUDA_CHECK_RETURN(cudaMemcpy(psi_cuda, temp_psi, sizeof(double)*numOfNucl, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(occNo_cuda, temp_occNo, sizeof(double)*numOfNucl, cudaMemcpyHostToDevice));

	double somedouble;
	getRho(numOfNucl,  psi, occNo, &somedouble, 1);


	getRho_cuda<<<1/*BLOCK_COUNT*/, numOfNucl/*BLOCK_SIZE*/, numOfNucl*sizeof(double)>>> (psi_cuda, occNo_cuda, rho_cuda);


	CUDA_CHECK_RETURN(cudaMemcpy(rho_out, rho_cuda, sizeof(double), cudaMemcpyDeviceToHost));

	printf("[GPU] Rho is : %f\n\n", *rho_out);


////////////////////////////////////////////                  END RHO DEBUG                                /////////////////////////////////////


	getDRho(numOfNucl,  psi,  (const double **)dpsi, occNo, drho, 1);



////////////////////////////////////////////                  BEGIN DRHO DEBUG                                /////////////////////////////////////

	double *drho_cuda, /**dpsi_cuda, *temp_dpsi, */*temp_drho;


	CUDA_CHECK_RETURN(cudaMalloc((void **)&drho_cuda, sizeof(double)*3));
/*	CUDA_CHECK_RETURN(cudaMalloc((void **)&dpsi_cuda, sizeof(double)*3*numOfNucl));*/


	/*temp_dpsi = (double*)malloc(sizeof(double) * 3 * numOfNucl);*/
	temp_drho = (double*)malloc(sizeof(double) * 3 );

//	for(int i = 0; i < numOfNucl; ++i)
//		for(int j = 0; j < 3; ++j)
//		{
//			temp_dpsi[j*numOfNucl + i] = dpsi[i][j];
//		}

//	for(int i = 0; i < numOfNucl * 3; ++i)
//	{
//		printf("DEBUG print (COMPR) for i=%d : %f\n", i, temp_dpsi[i]);
//	}

//	CUDA_CHECK_RETURN(cudaMemcpy(dpsi_cuda, temp_dpsi, sizeof(double)*numOfNucl*3, cudaMemcpyHostToDevice));


	getDRho_cuda<<<3/*BLOCK_COUNT*/, numOfNucl/*BLOCK_SIZE*/, numOfNucl*sizeof(double)>>> (psi_cuda, dpsi_cuda, occNo_cuda, drho_cuda);

	CUDA_CHECK_RETURN(cudaMemcpy(temp_drho, drho_cuda, sizeof(double)*3, cudaMemcpyDeviceToHost));

	printf("GPU: DRHO [%f]:\t[%f]:\t[%f]:\n", temp_drho[0], temp_drho[1], temp_drho[2]);
////////////////////////////////////////////                  END DRHO DEBUG                                /////////////////////////////////////


	getHesRho(numOfNucl, psi, (const double **)dpsi, occNo, (const double ***)hespsi, hesrho, 0);

	CUDA_CHECK_RETURN(cudaFree(cur_cuda));
	CUDA_CHECK_RETURN(cudaFree(origin_cuda));
	CUDA_CHECK_RETURN(cudaFree(out_cuda));
	CUDA_CHECK_RETURN(cudaFree(dist_cuda));
	CUDA_CHECK_RETURN(cudaFree(psi_cuda));
	CUDA_CHECK_RETURN(cudaFree(occNo_cuda));
	CUDA_CHECK_RETURN(cudaFree(rho_cuda));

	double normC = 99;
	int iterWL = 0;
	double hi[3];

	begin = clock();
	while(normC > 1E-6 && iterWL < 200)
	{
	    matrINV((const double **)hesrho, hesrhoINV, 0);
	    hi[0] = -( drho[0] * hesrhoINV[0][0] + drho[1] * hesrhoINV[1][0] + drho[2] * hesrhoINV[2][0] );
	    hi[1] = -( drho[0] * hesrhoINV[0][1] + drho[1] * hesrhoINV[1][1] + drho[2] * hesrhoINV[2][1] );
	    hi[2] = -( drho[0] * hesrhoINV[0][2] + drho[1] * hesrhoINV[1][2] + drho[2] * hesrhoINV[2][2] );

	    normC = myNormV(hi);
	    printf("Norm: %f\n", normC);

	    startPoint->x += hi[0];
	    startPoint->y += hi[1];
	    startPoint->z += hi[2];

	    getDistanceComp(startPoint->x, startPoint->y, startPoint->z, origin, numOfNucl, calcPoint, 0);

	    getDistanceAbs(calcPoint, numOfNucl, r_dist, 0);
	    getGRHO(numOfPrimFunc, molTypeFunc, r_dist, exponents, calcPoint, powcoef, primFunc, dermodvar, hesmodvar, HESS, 0);
		getHesPSI(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, (const double ***)hesmodvar, hespsi, 0 );
		getPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, primFunc, psi, 0 );
		getDPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, (const double **)dermodvar, dpsi, 0 );
		getDRho(numOfNucl,  psi,  (const double **)dpsi, occNo, drho, 0);
		getHesRho(numOfNucl, psi, (const double **)dpsi, occNo, (const double ***)hespsi, hesrho, 0 );
	}
	end = clock();
	printf("Total time %f\n", (double)(end - begin) / CLOCKS_PER_SEC);

 	double* beta = (double*)malloc(numOfNucl * sizeof(double));

 	for (int i = 0; i < numOfNucl; ++i)
 		beta[i] = 0.85 * r_dist[i];

 	double rk_step = 0.15;
 	printf("RK step : %f\n", rk_step);

 	double dist;
 	double ray;

 	double r, thet, phi;
 	double x, y, z;

	double* IASanswers_I = (double*)malloc( numOfNucl * numOfLebedevPoints * sizeof(double) );
	double (*IASanswers)[numOfLebedevPoints] = (double (*)[numOfLebedevPoints]) IASanswers_I;
 	char stopflag = 0;
 	double grad_1[3], grad_2[3], grad_3[3], grad_4[3];


 	double norm;
 	double cur_posrk[3];


 	begin = clock();


 	for (int nuclNumIter = 0; nuclNumIter < numOfNucl; ++nuclNumIter)
 	{
 		for (int curlebpoint = 0; curlebpoint < numOfLebedevPoints; ++curlebpoint)
 		{
 			dist = 0.15;
 			convcoord(&lx[curlebpoint], &ly[curlebpoint], &lz[curlebpoint], &r, &thet, &phi );
        	ray = beta[nuclNumIter] + dist;
        	while (dist > 1E-6 && ray < 10 )
        	{
        		backconvcoord( &ray, &thet, &phi, &x, &y, &z );
	            x += origin[nuclNumIter].x;
	            y += origin[nuclNumIter].y;
	            z += origin[nuclNumIter].z;

    	        getDistanceComp(x, y, z, origin, numOfNucl, calcPoint, 0);

        	    getDistanceAbs(calcPoint, numOfNucl, r_dist, 0);
			    getGRHO(numOfPrimFunc, molTypeFunc, r_dist, exponents, calcPoint, powcoef, primFunc, dermodvar, hesmodvar, GRAD, 0);
				getPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, primFunc, psi, 0 );
				getDPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, (const double **)dermodvar, dpsi, 0 );
				getDRho(numOfNucl,  psi,  (const double **)dpsi, occNo, drho, 0);
            	stopflag = 0;
            	while(stopflag == 0)
            	{
            		norm = myNormV (drho);
            	    grad_1[0] = drho[0]/norm;
            	    grad_1[1] = drho[1]/norm;
            	    grad_1[2] = drho[2]/norm;

            	    cur_posrk[0] = x + rk_step * 0.5 * grad_1[0];
            	    cur_posrk[1] = y + rk_step * 0.5 * grad_1[1];
            	    cur_posrk[2] = z + rk_step * 0.5 * grad_1[2];

            	    getDistanceComp(cur_posrk[0], cur_posrk[1], cur_posrk[2], origin, numOfNucl, calcPoint, 0);
            	    getDistanceAbs(calcPoint, numOfNucl, r_dist, 0);
					getGRHO(numOfPrimFunc, molTypeFunc, r_dist, exponents, calcPoint, powcoef, primFunc, dermodvar, hesmodvar, GRAD, 0);
					getPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, primFunc, psi, 0 );
					getDPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, (const double **)dermodvar, dpsi, 0 );
					getDRho(numOfNucl,  psi,  (const double **)dpsi, occNo, drho, 0);

					norm = myNormV (drho);

					grad_2[0] = drho[0]/norm;
					grad_2[1] = drho[1]/norm;
					grad_2[2] = drho[2]/norm;

					cur_posrk[0] = x + rk_step * 0.5 * grad_2[0];
            	    cur_posrk[1] = y + rk_step * 0.5 * grad_2[1];
            	    cur_posrk[2] = z + rk_step * 0.5 * grad_2[2];

            	    getDistanceComp(cur_posrk[0], cur_posrk[1], cur_posrk[2], origin, numOfNucl, calcPoint, 0);
            	    getDistanceAbs(calcPoint, numOfNucl, r_dist, 0);
					getGRHO(numOfPrimFunc, molTypeFunc, r_dist, exponents, calcPoint, powcoef, primFunc, dermodvar, hesmodvar, GRAD, 0);
					getPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, primFunc, psi, 0 );
					getDPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, (const double **)dermodvar, dpsi, 0 );
					getDRho(numOfNucl,  psi,  (const double **)dpsi, occNo, drho, 0);

					norm = myNormV (drho);

					grad_3[0] = drho[0]/norm;
					grad_3[1] = drho[1]/norm;
					grad_3[2] = drho[2]/norm;

					cur_posrk[0] = x + rk_step * grad_3[0];
            	    cur_posrk[1] = y + rk_step * grad_3[1];
            	    cur_posrk[2] = z + rk_step * grad_3[2];

            	    getDistanceComp(cur_posrk[0], cur_posrk[1], cur_posrk[2], origin, numOfNucl, calcPoint, 0);
            	    getDistanceAbs(calcPoint, numOfNucl, r_dist, 0);
					getGRHO(numOfPrimFunc, molTypeFunc, r_dist, exponents, calcPoint, powcoef, primFunc, dermodvar, hesmodvar, GRAD, 0);
					getPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, primFunc, psi, 0 );
					getDPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, (const double **)dermodvar, dpsi, 0 );
					getDRho(numOfNucl,  psi,  (const double **)dpsi, occNo, drho, 0);

					norm = myNormV (drho);

					grad_4[0] = drho[0]/norm;
					grad_4[1] = drho[1]/norm;
					grad_4[2] = drho[2]/norm;

					// cur_pos = cur_pos + (1/6)*(grad_1 + 2*(grad_2 + grad_3) + grad_4)*h;  % main equation

					x = x + (grad_1[0] + 2*(grad_2[0] + grad_3[0]) + grad_4[0]) * rk_step/6.0;
					y = y + (grad_1[1] + 2*(grad_2[1] + grad_3[1]) + grad_4[1]) * rk_step/6.0;
					z = z + (grad_1[2] + 2*(grad_2[2] + grad_3[2]) + grad_4[2]) * rk_step/6.0;

					getDistanceComp(x, y, z, origin, numOfNucl, calcPoint, 0);
            	    getDistanceAbs(calcPoint, numOfNucl, r_dist, 0);
					getGRHO(numOfPrimFunc, molTypeFunc, r_dist, exponents, calcPoint, powcoef, primFunc, dermodvar, hesmodvar, GRAD, 0);
					getPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, primFunc, psi, 0 );
					getDPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, (const double **)dermodvar, dpsi, 0 );
					getDRho(numOfNucl,  psi,  (const double **)dpsi, occNo, drho, 0);

					// printf("%f   %f   %f\n",drho[0], drho[1], drho[2] );
					for (int checkAtBond = 0; checkAtBond < numOfNucl; ++checkAtBond)
					{
						if (nuclNumIter == checkAtBond && r_dist[nuclNumIter] < beta[nuclNumIter])
						{
//							printf("First\n");
							ray += dist;
							stopflag = 1;
 							break;
						}
						else if (r_dist[checkAtBond] < beta[checkAtBond])
						{
//							printf("Second\n");
							dist /= 2;
							ray -= dist;
							stopflag = 1;
							break;
						}
					}
				}
        	}

        	printf("Point reached (%d) : %d with radius : %f\n", nuclNumIter, curlebpoint, ray );

			IASanswers[nuclNumIter][curlebpoint] = ray;
			dist = 0;
 		}
 	}

 	end = clock();
 	printf("IAS search time %f\n", (double)(end - begin) / CLOCKS_PER_SEC);

 	printf("INTEGRATION PART\n"); // INTEGRATION ------------------------------- INTEGRATION ---------------------------
	begin = clock();
 	int numberOfRadialPoints = 64;//points of quadrature
 	double rayQinput[64][2] = { { 0.0486909570091397,	-0.0243502926634244 },
								{ 0.0486909570091397,	0.0243502926634244  },
								{ 0.0485754674415034,	-0.0729931217877990 },
								{ 0.0485754674415034,	0.0729931217877990  },
								{ 0.0483447622348030,	-0.1214628192961206 },
								{ 0.0483447622348030,	0.1214628192961206  },
								{ 0.0479993885964583,	-0.1696444204239928 },
								{ 0.0479993885964583,	0.1696444204239928  },
								{ 0.0475401657148303,	-0.2174236437400071 },
								{ 0.0475401657148303,	0.2174236437400071  },
								{ 0.0469681828162100,	-0.2646871622087674 },
								{ 0.0469681828162100,	0.2646871622087674  },
								{ 0.0462847965813144,	-0.3113228719902110 },
								{ 0.0462847965813144,	0.3113228719902110  },
								{ 0.0454916279274181,	-0.3572201583376681 },
								{ 0.0454916279274181,	0.3572201583376681  },
								{ 0.0445905581637566,	-0.4022701579639916 },
								{ 0.0445905581637566,	0.4022701579639916  },
								{ 0.0435837245293235,	-0.4463660172534641 },
								{ 0.0435837245293235,	0.4463660172534641  },
								{ 0.0424735151236536,	-0.4894031457070530 },
								{ 0.0424735151236536,	0.4894031457070530  },
								{ 0.0412625632426235,	-0.5312794640198946 },
								{ 0.0412625632426235,	0.5312794640198946  },
								{ 0.0399537411327203,	-0.5718956462026340 },
								{ 0.0399537411327203,	0.5718956462026340  },
								{ 0.0385501531786156,	-0.6111553551723933 },
								{ 0.0385501531786156,	0.6111553551723933  },
								{ 0.0370551285402400,	-0.6489654712546573 },
								{ 0.0370551285402400,	0.6489654712546573  },
								{ 0.0354722132568824,	-0.6852363130542333 },
								{ 0.0354722132568824,	0.6852363130542333  },
								{ 0.0338051618371416,	-0.7198818501716109 },
								{ 0.0338051618371416,	0.7198818501716109  },
								{ 0.0320579283548516,	-0.7528199072605319 },
								{ 0.0320579283548516,	0.7528199072605319  },
								{ 0.0302346570724025,	-0.7839723589433414 },
								{ 0.0302346570724025,	0.7839723589433414  },
								{ 0.0283396726142595,	-0.8132653151227975 },
								{ 0.0283396726142595,	0.8132653151227975  },
								{ 0.0263774697150547,	-0.8406292962525803 },
								{ 0.0263774697150547,	0.8406292962525803  },
								{ 0.0243527025687109,	-0.8659993981540928 },
								{ 0.0243527025687109,	0.8659993981540928  },
								{ 0.0222701738083833,	-0.8893154459951141 },
								{ 0.0222701738083833,	0.8893154459951141  },
								{ 0.0201348231535302,	-0.9105221370785028 },
								{ 0.0201348231535302,	0.9105221370785028  },
								{ 0.0179517157756973,	-0.9295691721319396 },
								{ 0.0179517157756973,	0.9295691721319396  },
								{ 0.0157260304760247,	-0.9464113748584028 },
								{ 0.0157260304760247,	0.9464113748584028  },
								{ 0.0134630478967186,	-0.9610087996520538 },
								{ 0.0134630478967186,	0.9610087996520538  },
								{ 0.0111681394601311,	-0.9733268277899110 },
								{ 0.0111681394601311,	0.9733268277899110  },
								{ 0.0088467598263639,	-0.9833362538846260 },
								{ 0.0088467598263639,	0.9833362538846260  },
								{ 0.0065044579689784,	-0.9910133714767443 },
								{ 0.0065044579689784,	0.9910133714767443  },
								{ 0.0041470332605625,	-0.9963401167719553 },
								{ 0.0041470332605625,	0.9963401167719553  },
								{ 0.0017832807216964,	-0.9993050417357722 },
								{ 0.0017832807216964,	0.9993050417357722  } };


	double* Lit 	= (double*)malloc( numOfNucl * sizeof(double) );
	double* Git  	= (double*)malloc( numOfNucl * sizeof(double) );
	double* Kit  	= (double*)malloc( numOfNucl * sizeof(double) );
	double* Nintit	= (double*)malloc( numOfNucl * sizeof(double) );
	// double* nabPsiSq	= (double*)malloc( numOfNucl * sizeof(double) );
	// double* nabla2Psi	= (double*)malloc( numOfNucl * sizeof(double) );

	double Ltot = 0;
	double Gtot = 0;
	double Ktot = 0;
	double Ntot = 0;

	double sumL = 0;
	double sumG = 0;
	double sumK = 0;
	double sumN = 0;

	double Ls = 0;
	double Gs = 0;
	double Ks = 0;
	double Ns = 0;
	double nabla2R = 0;

	double rayQM = 0;

	double temp_sum = 0;

	// double rho;

	for (int atomPoint = 0; atomPoint < numOfNucl; ++atomPoint)
	{
		Lit[atomPoint]  	= 0;
		Git[atomPoint]  	= 0;
		Kit[atomPoint]  	= 0;
		Nintit[atomPoint]  	= 0;

		for (int lebPoint = 0; lebPoint < numOfLebedevPoints; ++lebPoint)
		{
			convcoord(&lx[lebPoint], &ly[lebPoint], &lz[lebPoint], &r, &thet, &phi );
			ray = IASanswers[atomPoint][lebPoint];
			sumL = 0;
	        sumG = 0;
	        sumK = 0;
	        sumN = 0;

	        for (int radialIntIterator = 0; radialIntIterator < numberOfRadialPoints; ++radialIntIterator)
	        {
	        	rayQM = ray * (rayQinput[radialIntIterator][1] + 1) / 2.0;
	        	backconvcoord( &rayQM, &thet, &phi, &x, &y, &z );
	        	x += origin[atomPoint].x;
	            y += origin[atomPoint].y;
	            z += origin[atomPoint].z;
	        	getDistanceComp(x, y, z, origin, numOfNucl, calcPoint, 0);

	        	getDistanceAbs(calcPoint, numOfNucl, r_dist, 0);
				getGRHO(numOfPrimFunc, molTypeFunc, r_dist, exponents, calcPoint, powcoef, primFunc, dermodvar, hesmodvar, HESS, 0);
				getHesPSI(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, (const double ***)hesmodvar, hespsi, 0 );
				getPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, primFunc, psi, 0 );
				getDPsi(numOfPrimFunc, numOfNucl, (const double **)molOrbitals, (const double **)dermodvar, dpsi, 0 );
				getRho(numOfNucl,  psi, occNo, &Ns, 0);
				getHesRho(numOfNucl, psi, (const double **)dpsi, occNo, (const double ***)hespsi, hesrho, 0);

				// Ns = rho;

				sumN += Ns * rayQinput[radialIntIterator][0] * rayQM * rayQM;
				nabla2R = hesrho[0][0] + hesrho[1][1] + hesrho[2][2];
				Ls = -0.25*nabla2R;
				sumL += Ls * rayQinput[radialIntIterator][0] * rayQM * rayQM;
				// Next code is optimized: I calculate Gs without intermediate results.
				// for (int i = 0; i < numOfNucl; ++i)
				// {
				// 	nabPsiSq[i] = dpsi[i][0]*dpsi[i][0] + dpsi[i][1]*dpsi[i][1] + dpsi[i][2]*dpsi[i][2];
				// }

				// Gs = 0.5 * (occNo[0] * nabPsiSq[0] + occNo[1] * nabPsiSq[1]);
				Gs = 0;
				for (int i = 0; i < numOfNucl; ++i)
				{
					Gs += occNo[i] * (dpsi[i][0]*dpsi[i][0] + dpsi[i][1]*dpsi[i][1] + dpsi[i][2]*dpsi[i][2]);
				}

				Gs /= 2.0;

				sumG += Gs * rayQinput[radialIntIterator][0] * rayQM * rayQM;

				// Next code is optimized: I calculate Ks without intermediate results.
				// for (int i = 0; i < numOfNucl; ++i)
				// {
				// 	nabla2Psi[i] = hespsi[i][0][0] + hespsi[i][1][1]  + hespsi[i][2][2] ;
				// }

				// Ks = -0.5 * (occNo[0] * psi[0] * nabla2Psi[0] + occNo[1] * psi[1] * nabla2Psi[1] );
				Ks = 0;
				for (int i = 0; i < numOfNucl; ++i)
				{
					Ks += occNo[i] * psi[i] * ( hespsi[i][0][0] + hespsi[i][1][1]  + hespsi[i][2][2] ) ;
				}

				Ks /= -2.0;

				sumK += Ks * rayQinput[radialIntIterator][0] * rayQM * rayQM;
	        }

	        // sumL = 0.5 * ray * sumL;
	        // sumG = 0.5 * ray * sumG;
	        // sumK = 0.5 * ray * sumK;
	        // sumN = 0.5 * ray * sumN;
	        temp_sum = 0.5 * ray * lw[lebPoint] * 4 * 3.14159265359;
	        Lit[atomPoint] 		+= sumL * temp_sum;
	        Git[atomPoint] 		+= sumG * temp_sum;
	        Kit[atomPoint] 		+= sumK * temp_sum;
	        Nintit[atomPoint] 	+= sumN * temp_sum;

	        // Ltot += lw[lebPoint] * sumL * 4 * 3.14159265359;
	        // Gtot += lw[lebPoint] * sumG * 4 * 3.14159265359;
	        // Ktot += lw[lebPoint] * sumK * 4 * 3.14159265359;
	        // Ntot += lw[lebPoint] * sumN * 4 * 3.14159265359;

		}
	}

	for (int atomPoint = 0; atomPoint < numOfNucl; ++atomPoint)
	{
		Ltot += Lit[atomPoint];
		Gtot += Git[atomPoint];
		Ktot += Kit[atomPoint];
		Ntot += Nintit[atomPoint];
	}
	end = clock();
	printf("Integration time %f\n", (double)(end - begin) / CLOCKS_PER_SEC);

	printf("Ktot = \t%f\n", Ktot);
	printf("Gtot = \t%f\n", Gtot);
	printf("Ltot = \t%f\n", Ltot);
	printf("Ntot = \t%f\n", Ntot);
	printf("Kit = \t%f\t%f\n", Kit[0], Kit[1]);
	printf("Lit = \t%f\t%f\n", Lit[0], Lit[1]);
	printf("Git = \t%f\t%f\n", Git[0], Git[1]);
	printf("Nintit = \t%f\t%f\n", Nintit[0], Nintit[1]);

	printf("Ktot - Gtot - Ltot = %1.16f\n", Ktot - Gtot - Ltot);



	printf("Finished \n");

	printf("Before memory release \n");
// DO NOT CROSS THIS LINE !!! FROM THIS POINT WE RELEASING THE MEMORY:
	// NOT ALL MEMORY WAS HANDLED BY releaseMemory -       THIS IS IMPORTANT
	// releaseMemory (	filename, numOfMolOrb, numOfPrimFunc, molName, origin, charge, types, molTypeFunc, molPowType,
	// 				exponents, occNo, orbEnergy, molOrbitals, numOfNucl, powcoef);

	free(calcPoint);
	calcPoint = NULL;

	free(startPoint);
	startPoint = NULL;

	free(molName);
	molName = NULL;

	free(origin);
	origin = NULL;

	free(charge);//array of charges
	charge = NULL;//array of charges

	free(types) ;
	types = NULL;

	free(molTypeFunc);
	molTypeFunc = NULL;

	free(molPowType);
	molPowType = NULL;

	free(exponents);
	exponents = NULL;

	free(occNo);
	occNo = NULL;

	free(orbEnergy);
	orbEnergy = NULL;

	free(molOrbitals);
	molOrbitals = NULL;

	// free(cur_posrk);
	// cur_posrk = NULL;

	free(IASanswers_I);
	IASanswers_I = NULL;
	IASanswers = NULL;

	free(beta);
	beta = NULL;

	free(hesrhoINV_I);
	hesrhoINV_I = NULL;

	free(hesrhoINV);
	hesrhoINV = NULL;

	free(hesrho_I);
	hesrho_I = NULL;

	free(hesrho);
	hesrho = NULL;

	// free(drho);
	// drho = NULL;

	free(psi);
	psi = NULL;

	free(dpsi_I);
	dpsi_I = NULL;

	free(dpsi);
	dpsi = NULL;

	free(hespsi_I);
	hespsi_I = NULL;

	free(hespsi);
	hespsi = NULL;

	free(hesmodvar_I);
	hesmodvar_I = NULL;

	free(hesmodvar);
	hesmodvar = NULL;

	free(dermodvar_I);
	dermodvar_I = NULL;

	free(dermodvar);
	dermodvar = NULL;

	free(primFunc);
	primFunc = NULL;

	free(r_dist);
	r_dist = NULL;

	free(lw);
	lw = NULL;

	free(lx);
	lx = NULL;

	free(ly);
	ly = NULL;

	free(lz);
	lz = NULL;

	free(Lit);
	Lit = NULL;

	free(Git);
	Git = NULL;

	free(Kit);
	Kit = NULL;

	free(Nintit);
	Nintit = NULL;

//	free(nabPsiSq);
//	nabPsiSq = NULL;
//
//	free(nabla2Psi);
//	nabla2Psi = NULL;


	printf("After memory release \n");
	printf("Exiting...\n\n");

	return 0;
}

void fixScientificNotation(char* str_to_read, int length)
{
	for (int i = 0; i < length; ++i)//fix in scientific notation D to e
		if (str_to_read[i] == 'd' || str_to_read[i] == 'D' )
			str_to_read[i] = 'e';
}

int parseWFNfile (const char* filename, int* numOfMolOrb, int* numOfPrimFunc, char*** molName, struct pointD** origin,
					double** charge, int** types, int** molTypeFunc, int** molPowType, double** exponents, double** occNo,
					double** orbEnergy, double*** molOrbitals, int* numOfNucl, struct pointI** powcoef)
{
	FILE *stream;

	stream = fopen(filename, "r");
	if (stream == NULL)
		exit(EXIT_FAILURE);

	char *string_to_read = NULL;
	ssize_t read_sym = 0;
	size_t len = 0;
	char * pch; // strtok

	printf("Begin parsing file %s.\n", filename );
	read_sym = getline(&string_to_read, &len, stream);//skip first line

	memset(string_to_read,0,read_sym );
	read_sym = getline(&string_to_read, &len, stream);
	pch = strtok (string_to_read," ");

	pch = strtok (NULL, " ");
	*numOfMolOrb = atoi(pch);
	printf ("MOL ORBITALS %d\n",*numOfMolOrb);

	pch = strtok (NULL, " ");
	pch = strtok (NULL, " ");
	pch = strtok (NULL, " ");

	*numOfPrimFunc = atoi(pch);
	printf ("PRIMITIVES %d\n",*numOfPrimFunc);

	pch = strtok (NULL, " ");
	pch = strtok (NULL, " ");

	*numOfNucl = atoi(pch);
	printf ("NUCLEI %d\n", *numOfNucl);

	*molName = 		(char**) malloc( *numOfNucl * sizeof(char*) );
	*origin = 		(struct pointD*)malloc( *numOfNucl * sizeof(struct pointD) );

	*charge = 		(double*)malloc(*numOfNucl * sizeof(double) );//array of charges
	*molTypeFunc = 	(int *)	 malloc(*numOfPrimFunc * sizeof(int) );
	*types = 		(int*)	 malloc(*numOfNucl * sizeof(int));
	*molPowType = 	(int *)	 malloc(*numOfPrimFunc * sizeof(int) );
	*exponents = 	(double*)malloc(*numOfPrimFunc * sizeof(double) );
	*occNo = 		(double*)malloc(*numOfNucl * sizeof(double) );
	*orbEnergy = 	(double*)malloc(*numOfNucl * sizeof(double) );

	*molOrbitals = 	(double**)malloc(*numOfNucl * sizeof(double*));
	for (int molOrbitalIterator = 0; molOrbitalIterator < *numOfNucl; ++molOrbitalIterator)
		(*molOrbitals)[molOrbitalIterator] = (double*)malloc(*numOfPrimFunc * sizeof(double) );

	*powcoef = (struct pointI*)malloc( *numOfPrimFunc * sizeof(struct pointI) );

	for (int molcounter = 0; molcounter < *numOfNucl;  ++molcounter)
	{
		read_sym = getline(&string_to_read, &len, stream);

		pch = strtok (string_to_read," ");
		(*molName)[molcounter] = (char*)malloc((strlen(pch) + 1)*_SIZE_OF_CHAR );
		memset((*molName)[molcounter],0,strlen(pch) + 1 );
		strcpy((*molName)[molcounter], pch);
		printf ("%s\n",(*molName)[molcounter]);

		pch = strtok (NULL," ");

		*types[molcounter] = atoi (pch);
		printf ("Type of %s is %d\n",(*molName)[molcounter], (*types)[molcounter]);

		pch = strtok (NULL,")");

		pch = strtok (NULL," ");
		(*origin)[molcounter].x = atof(pch);
		pch = strtok (NULL," ");
		(*origin)[molcounter].y = atof(pch);
		pch = strtok (NULL," ");
		(*origin)[molcounter].z = atof(pch);

		printf ("%s's origin is (%f, %f, %f)\n",(*molName)[molcounter], (*origin)[molcounter].x, (*origin)[molcounter].y, (*origin)[molcounter].z);
		pch = strtok (NULL,"=");
		pch = strtok (NULL," ");

		(*charge)[molcounter] = atof(pch);
		printf ("%s's charge is %f\n",(*molName)[molcounter], (*charge)[molcounter]);
	}

	printf ("CENTRE ASSIGNMENTS: ");
	for (int i = 0; i < *numOfPrimFunc; ++i)
	{
		if(i%20 == 0)
		{
			read_sym = getline(&string_to_read, &len, stream);
			pch = strtok (string_to_read," ");
			pch = strtok (NULL," ");
			pch = strtok (NULL," ");
		}
		(*molTypeFunc)[i] = atoi(pch) - 1;
		printf (" %d", (*molTypeFunc)[i] + 1 );
		pch = strtok (NULL," ");
	}
	printf ("\n");

	printf ("TYPE ASSIGNMENTS: ");
	for (int i = 0; i < *numOfPrimFunc; ++i)
	{
		if(i%20 == 0)
		{
			read_sym = getline(&string_to_read, &len, stream);
			pch = strtok (string_to_read," ");
			pch = strtok (NULL," ");
			pch = strtok (NULL," ");
		}
		(*molPowType)[i] = atoi(pch);
		printf (" %d", (*molPowType)[i]);
		pch = strtok (NULL," ");
	}
	printf ("\n");


	fillExpTypes(*molPowType, *numOfPrimFunc, *powcoef);

	printf ("Exponents: ");

	for(int parseCounter = 0; parseCounter < *numOfPrimFunc; ++parseCounter)
	{
		if (parseCounter%5 == 0)//for each new line
		{
			read_sym = getline(&string_to_read, &len, stream);

			fixScientificNotation(string_to_read, read_sym);

			pch = strtok (string_to_read," ");
		}
		pch = strtok (NULL, " ");

		(*exponents)[parseCounter] = atof(pch);
		printf ("%f ", (*exponents)[parseCounter]);
	}
	printf ("\n");

	for (int molcounter = 0; molcounter < *numOfNucl;  ++molcounter)
	{
		printf ("Molelular orbital %d", molcounter);
		read_sym = getline(&string_to_read, &len, stream);
		fixScientificNotation(string_to_read, read_sym);

		pch = strtok (string_to_read,"=");
		pch = strtok (NULL, " ");
		(*occNo)[molcounter] = atof(pch);
		printf("OCC NO = %f\n", (*occNo)[molcounter]);
		pch = strtok (NULL,"=");
		pch = strtok (NULL, " ");
		(*orbEnergy)[molcounter] = atof(pch);
		printf("ORB ENERGY = %f\n", (*orbEnergy)[molcounter]);

		printf("Mol orbital coeficients: \n");
		for (int primFuncIterator = 0; primFuncIterator < *numOfPrimFunc; ++primFuncIterator)
		{
			if (primFuncIterator%5 == 0)//each 6th element should be read from the new line
			{
				read_sym = getline(&string_to_read, &len, stream);
				fixScientificNotation(string_to_read, read_sym);
				pch = strtok (string_to_read," ");
			}

			(*molOrbitals)[molcounter][primFuncIterator] = atof(pch);
			printf(" %f", (*molOrbitals)[molcounter][primFuncIterator]);
			pch = strtok (NULL, " ");

		}
		printf("\n");
	}

	fprintf(stderr, "End parsing file %s.\n", filename );

	free(string_to_read);
	string_to_read = NULL;

	fclose(stream);

	return 0;
}


// int releaseMemory (const char* filename, int& numOfMolOrb, int& numOfPrimFunc, char**& molName, pointD*& origin,
// 					double*& charge, int*& types, int*& molTypeFunc, int*& molPowType, double*& exponents, double*& occNo,
// 					double*& orbEnergy, double**& molOrbitals, int& numOfNucl, pointI*& powcoef)
// {

// 	for (int molcounter = 0; molcounter < numOfNucl;  ++molcounter)
// 	{
// 		free(molName[molcounter]);
// 		molName[molcounter] = NULL;
// 	}
// 	free(molName);
// 	molName = NULL;

// 	free(origin);
// 	origin = NULL;
// 	free(charge);//array of charges
// 	charge = NULL;//array of charges
// 	free(types) ;
// 	types = NULL;
// 	free(molTypeFunc);
// 	molTypeFunc = NULL;
// 	free(molPowType);
// 	molPowType = NULL;
// 	free(exponents);
// 	exponents = NULL;
// 	free(occNo);
// 	occNo = NULL;
// 	free(orbEnergy);
// 	orbEnergy = NULL;
// 	for (int molOrbitalIterator = 0; molOrbitalIterator < numOfNucl; ++molOrbitalIterator)
// 	{
// 		free(molOrbitals[molOrbitalIterator]);
// 		molOrbitals[molOrbitalIterator] = NULL;
// 	}
// 	free(molOrbitals);
// 	molOrbitals = NULL;

// 	return 0;
// }

void fillExpTypes(const int molPowType[], const int numOfPrimFunc, struct pointI powcoef[])
{
	printf("Started exponent conversion\n");

	for (int i = 0; i < numOfPrimFunc; ++i)
	{
		switch ( molPowType[i] )
		{
			case 1 :
				powcoef[i].x = 0;
				powcoef[i].y = 0;
				powcoef[i].z = 0;
				break;
			case 2 :
				powcoef[i].x = 1;
				powcoef[i].y = 0;
				powcoef[i].z = 0;
				break;
			case 3 :
				powcoef[i].x = 0;
				powcoef[i].y = 1;
				powcoef[i].z = 0;
				break;
			case 4 :
				powcoef[i].x = 0;
				powcoef[i].y = 0;
				powcoef[i].z = 1;
				break;
			case 5 :
				powcoef[i].x = 2;
				powcoef[i].y = 0;
				powcoef[i].z = 0;
				break;
		}
	}

	printf("Finished exponent conversion\n");

	for (int i = 0; i < numOfPrimFunc; ++i)
		printf("coef for func [%d]\t= %d %d %d\n", i, powcoef[i].x, powcoef[i].y, powcoef[i].z);
}

void getDistanceComp(const double cur_x, const double cur_y, const double cur_z, const struct pointD* origin, const int numOfNucl, struct pointD* out, const char debug )
{
	for (int i = 0; i < numOfNucl; ++i)
	{
		out[i].x = cur_x - origin[i].x;
		out[i].y = cur_y - origin[i].y;
		out[i].z = cur_z - origin[i].z;
	}

	if (debug == 1)
	{
		printf("DEBUG print of DistanceComp:\n");
		for (int i = 0; i < numOfNucl; ++i)
			printf("nucl[%d] = %f\t%f\t%f\n", i, out[i].x, out[i].y, out[i].z);
		printf("\nThis is the last line(DistanceComp).\n");
	}
}

void getDistanceAbs(const struct pointD* calcPoint, const int numOfNucl, double* r_dist, const char debug )
{
	for (int i = 0; i < numOfNucl; ++i)
		r_dist[i] = sqrt( calcPoint[i].x*calcPoint[i].x + calcPoint[i].y*calcPoint[i].y + calcPoint[i].z*calcPoint[i].z );

	if (debug == 1)
	{
		printf("DEBUG print of getDistanceAbs:\n");
		for (int i = 0; i < numOfNucl; ++i)
			printf("nucl[%d] r_dist = %f\n", i, r_dist[i]);
		printf("This is the last line(getDistanceAbs).\n\n");
	}
}


void getGRHO(const int numOfPrimFunc, const int* molTypeFunc, const double* r_dist, const double* exponents, const struct pointD* calcPoint, const struct pointI* powcoef, double* primFunc, double** dermodvar, double*** hesmodvar, const char selection , const char debug )
{
	double temp;
	double partB;
	double partC;

	double powXto2[numOfPrimFunc];
	double powYto2[numOfPrimFunc];
	double powZto2[numOfPrimFunc];

	double powXtoX[numOfPrimFunc];
	double powYtoY[numOfPrimFunc];
	double powZtoZ[numOfPrimFunc];

	double ExpOfSumPow2ByNegExp[numOfPrimFunc];

	double xa = 0, yb = 0, zc = 0; //coefficients that have no meaning but improve performance
	double expmemb = 1;//coefficients that have no meaning but improve performance

	int cur_pos;

	for (int i = 0; i < numOfPrimFunc; ++i)
	{
		cur_pos = molTypeFunc[i];
		powXtoX[i] = pow(calcPoint[cur_pos].x, powcoef[i].x );
		powYtoY[i] = pow(calcPoint[cur_pos].y, powcoef[i].y );
		powZtoZ[i] = pow(calcPoint[cur_pos].z, powcoef[i].z );
	}

	for (int i = 0; i < numOfPrimFunc; ++i)
	{
		cur_pos = molTypeFunc[i];

		if (powcoef[i].x == 0)
		{
			xa = 1;
		}
		else
		{
			if (powcoef[i].x > 0 || calcPoint[cur_pos].x != 0 )
			{
				xa = powXtoX[i];
			}
			else
			{
				printf("WARNING, HANDLED EXCEPTION !!!! ( xa = 100000000000000 )\n");
				xa = 1E+10;
			}
		}

		if (powcoef[i].y == 0)
		{
			yb = 1;
		}
		else
		{
			if (powcoef[i].y > 0 || calcPoint[cur_pos].y != 0 )
			{
				yb = powYtoY[i];
			}
			else
			{
				printf("WARNING, HANDLED EXCEPTION !!!! ( yb = 100000000000000 )\n");
				yb = 1E+10;
			}
		}

		if (powcoef[i].z == 0)
		{
			zc = 1;
		}
		else
		{
			if (powcoef[i].z > 0 || calcPoint[cur_pos].z != 0 )
			{
				zc = powZtoZ[i];
			}
			else
			{
				printf("WARNING, HANDLED EXCEPTION !!!! ( zc = 100000000000000 )\n");
				zc = 1E+10;
			}
		}


		if (exponents[i] != 0 && r_dist[cur_pos] != 0 )
		{
			expmemb = exp( -exponents[i] * r_dist[cur_pos] * r_dist[cur_pos] );
		}
		else
		{
			if (r_dist[cur_pos] < 0)
				printf("WARNING, UNHANDLED EXCEPTION !!!! ( r_dist[cur_pos] < 0 )\n");

			expmemb = 1;
		}

		primFunc[i] = xa * yb * zc * expmemb;
	}

	if (debug == 1)
	{
		printf("DEBUG print of primitive functions:\n");
		for (int i = 0; i < numOfPrimFunc; ++i)
		{
			if (i%3 == 0)
				printf("\n");
			printf("pf[%d] = %f \t", i, primFunc[i]);
		}
		printf("\nThis is the last line(prim func).\n");
	}


	if(selection == PRIM_FUNC)
		return;
// GRADIENT ***********************************************************


	for (int i = 0; i < numOfPrimFunc; ++i)
	{
		cur_pos = molTypeFunc[i];
		powXto2[i] = calcPoint[cur_pos].x * calcPoint[cur_pos].x;
		powYto2[i] = calcPoint[cur_pos].y * calcPoint[cur_pos].y;
		powZto2[i] = calcPoint[cur_pos].z * calcPoint[cur_pos].z;

		ExpOfSumPow2ByNegExp[i] = exp( -exponents[i] * ( powXto2[i] + powYto2[i] + powZto2[i] ) );
	}

	for (int i = 0; i < numOfPrimFunc; ++i)
	{
		cur_pos = molTypeFunc[i];

		if (primFunc[i] == 0 || calcPoint[cur_pos].x == 0)
		{
			if( powcoef[i].x == 0 )
				dermodvar[i][0] = -ExpOfSumPow2ByNegExp[i] * powYtoY[i] * powZtoZ[i] * 2 * exponents[i] * calcPoint[cur_pos].x ;
			else
				dermodvar[i][0] =   ExpOfSumPow2ByNegExp[i] * powYtoY[i] * powZtoZ[i]
									* ( powcoef[i].x * pow(calcPoint[cur_pos].x, (powcoef[i].x - 1 )) - 2 * exponents[i] * powXtoX[i] * calcPoint[cur_pos].x );
		}
		else
			dermodvar[i][0] = primFunc[i] * ( powcoef[i].x / calcPoint[cur_pos].x - 2 * exponents[i] * calcPoint[cur_pos].x );

		if (primFunc[i] == 0 || calcPoint[cur_pos].y == 0 )
		{
			if( powcoef[i].y == 0 )
				dermodvar[i][1] = -ExpOfSumPow2ByNegExp[i] * powXtoX[i] * powZtoZ[i] * 2 * exponents[i] * calcPoint[cur_pos].y ;
			else
				dermodvar[i][1] =   ExpOfSumPow2ByNegExp[i] * powXtoX[i] * powZtoZ[i]
									* ( powcoef[i].y * pow(calcPoint[cur_pos].y, (powcoef[i].y - 1 )) - 2 * exponents[i] * powYtoY[i] * calcPoint[cur_pos].y );
		}
		else
			dermodvar[i][1] = primFunc[i] * ( powcoef[i].y / calcPoint[cur_pos].y - 2 * exponents[i] * calcPoint[cur_pos].y );

		if (primFunc[i] == 0 || calcPoint[cur_pos].z == 0)
		{

			if( powcoef[i].z == 0 )
				dermodvar[i][2] = -ExpOfSumPow2ByNegExp[i] * powYtoY[i] * powXtoX[i] * 2 * exponents[i] * calcPoint[cur_pos].z ;
			else
				dermodvar[i][2] =   ExpOfSumPow2ByNegExp[i] * powYtoY[i] * powXtoX[i]
									* ( powcoef[i].z * pow(calcPoint[cur_pos].z, (powcoef[i].z - 1 )) - 2 * exponents[i] * powZtoZ[i] * calcPoint[cur_pos].z );
		}
		else
			dermodvar[i][2] = primFunc[i] * ( powcoef[i].z / calcPoint[cur_pos].z - 2 * exponents[i] * calcPoint[cur_pos].z );
	}


	if (debug == 1)
	{
		printf("DEBUG print of gradient:\tx\ty\tz\n");
		for (int i = 0; i < numOfPrimFunc; ++i)
			printf("  (%d)\t%f,\t%f,\t%f\n", i, dermodvar[i][0], dermodvar[i][1], dermodvar[i][2]);
		printf("\nThis is the last line(gradient).\n");
	}

	if(selection == GRAD)
		return;

	// HESSIAN ****************************************************************************

	for (int i = 0; i < numOfPrimFunc; ++i)
	{
		cur_pos = molTypeFunc[i];
// FOR XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX
		if ( dermodvar[i][0] == 0 || calcPoint[cur_pos].x == 0 )
		{
			if( powcoef[i].x > 2 )
				hesmodvar[i][0][0] =  ExpOfSumPow2ByNegExp[i] * powYtoY[i] * powZtoZ[i] *
					( (powcoef[i].x - 1) * powcoef[i].x * pow(calcPoint[cur_pos].x, powcoef[i].x - 2 ) +
						4 * exponents[i] * powXtoX[i] * (- powcoef[i].x - 0.5 + exponents[i] * powXto2[i] ) );
			else
			{
				if( powcoef[i].x == 1 )
				{
					if( calcPoint[cur_pos].x == 0 )
						hesmodvar[i][0][0] = 0;
					else
						hesmodvar[i][0][0] = ExpOfSumPow2ByNegExp[i] * powYtoY[i] * powZtoZ[i] * 2 * exponents[i] * calcPoint[cur_pos].x * ( -3 + 2 * exponents[i] * powXto2[i] );
				}
				else
				{
					if( powcoef[i].x == 0 )
					{
						if( calcPoint[cur_pos].x == 0 )
							hesmodvar[i][0][0] = exp(-exponents[i] * ( powYto2[i] + powZto2[i] ) ) * ( -2 * exponents[i] *  powYtoY[i] * powZtoZ[i] );
						else
							hesmodvar[i][0][0] = exp(-exponents[i] * ( powXto2[i] + powYto2[i] + powZto2[i]) ) * powYtoY[i] * powZtoZ[i] * ( -2 * exponents[i] + 4 * pow(exponents[i], 2) * powXtoX[i] );
					}
    			}
			}
		}
	    else
	    {
	    	temp = powcoef[i].x - 2 * exponents[i] * powXto2[i];
		    if( temp == 0)
		    {
		        partB = 1E+10; // very bad, but this is usually never met
		        printf("If you see this, go to the source code and find out that first power coeficient has bad value\n");
		    }
		    else
		        partB = 2* exponents[i] * calcPoint[cur_pos].x * ( 1 + 2/temp );

		    if((powcoef[i].x - 1) == 0)
		        hesmodvar[i][0][0] = - dermodvar[i][0] * partB;
		    else
		        hesmodvar[i][0][0] = dermodvar[i][0] * ( (powcoef[i].x - 1)/ calcPoint[cur_pos].x - partB );
	    }

	    // printf("xx[%d] = %f\n", i, hesmodvar[i][0][0]);

// for YY YY YY YY YY YY YY YY YY YY YY YY YY YY YY YY YY YY

	    if ( dermodvar[i][1] == 0 || calcPoint[cur_pos].y == 0 )
		{
			if( powcoef[i].y > 2 )
				hesmodvar[i][1][1] =  ExpOfSumPow2ByNegExp[i] * powXtoX[i] * powZtoZ[i] *
					( (powcoef[i].y - 1) * powcoef[i].y * pow(calcPoint[cur_pos].y, powcoef[i].y - 2 ) +
						4 * exponents[i] * powYtoY[i] * (- powcoef[i].y - 0.5 + exponents[i] * powYto2[i] ) );
			else
			{
				if( powcoef[i].y == 1 )
				{
					if( calcPoint[cur_pos].y == 0 )
						hesmodvar[i][1][1] = 0;
					else
						hesmodvar[i][1][1] = ExpOfSumPow2ByNegExp[i] * powXtoX[i] * powZtoZ[i] * 2 * exponents[i] * calcPoint[cur_pos].y * ( -3 + 2 * exponents[i] * powYto2[i] );
				}
				else
				{
					if( powcoef[i].y == 0 )
					{
						if( calcPoint[cur_pos].y == 0 )
							hesmodvar[i][1][1] = exp(-exponents[i] * ( powXto2[i] + powZto2[i] ) ) * ( -2 * exponents[i] *  powXtoX[i] * powZtoZ[i] );
						else
							hesmodvar[i][1][1] = exp(-exponents[i] * ( powYto2[i] + powXto2[i] + powZto2[i]) ) * powXtoX[i] * powZtoZ[i] * ( -2 * exponents[i] + 4 * pow(exponents[i], 2) * powYto2[i] );
					}
    			}
			}
		}
	    else
	    {
	    	temp = powcoef[i].y - 2 * exponents[i] * powYto2[i];
		    if( temp == 0)
		    {
		        partB = 10000000000; // very bad, but this is usually never met
		        printf("If you see this, go to the source code and find out that first power coeficient has bad value\n");
		    }
		    else
		        partB = 2* exponents[i] * calcPoint[cur_pos].y * ( 1 + 2/temp );

		    if((powcoef[i].y - 1) == 0)
		        hesmodvar[i][1][1] = - dermodvar[i][1] * partB;
		    else
		        hesmodvar[i][1][1] = dermodvar[i][1] * ( (powcoef[i].y - 1)/ calcPoint[cur_pos].y - partB );
	    }


	    // printf("yy[%d] = %f\n", i, hesmodvar[i][1][1]);

// FOR ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ ZZ

	    if ( dermodvar[i][2] == 0 || calcPoint[cur_pos].z == 0 )
		{
			if( powcoef[i].z > 2 )
				hesmodvar[i][2][2] =  ExpOfSumPow2ByNegExp[i] * powXtoX[i] * powYtoY[i] *
					( (powcoef[i].z - 1) * powcoef[i].z * pow(calcPoint[cur_pos].z, powcoef[i].z - 2 ) +
						4 * exponents[i] * powZtoZ[i] * (- powcoef[i].z - 0.5 + exponents[i] * powZto2[i] ) );
			else
			{
				if( powcoef[i].z == 1 )
				{
					if( calcPoint[cur_pos].z == 0 )
						hesmodvar[i][2][2] = 0;
					else
						hesmodvar[i][2][2] = ExpOfSumPow2ByNegExp[i] * powYtoY[i] * powXtoX[i] * 2 * exponents[i] * calcPoint[cur_pos].z * ( -3 + 2 * exponents[i] * powZto2[i] );
				}
				else
				{
					if( powcoef[i].z == 0 )
					{
						if( calcPoint[cur_pos].z == 0 )
							hesmodvar[i][2][2] = exp(-exponents[i] * ( powYto2[i] + powXto2[i] ) ) * ( -2 * exponents[i] *  powYtoY[i] * powXtoX[i] );
						else
							hesmodvar[i][2][2] = exp(-exponents[i] * ( powZto2[i] + powYto2[i] + powXto2[i]) ) * powYtoY[i] * powXtoX[i] * ( -2 * exponents[i] + 4 * pow(exponents[i], 2) * powZto2[i] );
					}
    			}
			}
		}
	    else
	    {
	    	temp = powcoef[i].z - 2 * exponents[i] * powZto2[i];
		    if( temp == 0)
		    {
		        partB = 10000000000; // very bad, but this is usually never met
		        printf("If you see this, go to the source code and find out that power coeficient has bad value\n");
		    }
		    else
		        partB = 2* exponents[i] * calcPoint[cur_pos].z * ( 1 + 2/temp );

		    if((powcoef[i].z - 1) == 0)
		        hesmodvar[i][2][2] = - dermodvar[i][2] * partB;
		    else
		        hesmodvar[i][2][2] = dermodvar[i][2] * ( (powcoef[i].z - 1)/calcPoint[cur_pos].z - partB );
	    }
	    // printf("zz[%d] = %f\n", i, hesmodvar[i][2][2]);

// FOR XY XY XY XY XY XY XY XY XY XY XY XY XY XY XY XY XY XY XY XY XY

	    if (dermodvar[i][0] == 0 || /*calcPoint[cur_pos].x == 0 ||*/ calcPoint[cur_pos].y == 0 )
		{
			if( powcoef[i].x > 1 )
			    partB = pow(calcPoint[cur_pos].x, powcoef[i].x - 1) * (powcoef[i].x - 2 * exponents[i] * powXtoX[i] );
			else if( powcoef[i].x == 1 )
			    partB = 1 - 2 * exponents[i] * powXtoX[i];
			else if( powcoef[i].x == 0 )
			    partB =  - 2 * exponents[i] * calcPoint[cur_pos].x;
			else
			{
				partB = 0;
				printf("EXEPTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			}


			if ( powcoef[i].y > 0 )
			    partC = pow(calcPoint[cur_pos].y, powcoef[i].y - 1) * ( powcoef[i].y - 2 * exponents[i] * powYto2[i] );
			else if( powcoef[i].y == 1 )
			    partC = 1 - 2 * exponents[i] * powYto2[i];
			else if( powcoef[i].y == 0 )
			    partC =  calcPoint[cur_pos].y * 2 * exponents[i];
			else
			{
				partC = 0;
				printf("EXEPTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			}

			hesmodvar[i][0][1] = ExpOfSumPow2ByNegExp[i] * partB * partC * powZtoZ[i];
		}
		else
			hesmodvar[i][0][1] = dermodvar[i][0] * ( powcoef[i].y / calcPoint[cur_pos].y - 2 * exponents[i] * calcPoint[cur_pos].y );

		// printf("xy[%d] = %f\n", i, hesmodvar[i][0][1]);

// FOR XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ XZ

		if (dermodvar[i][0] == 0 || /*calcPoint[cur_pos].x == 0 || */calcPoint[cur_pos].z == 0 )
		{
			if( powcoef[i].x > 1 )
			    partB = pow(calcPoint[cur_pos].x, powcoef[i].x - 1) * (powcoef[i].x - 2 * exponents[i] * powXtoX[i] );
			else if( powcoef[i].x == 1 )
			    partB = 1 - 2 * exponents[i] * powXtoX[i];
			else if( powcoef[i].x == 0 )
			    partB =  - 2 * exponents[i] * calcPoint[cur_pos].x;
			else
			{
				partB = 0;
				printf("EXEPTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			}


			if ( powcoef[i].z > 0 )
			    partC = pow(calcPoint[cur_pos].z, powcoef[i].z - 1) * ( powcoef[i].z - 2 * exponents[i] * powZto2[i] );
			else if( powcoef[i].z == 1 )
			    partC = 1 - 2 * exponents[i] * powZto2[i];
			else if( powcoef[i].z == 0 )
			    partC =  calcPoint[cur_pos].z * 2 * exponents[i];
			else
			{
				partC = 0;
				printf("EXEPTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			}

			hesmodvar[i][0][2] = ExpOfSumPow2ByNegExp[i] * partB * partC * powYtoY[i];
		}
		else
			hesmodvar[i][0][2] = dermodvar[i][0] * ( powcoef[i].z / calcPoint[cur_pos].z - 2 * exponents[i] * calcPoint[cur_pos].z );

		// printf("xz[%d] = %f\n", i, hesmodvar[i][0][2]);

// FOR YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ YZ

		if (dermodvar[i][1] == 0 || /*calcPoint[cur_pos].y == 0 || */calcPoint[cur_pos].z == 0 )
		{
			if( powcoef[i].y > 1 )
			    partB = pow( calcPoint[cur_pos].y, powcoef[i].y - 1 ) * ( powcoef[i].y - 2 * exponents[i] * powYto2[i] );
			else if( powcoef[i].y == 1 )
			    partB = ( 1 - 2 * exponents[i] * powYto2[i] );
			else if( powcoef[i].y == 0 )
			    partB =  - 2 * exponents[i] * calcPoint[cur_pos].y;
			else
			{
				partB = 0;
				printf("EXEPTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			}


			if ( powcoef[i].z > 0 )
			    partC = pow(calcPoint[cur_pos].z, powcoef[i].z - 1) * ( powcoef[i].z - 2 * exponents[i] * powZto2[i] );
			else if( powcoef[i].z == 1 )
			    partC = 1 - 2 * exponents[i] * powZto2[i];
			else if( powcoef[i].z == 0 )
			    partC =  calcPoint[cur_pos].z * 2 * exponents[i];
			else
			{
				partC = 0;
				printf("EXEPTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			}

			hesmodvar[i][1][2] = ExpOfSumPow2ByNegExp[i] * partB * partC * powXtoX[i];
		}
		else
			hesmodvar[i][1][2] = dermodvar[i][1] * ( powcoef[i].z / calcPoint[cur_pos].z - 2 * exponents[i] * calcPoint[cur_pos].z );

		// printf("yz[%d] = %f\n", i, hesmodvar[i][1][2]);

	    // printf(" ");
	    // printf("\n");
	}

	if (debug == 1)
	{
		printf("DEBUG print of Hessian(in matrix form for each function):\n");
		for (int i = 0; i < numOfPrimFunc; ++i)
		{
			printf("Now printing function #%d:\n\n", i);
			printf( "   / %f\t%f\t%f  \\ \n", hesmodvar[i][0][0], hesmodvar[i][0][1], hesmodvar[i][0][2]);
			printf( "  |  %f\t%f\t%f   | \n", hesmodvar[i][1][0], hesmodvar[i][1][1], hesmodvar[i][1][2]);
			printf("   \\ %f\t%f\t%f  / \n\n", hesmodvar[i][2][0], hesmodvar[i][2][1], hesmodvar[i][2][2]);
		}
		printf("\nThis is the last line(hessian).\n");
	}
}

void getHesPSI (const int numOfPrimFunc, const int numOfNucl, const double** molOrbitals, const double*** hesmodvar, double*** hespsi, const char debug)
// void getHesPSI (const int numOfPrimFunc, const int numOfNucl, const double* const* molOrbitals, const double*  const * const * hesmodvar, double*** hespsi, const char debug)
{
	// in this step we initialize hespsi(not zero, so we save one loop)
	for(int j = 0; j < numOfNucl; ++j)
	{
		hespsi[j][0][0] = molOrbitals[j][0] * hesmodvar[0][0][0];
		hespsi[j][0][1] = molOrbitals[j][0] * hesmodvar[0][0][1];
		hespsi[j][0][2] = molOrbitals[j][0] * hesmodvar[0][0][2];
		// hespsi[j][1][0] = molOrbitals[j][0] * hesmodvar[0][1][0];
		hespsi[j][1][1] = molOrbitals[j][0] * hesmodvar[0][1][1];
		hespsi[j][1][2] = molOrbitals[j][0] * hesmodvar[0][1][2];
		// hespsi[j][2][0] = molOrbitals[j][0] * hesmodvar[0][2][0];
		// hespsi[j][2][1] = molOrbitals[j][0] * hesmodvar[0][2][1];
		hespsi[j][2][2] = molOrbitals[j][0] * hesmodvar[0][2][2];
	}

	for(int j = 0; j < numOfNucl; ++j)
		for(int i = 1; i < numOfPrimFunc; ++i)
		{
			hespsi[j][0][0] += molOrbitals[j][i] * hesmodvar[i][0][0];
			hespsi[j][0][1] += molOrbitals[j][i] * hesmodvar[i][0][1];
			hespsi[j][0][2] += molOrbitals[j][i] * hesmodvar[i][0][2];
			// hespsi[j][1][0] += molOrbitals[j][i] * hesmodvar[i][1][0];
			hespsi[j][1][1] += molOrbitals[j][i] * hesmodvar[i][1][1];
			hespsi[j][1][2] += molOrbitals[j][i] * hesmodvar[i][1][2];
			// hespsi[j][2][0] += molOrbitals[j][i] * hesmodvar[i][2][0];
			// hespsi[j][2][1] += molOrbitals[j][i] * hesmodvar[i][2][1];
			hespsi[j][2][2] += molOrbitals[j][i] * hesmodvar[i][2][2];
		}

	for(int j = 0; j < numOfNucl; ++j)
	{
		hespsi[j][1][0] = hespsi[j][0][1];
		hespsi[j][2][0] = hespsi[j][0][2];
		hespsi[j][2][1] = hespsi[j][1][2];
	}

	if (debug == 1)
	{
		printf("DEBUG print of Hessian PSI(in matrix form for each function):\n");
		for (int i = 0; i < numOfNucl; ++i)
		{
			printf("Now printing function #%d:\n\n", i);
			printf( "   / %f\t%f\t%f  \\ \n", hespsi[i][0][0], hespsi[i][0][1], hespsi[i][0][2]);
			printf( "  |  %f\t%f\t%f   | \n", hespsi[i][1][0], hespsi[i][1][1], hespsi[i][1][2]);
			printf("   \\ %f\t%f\t%f  / \n\n", hespsi[i][2][0], hespsi[i][2][1], hespsi[i][2][2]);
		}
		printf("\nThis is the last line(hess psi).\n");
	}
}

void convcoord(const double* x, const double* y, const double* z, double* r, double* thet, double* phi )
{
	*r = sqrt( *x * *x + *y * *y + *z * *z );
    if( *r == 0 )
    	*thet = acos( *z/(0.0000000000001) );
    else
        *thet = acos( *z / *r );
    // WARNING division by zero
    *phi = atan2(*y,*x);
}

void backconvcoord(const double* r, const double* thet, const double* phi, double* x, double* y, double* z )
{
	double temp = *r * sin(*thet);
    *x = temp * cos(*phi);
    *y = temp * sin(*phi);
    *z = *r   * cos(*thet);
}

void getPsi (const int numOfPrimFunc, const int numOfNucl, const double** molOrbitals, const double* primFunc, double* psi, const char debug)
// void getPsi (const int numOfPrimFunc, const int numOfNucl, const double* const* molOrbitals, const double* primFunc, double* psi, const char debug)
{
	for (int i = 0; i < numOfNucl; ++i)
		psi[i]	= molOrbitals[i][0] * primFunc[0];

	for (int i = 0; i < numOfNucl; ++i)
		for (int j = 1; j < numOfPrimFunc; ++j)
		{
			psi[i]	+= molOrbitals[i][j] * primFunc[j];
//			printf("result[%d][%d](%d) = %f\n", i, j, i*numOfPrimFunc + j, molOrbitals[i][j] * primFunc[j]);
		}

	if (debug == 1)
		printf("DEBUG print of Psi:  \t%f,\t%f\nThis is the last line(Psi).\n\n", psi[0], psi[1]);
}

void getDPsi (const int numOfPrimFunc, const int numOfNucl, const double** molOrbitals, const double** dermodvar, double** dpsi, const char debug)
// void getDPsi (const int numOfPrimFunc, const int numOfNucl, const double* const* molOrbitals, const double* const* dermodvar, double** dpsi, const char debug)
{

	for (int i = 0; i < numOfNucl; ++i)
	{
		dpsi[i][0]	= molOrbitals[i][0] * dermodvar[0][0];
//		printf("dpsi[%d][0] = %f * %f = %f\n", i, molOrbitals[i][0], dermodvar[0][0], dpsi[i][0]);

		dpsi[i][1]	= molOrbitals[i][0] * dermodvar[0][1];
//		printf("dpsi[%d][1] = %f * %f = %f\n", i, molOrbitals[i][0], dermodvar[0][1], dpsi[i][1]);
		dpsi[i][2]	= molOrbitals[i][0] * dermodvar[0][2];
//		printf("dpsi[%d][2] = %f * %f = %f\n", i, molOrbitals[i][0], dermodvar[0][2], dpsi[i][2]);
	}

//	for (int j = 1; j < numOfPrimFunc; ++j){
//	printf("dermodvar[%d][0] = %5.16f\n", j, dermodvar[j][0]);
//	printf("dermodvar[%d][1] = %5.16f\n", j, dermodvar[j][1]);
//	printf("dermodvar[%d][2] = %5.16f\n", j, dermodvar[j][2]);
//	}

//	for (int i = 0; i < numOfNucl; ++i)
//			for (int j = 0; j < numOfPrimFunc; ++j)
//				printf("molOrbitals[%d] = %5.16f\n",j*2 + i, molOrbitals[i][j]);

	for (int i = 0; i < numOfNucl; ++i)
		for (int j = 1; j < numOfPrimFunc; ++j)
		{
			dpsi[i][0] += molOrbitals[i][j] * dermodvar[j][0];
//			printf("dpsi[%d][0][%d] = %f * %f = %f\n", i, i*numOfPrimFunc + j, molOrbitals[i][j], dermodvar[j][0], molOrbitals[i][j] * dermodvar[j][0]);
			dpsi[i][1] += molOrbitals[i][j] * dermodvar[j][1];
//			printf("dpsi[%d][1][%d] = %f * %f = %f\n", i, i*numOfPrimFunc + j, molOrbitals[i][j], dermodvar[j][1], molOrbitals[i][j] * dermodvar[j][1]);
			dpsi[i][2] += molOrbitals[i][j] * dermodvar[j][2];
//			printf("dpsi[%d][2][%d] = %f * %f = %f\n", i, i*numOfPrimFunc + j, molOrbitals[i][j], dermodvar[j][2], molOrbitals[i][j] * dermodvar[j][2]);
		}

	if (debug == 1)
	{
		printf("DEBUG print of DPsi:\n");

		for (int i = 0; i < numOfNucl; ++i)
			printf(" (%d) \t%f,\t%f,\t%f\n", i,dpsi[i][0], dpsi[i][1], dpsi[i][2]);

		printf("This is the last line(DPsi).\n\n");
	}
}

void getDRho(const int numOfNucl, const double* psi, const double** dpsi, const double* occNo, double* drho, const char debug)
// void getDRho(const int numOfNucl, const double* psi, const double* const* dpsi, const double* occNo, double* drho, const char debug)
{
	drho[0] = 0;
	drho[1] = 0;
	drho[2] = 0;

	for (int i = 0; i < numOfNucl; ++i)
	{
		drho[0] = drho[0] + 2* occNo[i] * psi[i] * dpsi[i][0];
//		printf("drho[0] = %f + 2 * %f * %f * %f (%f)\n",drho[0], occNo[i], psi[i], dpsi[i][0], 2* occNo[i] * psi[i] * dpsi[i][0] );
		drho[1] = drho[1] + 2* occNo[i] * psi[i] * dpsi[i][1];
//		printf("drho[1] = %f + 2 * %f * %f * %f (%f)\n",drho[1], occNo[i], psi[i], dpsi[i][1], 2* occNo[i] * psi[i] * dpsi[i][1] );
		drho[2] = drho[2] + 2* occNo[i] * psi[i] * dpsi[i][2];
//		printf("drho[2] = %f + 2 * %f * %f * %f (%f)\n",drho[2], occNo[i], psi[i], dpsi[i][2], 2* occNo[i] * psi[i] * dpsi[i][2] );
	}

	if (debug == 1)
		printf("DEBUG print of DRHO:\n \t%f\t%f\t%f\nThis is the last line(DRHO).\n\n", drho[0], drho[1], drho[2]);
}

void getRho(const int numOfNucl, const double* psi, const double* occNo, double* rho, const char debug)
{
	*rho = 0;

	for (int i = 0; i < numOfNucl; ++i)
		*rho += occNo[i] * psi[i] * psi[i];

	if (debug == 1)
		printf("DEBUG print of RHO:\n RHO = %f\nThis is the last line(RHO).\n\n", *rho);
}

double myNormS ( const double a, const double b, const double c)
{
	return ( sqrt(a*a + b*b + c*c ) );
}

double myNormV ( const double* arr)
{
	return ( sqrt(arr[0]*arr[0] + arr[1]*arr[1] + arr[2]*arr[2] ) );
}

void getHesRho (const int numOfNucl, const double* psi, const double** dpsi, const double* occNo, const double*** hespsi, double** hesrho, const char debug)
// void getHesRho (const int numOfNucl, const double* psi, const double*  const * dpsi, const double* occNo, const double*  const * const * hespsi, double** hesrho, const char debug)
{
	hesrho[0][0] = 0;
	hesrho[0][1] = 0;
	hesrho[0][2] = 0;
	hesrho[1][1] = 0;
	hesrho[1][2] = 0;
	hesrho[2][2] = 0;

	for (int i = 0; i < numOfNucl; ++i)
	{
		hesrho[0][0] = hesrho[0][0] + 2 * occNo[i] * ( psi[i] * hespsi[i][0][0] + dpsi[i][0] * dpsi[i][0] );
		hesrho[0][1] = hesrho[0][1] + 2 * occNo[i] * ( psi[i] * hespsi[i][0][1] + dpsi[i][0] * dpsi[i][1] );
		hesrho[0][2] = hesrho[0][2] + 2 * occNo[i] * ( psi[i] * hespsi[i][0][2] + dpsi[i][0] * dpsi[i][2] );
		hesrho[1][1] = hesrho[1][1] + 2 * occNo[i] * ( psi[i] * hespsi[i][1][1] + dpsi[i][1] * dpsi[i][1] );
		hesrho[1][2] = hesrho[1][2] + 2 * occNo[i] * ( psi[i] * hespsi[i][1][2] + dpsi[i][1] * dpsi[i][2] );
		hesrho[2][2] = hesrho[2][2] + 2 * occNo[i] * ( psi[i] * hespsi[i][2][2] + dpsi[i][2] * dpsi[i][2] );
	}

	hesrho[1][0] = hesrho[0][1];
	hesrho[2][0] = hesrho[0][2];
	hesrho[2][1] = hesrho[1][2];

	if (debug == 1)
	{
		printf("DEBUG print of HesRho:\n");
			printf( "   / %f\t%f\t%f  \\ \n", hesrho[0][0], hesrho[0][1], hesrho[0][2]);
			printf( "  |  %f\t%f\t%f   | \n", hesrho[1][0], hesrho[1][1], hesrho[1][2]);
			printf("   \\ %f\t%f\t%f  / \n", hesrho[2][0], hesrho[2][1], hesrho[2][2]);
		printf("This is the last line(HesRho).\n");
	}
}


void reduction(double a[][6], int size, int pivot, int col)
{
   int i, j;
   double factor = a[pivot][col];

	for (i = 0; i < 2 * size; i++)
		a[pivot][i] /= factor;

	for (i = 0; i < size; i++)
	{
		if (i != pivot)
		{
			factor = a[i][col];
			for (j = 0; j < 2 * size; j++)
				a[i][j] = a[i][j] - a[pivot][j] * factor;
		}
	}
}

void matrINV(const double** in_matr, double** out_matr, const char debug)
// void matrINV(const double* const * in_matr, double** out_matr, const char debug)
{
	double matrix[3][6];

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 6; j++)
			if (j == i + 3)
				matrix[i][j] = 1;
			else
				matrix[i][j] = 0;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			matrix[i][j] = in_matr[i][j];

	for (int i = 0; i < 3; i++)
		reduction(matrix, 3, i, i);

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			out_matr[i][j] = matrix[i][j + 3];

	if (debug == 1)
	{
		printf("DEBUG print of matrINV:\n");

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
			{
				if (j == 0)
					printf("\n");
				printf("%f  \t", out_matr[i][j]);
			}

		printf("\nThis is the last line(matrINV).\n");
	}
}
