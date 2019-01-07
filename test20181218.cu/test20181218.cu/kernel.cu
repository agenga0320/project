#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include<cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define NUM_THREADS 512

bool InitCUDA()


{

	int count;

	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}

	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}

__global__ static void matrixmul(const int* a,const int* b, int* ans, int n)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int idx = bid * blockDim.x + tid;
	const int row = idx / n;
	const int column = idx % n;
	int i;

	if (row < n && column < n) {
		int t = 0;
		for (i = 0; i < n; i++) {
			t += a[row * n + i] * b[i * n + column];
		}
		ans[row * n + column] = t;
	}
}

int *cudamatrixmul(int *ma, int *mb, int n)
{
	int *gpuma, *gpumb, *gpuans, *ans;

	ans = (int*)malloc(sizeof(int)*n*n);

	cudaMalloc((void**)&gpuma, sizeof(int) * n * n);
	cudaMalloc((void**)&gpumb, sizeof(int) * n * n);
	cudaMalloc((void**)&gpuans, sizeof(int) * n * n);

	cudaMemcpy2D(gpuma, sizeof(int) * n, ma, sizeof(int) * n, sizeof(int) * n, n, cudaMemcpyHostToDevice);
	cudaMemcpy2D(gpumb, sizeof(int) * n, mb, sizeof(int) * n, sizeof(int) * n, n, cudaMemcpyHostToDevice);

	int blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
	matrixmul << <blocks * n, NUM_THREADS >> > (gpuma, gpumb, gpuans, n);

	cudaMemcpy2D(ans, sizeof(int) * n, gpuans, sizeof(int) * n, sizeof(int) * n, n, cudaMemcpyDeviceToHost);
	/*
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			printf("%d ", ans[i * n + j]);
		printf("\n");
	}
	*/

	cudaFree(gpuma);
	cudaFree(gpumb);
	cudaFree(gpuans);
	return ans;
}

void matrixmul(int *ma, int *mb, int *ans, int n)
{
	clock_t start, end;

	start = clock();

	memset(ans, 0, sizeof(ans));

	for (int i = 0; i < n; i++) 
		for (int j = 0; j < n; j++) {
			int t = 0;
			for (int k = 0; k < n; k++)
				t += ma[i * n + k] * mb[k * n + j];
			ans[i*n + j] = t;
		}
	/*
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			printf("%d ", ans[i * n + j]);
		printf("\n");
	}
	*/
	end = clock();
	printf("cpu:%ld\n\n", end - start);
}

int main()
{
	if (!InitCUDA()) {
		return 0;
	}

	int *ma, *mb, *ans;
	int n = 1000;
	clock_t start, end;
	
	while (n < 1040)
	{
		start = clock();

		ma = (int*)malloc(sizeof(int) * n * n);
		mb = (int*)malloc(sizeof(int) * n * n);
		ans = (int*)malloc(sizeof(int) * n * n);
		/*
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				scanf("%d", &ma[i * n + j]);

		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				scanf("%d", &mb[i * n + j]);
		*/

		srand(0);

		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				ma[i * n + j] = rand() % 10;

		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				mb[i * n + j] = rand() % 10;

		//matrixmul(ma, mb, ans, n);

		ans = cudamatrixmul(ma, mb, n);
		/*
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				printf("%d ", ans[i * n + j]);
			printf("\n");
		}
		*/
		free(ma);
		free(mb);
		free(ans);

		end = clock();
		int s = (int)start;
		int e = (int)end;
		printf("%d\n", e - s);
		n++;
	}

	return 0;
}