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

__global__ static void matrixmul(const int* a, size_t lda, const int* b, size_t ldb, int* ans, size_t ldans, int n)
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
			t += a[row * lda + i] * b[i * ldb + column];
		}
		ans[row * ldans + column] = t;
	}
}

int *cudamatrixmul(int *ma, int *mb, int n)
{
	int *gpuma, *gpumb, *gpuans ,*ans;

	ans = (int*)malloc(sizeof(int)*n*n);

	cudaMalloc((void**)&gpuma, sizeof(int) * n * n);
	cudaMalloc((void**)&gpumb, sizeof(int) * n * n);
	cudaMalloc((void**)&gpuans, sizeof(int) * n * n);

	cudaMemcpy2D(gpuma, sizeof(int) * n, ma, sizeof(int) * n, sizeof(int) * n, n, cudaMemcpyHostToDevice);
	cudaMemcpy2D(gpumb, sizeof(int) * n, mb, sizeof(int) * n, sizeof(int) * n, n, cudaMemcpyHostToDevice);

	int blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
	matrixmul << <blocks * n, NUM_THREADS >> > (gpuma, n, gpumb, n, gpuans, n, n);

	cudaMemcpy2D(ans, sizeof(int) * n, gpuans, sizeof(int) * n, sizeof(int) * n, n, cudaMemcpyDeviceToHost);

	cudaFree(gpuma);
	cudaFree(gpumb);
	cudaFree(gpuans);

	return ans;
}

int* strassen(int *ma, int *mb, int t)
{
	int *m, *a, *b, *ans;
	a = (int*)malloc(sizeof(int) * t / 2 * t / 2);
	b = (int*)malloc(sizeof(int) * t / 2 * t / 2);
	m = (int*)malloc(sizeof(int) * t / 2 * t / 2);
	ans = (int*)malloc(sizeof(int) * t * t);
	int i, j, k;
	for (i = 0; i < t; i++)
		for (j = 0; j < t; j++)
			ans[i*t + j] = 0;

	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			a[i*t / 2 + j] = ma[i*t + j];
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			b[i*t / 2 + j] = mb[i*t + j + t / 2] - mb[(i + t / 2)*t + j + t / 2];
	m = cudamatrixmul(a, b, t / 2);//1
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			ans[i*t + j + t / 2] += m[i*t / 2 + j];
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			ans[(i + t / 2)*t + j + t / 2] += m[i*t / 2 + j];

	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			a[i*t / 2 + j] = ma[i*t + j] + ma[i*t + j + t / 2];
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			b[i*t / 2 + j] = mb[(i + t / 2)*t + j + t / 2];
	m = cudamatrixmul(a, b, t / 2);//2
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			ans[i*t + j + t / 2] += m[i*t / 2 + j];
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			ans[i*t + j] -= m[i*t / 2 + j];

	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			a[i*t / 2 + j] = ma[(i + t / 2)*t + j] + ma[(i + t / 2)*t + j + t / 2];
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			b[i*t / 2 + j] = mb[i*t + j];
	m = cudamatrixmul(a, b, t / 2);//3
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			ans[(i + t / 2)*t + j] += m[i*t / 2 + j];
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			ans[(i + t / 2)*t + j + t / 2] -= m[i*t / 2 + j];

	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			a[i*t / 2 + j] = ma[(i + t / 2)*t + j + t / 2];
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			b[i*t / 2 + j] = mb[(i + t / 2)*t + j] - mb[i*t + j];
	m = cudamatrixmul(a, b, t / 2);//4
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			ans[i*t + j] += m[i*t / 2 + j];
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			ans[(i + t / 2)*t + j] += m[i*t / 2 + j];

	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			a[i*t / 2 + j] = ma[i*t + j] + ma[(i + t / 2)*t + j + t / 2];
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			b[i*t / 2 + j] = mb[i*t + j] + mb[(i + t / 2)*t + j + t / 2];
	m = cudamatrixmul(a, b, t / 2);//5
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			ans[i*t + j] += m[i*t / 2 + j];
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			ans[(i + t / 2)*t + j + t / 2] += m[i*t / 2 + j];

	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			a[i*t / 2 + j] = ma[i*t + j + t / 2] - ma[(i + t / 2)*t + j + t / 2];
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			b[i*t / 2 + j] = mb[(i + t / 2)*t + j] + mb[(i + t / 2)*t + j + t / 2];
	m = cudamatrixmul(a, b, t / 2);//6
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			ans[i*t + j] += m[i*t / 2 + j];

	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			a[i*t / 2 + j] = ma[(i + t / 2)*t + j] - ma[i*t + j];
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			b[i*t / 2 + j] = mb[i*t + j] + mb[i*t + j + t / 2];
	m = cudamatrixmul(a, b, t / 2);//7
	for (i = 0; i < t / 2; i++)
		for (j = 0; j < t / 2; j++)
			ans[(i + t / 2)*t + j + t / 2] += m[i*t / 2 + j];

	free(m);
	free(a);
	free(b);
	return ans;
}

int main()
{
	if (!InitCUDA()) {
		return 0;
	}

	int *ma, *mb, *ans;
	int n = 1035, t;
	clock_t start, end;

	while (n < 1041) {
		start = clock();
		if (n % 2 == 1)
			t = n + 1;
		else
			t = n;

		ma = (int*)malloc(sizeof(int) * t * t);
		mb = (int*)malloc(sizeof(int) * t * t);
		ans = (int*)malloc(sizeof(int) * t * t);

		srand(0);

		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
			{
				ma[i * t + j] = rand() % 10;
				mb[i * t + j] = rand() % 10;
			}
		for (int i = n; i < t; i++)
			for (int j = 0; j < n; j++)
			{
				ma[i*t + j] = 0;
				mb[i*t + j] = 0;
			}
		for (int i = 0; i < t; i++)
			for (int j = n; j < t; j++)
			{
				ma[i*t + j] = 0;
				mb[i*t + j] = 0;
			}

		ans = strassen(ma, mb, t);

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