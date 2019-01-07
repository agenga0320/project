#include <stdio.h>
#include <stdlib.h>
#include<time.h>
int* matrixmul(int *ma, int *mb, int n)
{
	int *ans ,i ,j ,k;
	ans = (int*)malloc(sizeof(int) * n * n);
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++) {
			int t = 0;
			for (k = 0; k < n; k++)
				t += ma[i * n + k] * mb[k * n + j];
			ans[i*n + j] = t;
		}
	return ans;
}

int main()
{
	int *ma, *mb, *ans, i, j ,n=900;
    clock_t start ,end ;
	while(n<1040)
    {
        start = clock() ;

        ma = (int*)malloc(sizeof(int) * n * n);
        mb = (int*)malloc(sizeof(int) * n * n);
        ans = (int*)malloc(sizeof(int) * n * n);

        srand(0);

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                ma[i * n + j] = rand() % 10;

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                mb[i * n + j] = rand() % 10;

        ans = matrixmul(ma ,mb ,n ) ;

        free(ma) ;
        free(mb) ;
        free(ans) ;
        end = clock() ;
        int s = (int)start ;
        int e = (int)end ;
        printf("%d\n",e-s) ;
        n++ ;
    }
    return 0 ;
}
