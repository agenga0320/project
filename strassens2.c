#include <stdio.h>
#include <stdlib.h>
#include<time.h>

int *matrixmul(int *ma ,int *mb ,int t)
{
    int *ans ,i ,j ,k ;
    ans = (int*)malloc(sizeof(int)*t*t) ;
    for(i=0;i<t;i++)
        for(j=0;j<t;j++)
        {
            int temp=0 ;
            for(k=0;k<t;k++)
                temp += ma[i*t+k] * mb[k*t+j] ;
            ans[i*t+j] = temp ;
        }
    return ans ;
}

int* strassen(int *ma, int *mb, int t)
{
    int *m ,*a ,*b ,*ans ;
    a = (int*)malloc(sizeof(int) * t / 2 * t / 2);
    b = (int*)malloc(sizeof(int) * t / 2 * t / 2);
    m = (int*)malloc(sizeof(int) * t / 2 * t / 2);
    ans = (int*)malloc(sizeof(int) * t * t );
    int i ,j ,k;
    for(i=0;i<t;i++)
        for(j=0;j<t;j++)
            ans[i*t+j] = 0 ;

    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            a[i*t / 2 + j] = ma[i*t + j];
    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            b[i*t / 2 + j] = mb[i*t + j + t / 2] - mb[(i + t / 2)*t + j + t / 2];
    m = matrixmul(a, b, t/2);//1
    for(i=0;i<t/2;i++)
        for(j=0;j<t/2;j++)
            ans[i*t+j+t/2] += m[i*t/2+j] ;
    for(i=0;i<t/2;i++)
        for(j=0;j<t/2;j++)
            ans[(i+t/2)*t+j+t/2] += m[i*t/2+j] ;

    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            a[i*t / 2 + j] = ma[i*t + j] + ma[i*t + j + t / 2];
    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            b[i*t / 2 + j] = mb[(i + t / 2)*t + j + t / 2];
    m = matrixmul(a, b, t/2);//2
    for(i=0;i<t/2;i++)
        for(j=0;j<t/2;j++)
            ans[i*t+j+t/2] += m[i*t/2+j] ;
    for(i=0;i<t/2;i++)
        for(j=0;j<t/2;j++)
            ans[i*t+j] -= m[i*t/2+j] ;

    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            a[i*t / 2 + j] = ma[(i + t / 2)*t + j] + ma[(i + t / 2)*t + j + t / 2];
    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            b[i*t / 2 + j] = mb[i*t + j];
    m = matrixmul(a, b, t/2);//3
    for(i=0;i<t/2;i++)
        for(j=0;j<t/2;j++)
            ans[(i+t/2)*t+j] += m[i*t/2+j] ;
    for(i=0;i<t/2;i++)
        for(j=0;j<t/2;j++)
            ans[(i+t/2)*t+j+t/2] -= m[i*t/2+j] ;

    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            a[i*t / 2 + j] = ma[(i + t / 2)*t + j + t / 2];
    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            b[i*t / 2 + j] = mb[(i + t / 2)*t + j] - mb[i*t + j];
    m = matrixmul(a, b, t/2);//4
    for(i=0;i<t/2;i++)
        for(j=0;j<t/2;j++)
            ans[i*t+j] += m[i*t/2+j] ;
    for(i=0;i<t/2;i++)
        for(j=0;j<t/2;j++)
            ans[(i+t/2)*t+j] += m[i*t/2+j] ;

    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            a[i*t / 2 + j] = ma[i*t + j] + ma[(i + t / 2)*t + j + t / 2];
    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            b[i*t / 2 + j] = mb[i*t + j] + mb[(i + t / 2)*t + j + t / 2];
    m = matrixmul(a, b, t/2);//5
    for(i=0;i<t/2;i++)
        for(j=0;j<t/2;j++)
            ans[i*t+j] += m[i*t/2+j] ;
    for(i=0;i<t/2;i++)
        for(j=0;j<t/2;j++)
            ans[(i+t/2)*t+j+t/2] += m[i*t/2+j] ;

    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            a[i*t / 2 + j] = ma[i*t + j + t / 2] - ma[(i + t / 2)*t + j + t / 2];
    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            b[i*t / 2 + j] = mb[(i + t / 2)*t + j] + mb[(i + t / 2)*t + j + t / 2];
    m = matrixmul(a, b, t/2);//6
    for(i=0;i<t/2;i++)
        for(j=0;j<t/2;j++)
            ans[i*t+j] += m[i*t/2+j] ;

    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            a[i*t / 2 + j] = ma[(i + t / 2)*t + j] - ma[i*t + j];
    for (i = 0; i < t / 2; i++)
        for (j = 0; j < t / 2; j++)
            b[i*t / 2 + j] = mb[i*t + j] + mb[i*t + j + t / 2];
    m = matrixmul(a, b, t/2);//7
    for(i=0;i<t/2;i++)
        for(j=0;j<t/2;j++)
            ans[(i+t/2)*t+j+t/2] += m[i*t/2+j] ;

    free(m) ;
    free(a) ;
    free(b) ;
    return ans;
}

int main()
{
	int *ma, *mb, *ans, i, j;
	int n=902 ,t;
    clock_t start ,end ;

	while(n<1040)
    {
        start = clock() ;
        if(n%2==0)
            t = n ;
        else
            t = n+1 ;
        ma = (int*)malloc(sizeof(int) * t * t);
        mb = (int*)malloc(sizeof(int) * t * t);
        ans = (int*)malloc(sizeof(int) * t * t);

        srand(0);

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                ma[i * t + j] = rand() % 10;

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                mb[i * t + j] = rand() % 10;

        for(i = n;i < t;i++)
            for(j=0;j<n;j++)
                ma[i*t+j] = mb[i*t+j] = 0 ;

        for(i = 0;i < t;i++)
            for(j=n;j<t;j++)
                ma[i*t+j] = mb[i*t+j] = 0 ;

        ans = strassen(ma ,mb ,t) ;

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
