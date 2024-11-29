extern "C"
{
    void matmul_kernel(const float *A, const float *B, float *C, int N)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < N; k++)
                {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    const char *get_matmul_signature()
    {
        static const char *sig = "float*,float*,float*,int";
        return sig;
    }
}