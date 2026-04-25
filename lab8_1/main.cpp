#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

extern "C" void vec_add_cuda(float* a, float* b, float* c, int n);

const int N = 1024;

float a[N], b[N], c_gpu[N], c_cpu[N];

void vec_add_cpu(float* a, float* b, float* c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    srand((unsigned int)time(nullptr));

    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
        c_gpu[i] = 0;
        c_cpu[i] = 0;
    }

    vec_add_cuda(a, b, c_gpu, N);
    vec_add_cpu(a, b, c_cpu, N);

    bool correct = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(c_gpu[i] - c_cpu[i]) > 0.0001)
        {
            correct = false;
            break;
        }
    }

    cout << "First 20 results:" << endl;

    for (int i = 0; i < 20; i++)
    {
        cout << a[i] << " + " << b[i] << " = " << c_gpu[i] << endl;
    }

    cout << endl;

    if (correct)
    {
        cout << "GPU and CPU results are equal. Calculations are correct." << endl;
    }
    else
    {
        cout << "Error: GPU and CPU results are different." << endl;
    }

    system("pause");
    return 0;
}
