//CONSTRAINTS TRICKS

#include<bits/stdc++.h>
using namespace std;

n = size of array;

1 <= n <= 10^8

1 <= arr[i] <= 10^9          (int is enough),    9 > (10, 11, 12,...) long required


/*
CONSTRAINTS (N MAX VALUE)                TIME COMPLEXITY

        10^18                                O(LOGN)
        10^8                                 O(N)
        10^4                                 O(N^2)
        10^6                                 O(N*LOGN)
        500                                  O(N^3)
        85-90                                O(N^4)
        20                                   O(2^N)
        11                                   O(N!)
*/


//TIME COMPLEXITY OF RECURSIVE TREE

T.C = NO. OF NODES * WORK DONE BY EACH NODE 

//FOR PERMUTATION OF STRING
T.C = N! * N^2