//01. SUBARRAY SUM EQUALS K

//brute force find all subarray then count valid     {T.C = O(N^3), S.C = O(1)}
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int n = nums.size();
        int count = 0;
        for(int i = 0 ; i < n; i++){
            for(int j = i; j < n; j++){                       //starts from i (not 0)
                int sum = 0;
                for(int l = i ; l <= j; l++){                 //each subarray (i->j)
                    sum += nums[l];
                }
                if(sum == k) count++;
            }
        }
        return count;
    }
};

//slight better approach (remove inner loop , running sum)  {T.C = O(N^2), S.C = O(1)}
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int n = nums.size();
        int count = 0;
        for(int i = 0 ; i < n; i++){
            int sum = 0;
            for(int j = i; j < n; j++){
                sum += nums[j];
                if(sum == k) count++;
            }
        }
        return count;
    }
};

//optimized approach  (using prefix Sum + map)               {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int n = nums.size();
        unordered_map<int,int>mp;
        int count = 0, cumlSum = 0;                  //cumilative sum (prefix array)
        mp.insert({0, 1});                           //cumlSum, occurance  (add 0 sum for edge case)

        for(int i = 0 ; i < n; i++){
            cumlSum += nums[i];
            if(mp.count(cumlSum-k)){
                count += mp[cumlSum - k];                  //add freq (not element)
            }

            mp[cumlSum]++;
        }
        return count;
    }
};
/*
Example 1:
Input: nums = [1,1,1], k = 2
Output: 2

Example 2:
Input: nums = [1,2,3], k = 3
Output: 2
*/


//02. BINARY SUBARRAYS WITH SUM                             {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int numSubarraysWithSum(vector<int>& nums, int goal) {
        int n = nums.size();
        unordered_map<int, int>mp;
        mp.insert({0, 1});                            //cumlSum, freq

        int cumlSum = 0, count = 0;
        for(int i = 0 ; i < n; i++){
            cumlSum  += nums[i];
            if(mp.count(cumlSum - goal)){
                count += mp[cumlSum-goal];           //add freq not ele
            }

            mp[cumlSum]++;
        }
        return count;
    }
};
/*
Example 1:
Input: nums = [1,0,1,0,1], goal = 2
Output: 4
Explanation: The 4 subarrays are bolded and underlined below:
[1,0,1,0,1]
[1,0,1,0,1]
[1,0,1,0,1]
[1,0,1,0,1]

Example 2:
Input: nums = [0,0,0,0,0], goal = 0
Output: 15
*/


//03. NUMBER OF SUBMATRICES THAT SUM TO TARGET
//brute force  (using 6 loops)                                   {T.C = O(N^3M^3), S.C = O(1)}
class Solution {
public:
    int numSubmatrixSumTarget(vector<vector<int>>& matrix, int target) {
        int n = matrix.size(), m = matrix[0].size();
        int count = 0;
        for(int sr = 0 ; sr < n; sr++){                      //sr = startingRow
            for(int sc = 0; sc < m ; sc++){                  //sc = startringCol

                for(int er = sr ; er < n; er++){             //er = endingRow
                    for(int ec = sc ; ec < m; ec++){         //ec = ending Col

                        int sum = 0;
                        for(int i = sr ; i <= er ; i++){
                            for(int j = sc ; j <= ec ; j++){
                                sum += matrix[i][j];
                            }
                        }
                        if(sum == target) count++;
                    }
                }
            }
        }
        return count;
    }
};


//optimal approach using 1st question (prefixSum + map)                   {T.C = O(N^2*M), S.C = O(N*M)}
class Solution {
public:
    int numSubmatrixSumTarget(vector<vector<int>>& matrix, int target) {
        int n = matrix.size(), m = matrix[0].size();

        //update matrix with take row wise cumulative sum
        for(int i = 0 ; i < n; i++){
            for(int j = 1; j < m; j++){                  //j= 0 (no need)
                matrix[i][j] += matrix[i][j-1];
            }
        }

        //find number of subarray with sum == target (leetcode 560) downwards directions
        int count = 0;
        for(int sc = 0; sc < m; sc++){              //sc = start col
            for(int j = sc; j < m; j++){

                unordered_map<int,int>mp;
                mp.insert({0, 1});                   //cumlSum, freq
                int cumlSum = 0;
                for(int i = 0 ;i < n; i++){
                    cumlSum += matrix[i][j] - (sc > 0 ? matrix[i][sc-1] : 0);    //if start col move to next col (reduce its previous subarrays) 
                    if(mp.count(cumlSum-target)){
                        count += mp[cumlSum-target];
                    }
                    mp[cumlSum]++;
                }
            }
        }
        return count;
    }
};
/*
Input: matrix = [[0,1,0],[1,1,1],[0,1,0]], target = 0
Output: 4
Explanation: The four 1x1 submatrices that only contain 0.

Example 2:
Input: matrix = [[1,-1],[-1,1]], target = 0
Output: 5
Explanation: The two 1x2 submatrices, plus the two 2x1 submatrices, plus the 2x2 submatrix.

Example 3:
Input: matrix = [[904]], target = 0
Output: 0
*/