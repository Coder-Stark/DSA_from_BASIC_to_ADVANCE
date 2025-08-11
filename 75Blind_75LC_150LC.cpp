//75 BLIND SHEET
#include<bits/stdc++.h>
using namespace std;

/********************************************* REPEATED 3 TIMES ********************************************** */

//01. TWO SUM                 (CUTTED)
//BRUTE FORCE APPROACH                                                          {T.C = O(N^2), S.C = O(1)}
//using 2 for loop iterate vector and return index
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        for(int i = 0 ; i < n ; i++){
            for(int j = i+1; j < n ; j++){
                if(nums[i]+nums[j] == target){
                    return {i, j};
                }
            }
        }
        return {};
    }
};

//OPTIMIZED APPROACH                                                            {T.C = O(N), S.C = O(N)}
//using unordered map first store vector in map then find its complement is present or not if present just return 
//mp index of complement and original i index
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        unordered_map<int, int>mp;

        //find the complement
        for(int i = 0 ; i < n  ; i++){
            int complement = target - nums[i];
            if(mp.find(complement) != mp.end()){
                return {mp[complement], i};
            }
            mp[nums[i]] = i;                         //mapping complement->i
        }
        return {};
    }
};
/*
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
*/


//02. BEST TIME TO BUY STOCK  (ALL)
//2.1. BUY ON ONE DAY SELL ON ANOTHER DAY                                       {T.C = O(N), S.C = O(1)}
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int mini = prices[0];
        int maxPft = 0;

        for(int i = 1 ; i < prices.size() ; i++){
            int profit = prices[i] - mini;
            maxPft = max(maxPft, profit);
            mini = min(mini, prices[i]);                //for updating mini element
        }
        return maxPft;
    }
};
/*
Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

Example 2:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.
*/


//02.2 BUY/SELL ON SAME DAY WITH HOLIDING ATMOST 1 STOCK AT A TIME             {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int solveMem(vector<int>&prices, int idx, int buy, vector<vector<int>>&dp){
        //base case
        if(idx == prices.size()){
            return 0;
        }
        
        //step3 if answer exist return it
        if(dp[idx][buy] != -1){
            return dp[idx][buy];
        }

        int profit = 0;
        if(buy){          //buy == 1  allowed
            int buyKaro = -prices[idx] + solveMem(prices, idx+1, 0, dp);        //0 for next buy / buy not allowed 
            int skipKaro = 0 + solveMem(prices, idx+1, 1, dp);
            profit = max(buyKaro, skipKaro);
        }else{            //buy == 0   not allowed
            int sellKaro = +prices[idx] + solveMem(prices, idx+1, 1, dp);       //1 for next buy / buy allowed
            int skipKaro = 0 + solveMem(prices, idx+1, 0, dp);
            profit = max(sellKaro, skipKaro);
        }

        //step2 store ans in dp
        dp[idx][buy] = profit;
        return dp[idx][buy];
    }
    int maxProfit(vector<int>& prices) {
        int n =  prices.size();
        //step1 create a dp vector
        vector<vector<int>>dp(n, vector<int>(2, -1));          //col = 0 or 1
        return solveMem(prices, 0, 1, dp);                     //0 = index, 1 = buy(allowed{bool value})
    }
};


//02.3 BUY/SELL ON SAME DAY WITH HOLIDING ATMOST 2 STOCK AT A TIME             {T.C = O(N), S.C = O(N)}
//RECURSION + MEMOIZATION
class Solution {
public:
    int solveMem(vector<int>&prices, int index, int buy, int limit ,vector<vector<vector<int>>>&dp){
        //base case
        int n = prices.size();
        if(index == n || limit == 0){
            return 0;
        }

        //step-3 if ans already present print it
        if(dp[index][buy][limit] != -1){
            return dp[index][buy][limit];
        }

        int profit = 0;
        if(buy){
            int buyKaro = -prices[index] + solveMem(prices, index+1, 0, limit, dp);
            int skipKaro = 0 + solveMem(prices, index+1, 1, limit, dp);
            profit = max(buyKaro, skipKaro);
        }
        else{
            int sellKaro = +prices[index] + solveMem(prices, index+1, 1, limit-1, dp);
            int skipKaro = 0 + solveMem(prices, index+1, 0, limit, dp);
            profit = max(sellKaro , skipKaro);
        }

        //step-2 store ans in dp
        dp[index][buy][limit] = profit;
        return dp[index][buy][limit];
    }
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        //step-1 create dp vector
        vector<vector<vector<int>>>dp(n , vector<vector<int>>(2, vector<int>(3, -1)));      //row = n , col = 2  , entry (limit) = 3 (at most 2 = 0 , 1 , 2)
        return solveMem(prices, 0, 1, 2, dp);                  //index = 0 , buy = 1, limit = 2
    }
};
/*
Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.

Example 2:
Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Total profit is 4.
*/


//02.4. BUY/SELL ON SAME DAY WITH HOLIDING ATMOST K(TRANSACTION) STOCK AT A TIME             {T.C = O(N*K), S.C = O(N*K)}
//RECURSION + MEMOIZATION
class Solution {
public:
    int solveMem(vector<int>&prices, int index, int operationNo, int k , vector<vector<int>>&dp){
        int n = prices.size();
        if(index == n || operationNo == 2*k){
            return 0;
        }
        
        //step-3 if ans already present return it
        if(dp[index][operationNo] != -1){
            return dp[index][operationNo];
        }
        int profit = 0;
        if(operationNo % 2 == 0){                            //even = buy allow
            int buyKaro = -prices[index] + solveMem(prices, index+1, operationNo+1, k, dp);
            int skipKaro = 0 + solveMem(prices, index+1, operationNo, k, dp);
            profit = max(buyKaro, skipKaro);
        }
        else{                                                //odd = sell allow
            int sellKaro = +prices[index] + solveMem(prices, index+1, operationNo+1, k, dp);
            int skipKaro = 0 + solveMem(prices, index+1, operationNo , k, dp);
            profit = max(sellKaro, skipKaro);
        }

        //step-2 store ans in dp
        dp[index][operationNo] = profit;
        return dp[index][operationNo];
    }
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        //step-1 create dp vector
        vector<vector<int>>dp(n, vector<int>(2*k, -1));             //col = operation = 2*k
        return solveMem(prices, 0, 0, k, dp);                         //index = 0, operationNo = 0
    }
};
/*
Example 1:

Input: k = 2, prices = [2,4,1]
Output: 2
Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 4-2 = 2.
Example 2:

Input: k = 2, prices = [3,2,6,5,0,3]
Output: 7
Explanation: Buy on day 2 (price = 2) and sell on day 3 (price = 6), profit = 6-2 = 4. Then buy on day 5 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
*/


//03. PRODUCT OF ARRAY EXCEPT SELF            
//BRUTE FORCE APPROACH (NOT WORKING)
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int>ans(n, 1);
        int totalProduct = 1;
        for(int i = 0 ; i < n; i++){
            totalProduct *= nums[i];
        }

        for(int i = 0 ; i < n ; i++){
            ans[i] = totalProduct/nums[i];
        }
        return ans;
    }
};
//OPTIMIZED APPROACH                                                               {T.C = O(N), S.C = O(1)/EXCEPT ANS}
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int>ans(n, 1);
        int leftProd = 1;                                //prefix
        for(int i = 0 ; i < n ; i++){
            ans[i] *= leftProd;
            leftProd *= nums[i];
            
        }
        int rightProd = 1;                              //suffix
        for(int i = n-1 ; i >= 0 ; i--){
            ans[i] *= rightProd;
            rightProd *= nums[i];
        }
        return ans;
    }
};
/*
Input: nums = [1,2,3,4]
Output: [24,12,8,6]
*/


//04. CONTAINER WITH MOST WATER                                               {T.C = O(N), S.C = O(1)}
class Solution {
public:
    int maxArea(vector<int>& height) {
        int n = height.size();
        int maxA = 0;
        int i = 0, j = n-1;
        while(i <= j){
            int h = min(height[i], height[j]);         //min height(hold water else flow)
            int w = j-i;
            int a = h*w;
            maxA  = max(maxA, a);
            if(height[i] <= height[j]) i++;
            else j--;
        }
        return maxA;
    }
};
/*
Example 1:
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.

Example 2:
Input: height = [1,1]
Output: 1
*/


//05. HOUSE ROBBER I                                               {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int dp[105];
    int solveMem(vector<int>&nums , int i){
        int n = nums.size();
        //base case
        if(i >= n) return 0;

        if(dp[i] != -1 ) return dp[i];
        int incl = nums[i] + solveMem(nums, i+2);       //adjacent skip
        int excl = 0 + solveMem(nums, i+1);

        return dp[i] = max(incl, excl);
    }
    int rob(vector<int>& nums) {
        memset(dp, -1, sizeof(dp));
        return solveMem(nums, 0);             //0 = initial index
    }
};
/*
Example 1:
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

Example 2:
Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.
*/


//06. REVERSE LINKED LIST                                        {T.C = O(N), S.C = O(1)}
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = NULL;
        ListNode* curr = head;
        ListNode* forw = NULL;

        while(curr){
            forw = curr->next;
            curr->next = prev;
            prev = curr;
            curr = forw;
        }
        return prev;
    }
};
/*
Example 1:
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Example 2:
Input: head = [1,2]
Output: [2,1]

Example 3:
Input: head = []
Output: []
*/


//07. WORD SEARCH I                                                    {T.C = O(4^N*N*M), S.C = O(N*M)}
class Solution {
public:
    bool isValid(int i, int j, int n, int m){
        return (i >= 0 && i < n && j >= 0 && j < m);
    }
    bool dfs(vector<vector<char>>&board, string &word, vector<vector<bool>>&vis, int i, int j, int idx){
        int n = board.size(), m = board[0].size();
        if(idx == word.length()) return true;                              //word found

        if(!isValid(i, j, n, m) || vis[i][j] || board[i][j] != word[idx]) return false;

        vis[i][j] = true;
        //Check in all 4 directions               
        bool found = dfs(board, word, vis, i + 1, j, idx+1) ||
                     dfs(board, word, vis, i - 1, j, idx+1) ||
                     dfs(board, word, vis, i, j + 1, idx+1) ||
                     dfs(board, word, vis, i, j - 1, idx+1);
        /*for this pruning is required
        for (auto it : directions) {
            int newI = i + it[0];
            int newJ = j + it[1];

            if (dfs(board, word, vis, newI, newJ, idx+1)) return true; 
        }
        */

        // Mark the current cell as unvisited after exploration
        vis[i][j] = false;                   //backtrack

        return found;
    }
    bool exist(vector<vector<char>>& board, string word) {
        int n = board.size(), m = board[0].size();
        vector<vector<bool>>vis(n, vector<bool>(m, false));
        for(int i = 0 ; i < n; i++){
            for(int j = 0; j < m; j++){
                if(dfs(board, word, vis, i, j, 0)) return true;
            }
        }
        return false;
    }
};
/*
Example 1:
Input: board = [["A","B","C","E"],
                ["S","F","C","S"],
                ["A","D","E","E"]], 
                word = "ABCCED"
Output: true

Example 2:
Input: board = [["A","B","C","E"],
                ["S","F","C","S"],
                ["A","D","E","E"]], 
                word = "SEE"
Output: true

Example 3:
Input: board = [["A","B","C","E"],
                ["S","F","C","S"],
                ["A","D","E","E"]], 
                word = "ABCB"
Output: false
*/


//08. MAXIMUM DEPTH / HEIGHT OF BINARY TREE                      {T.C = O(N), S.C = O(H)}
class Solution {
public:
    int maxDepth(TreeNode* root) {
        //base case
        if(!root) return 0;
        
        int left = maxDepth(root->left);
        int right = maxDepth(root->right);

        return max(left, right)+1;               //1 = for current level
    }
};
/*
Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: 3

Example 2:
Input: root = [1,null,2]
Output: 2
*/



//09. LOWEST COMMON ANCESTOR OF BST                              
//WITHOUT USING PROPERTIES OF BST (SIMPLE BT ALSO WORKS)       {T.C = O(N), S.C = O(H)}
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        //base case
        if(!root) return NULL;

        if(root->val == p->val || root->val == q->val) return root;

        TreeNode* leftSubTree = lowestCommonAncestor(root->left, p, q);
        TreeNode* rightSubTree= lowestCommonAncestor(root->right,p, q);

        if(leftSubTree && rightSubTree) return root;
        else if(!leftSubTree && rightSubTree) return rightSubTree;
        else if(leftSubTree && !rightSubTree) return leftSubTree;
        else return NULL;                             //!left && !right
    }
};
//USING BST PROPERTIES                                          {T.C = O(N), S.C = O(H)}
class Solution {
public:
    //using bst property 
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        //base case
        if(!root) return NULL;

        if(root->val > p->val && root->val > q->val){
            return lowestCommonAncestor(root->left, p, q);         //DIRECT RETURN
        } 
        else if(root->val < p->val && root->val < q->val){
            return lowestCommonAncestor(root->right,p, q);
        }
        return root;
    }
};
/*
Example 1:
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.

Example 2:
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.

Example 3:
Input: root = [2,1], p = 2, q = 1
Output: 2
*/



/********************************************** REPEATED 2 TIMES *********************************************** */
//10. CONTAINS DUPLICATES                                         {T.C = O(N), S.C = O(N)}
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        unordered_map<int,int>mp;
        for(auto it : nums) mp[it]++;

        if(mp.size() < nums.size()) return true;

        return false;
    }
};
/*
Example 1:
Input: nums = [1,2,3,1]
Output: true
Explanation:
The element 1 occurs at the indices 0 and 3.

Example 2:
Input: nums = [1,2,3,4]
Output: false
Explanation:
All elements are distinct.
*/


//11. MAXIMUM SUM SUBARRAY (KADANE'S ALGO)      (CUTTED)
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        int sum = 0;
        int maxi = INT_MIN;               
        for(int i = 0 ; i < n ; i++){
            sum += nums[i];
            maxi = max(maxi, sum);
            if(sum < 0){                        //necessary condition for kadane's algo
                sum = 0;
            }
        }
        return maxi;
    }
};
/*
Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.
Example 2:

Input: nums = [1]
Output: 1
Explanation: The subarray [1] has the largest sum 1.

Example 3:
Input: nums = [5,4,-1,7,8]
Output: 23
Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
*/


//12. MAXIMUM PRODUCT SUBARRAY (KADANE'S ALGO)                                          {T.C = O(N), S.C = O(1)}
//USING KADANE'S ALGO(LEFT TO RIGTH, RIGTH TO LEFT BOTH)
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int n = nums.size();
        int prod = 1;
        int maxProd = INT_MIN;
        for(int i = 0 ; i < n ; i++){             //left to right
            prod *= nums[i];
            maxProd = max(maxProd, prod);
            if(prod == 0) prod = 1;
        }
        prod = 1;                                 //reset
        for(int i = n-1 ; i >= 0 ; i--){
            prod *= nums[i];
            maxProd = max(maxProd, prod);
            if(prod == 0) prod = 1;
        }

        return maxProd;
    }
};
/*
Example 1:
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.

Example 2:
Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
*/


//13. FIND MINIMUM IN ROTATED SORTED ARRAY        (CUTTED)
class Solution {
public:
    int findMin(vector<int>& nums) {
        int n = nums.size();
        int start = 0;
        int end = n - 1;

        while (start < end) {
            // If the array is not rotated (sorted in ascending order), return the first element
            if (nums[start] < nums[end]) {
                return nums[start];
            }

            int mid = start + (end - start) / 2;

            if (nums[start] > nums[mid]) {   // Minimum element lies in the left half
                end = mid;
            } else {                         // Minimum element lies in the right half
                start = mid + 1;
            }
        }

        return nums[start];                  // the minimum element is at nums[start]
    }
};
/*
Example 1:
Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.

Example 2:
Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.

Example 3:
Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
*/


//14. SEARCH IN ROTATED SORTED ARRAY                              {T.C = O(LOGN), S.C = O(1)}
//BINARY SEARCH ON INDEX
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int n = nums.size();
        int start = 0, end = n-1;
        while(start <= end){
            int midIdx = (start + end)/2;
            if(nums[midIdx] == target) return midIdx;
            else if(nums[start] <= nums[midIdx]){                  //left part is sorted
                if(nums[start] <= target && target <= nums[midIdx]) end = midIdx-1;  //ans in left 
                else start = midIdx+1;
            }else{ //nums[start] > nums[midIdx]                   //right part is sorted
                if(nums[midIdx] <= target && target <= nums[end]) start = midIdx+1;  //ans in right
                else end = midIdx-1;
            }
        }
        return -1;
    }
};
/*
Example 1:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Example 3:
Input: nums = [1], target = 0
Output: -1
*/


//15. 3 SUM                                                         {T.C = O(N^2), S.C = O(N)}
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        set<vector<int>>st;                              //unique ans required
        for(int i = 0 ; i < n-2 ; i++){
            int start = i+1;
            int end = n-1;
            while(start < end){
                int sum = nums[i] + nums[start] + nums[end];
                if(sum == 0){
                    st.insert({nums[i], nums[start], nums[end]}); 
                    start++, end--;
                } 
                else if(sum < 0) start++;
                else end--;
            }
        }
        vector<vector<int>>ans;
        for(auto it : st){
            ans.push_back(it);
        }
        return ans;
    }
};
/*
Example 1:
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation: 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
Notice that the order of the output and the order of the triplets does not matter.

Example 2:
Input: nums = [0,1,1]
Output: []
Explanation: The only possible triplet does not sum up to 0.

Example 3:
Input: nums = [0,0,0]
Output: [[0,0,0]]
Explanation: The only possible triplet sums up to 0.
*/


//16. SUM OF 2 INTEGERS                                            {T.C = O(1), S.C = O(1)}
class Solution {
public:
    int getSum(int a, int b) {
        while (b != 0) {
            int Xor = a ^ b;                // sum of bits where at least one is 1
            int carry = (a & b) << 1;       // carry where both bits are 1
            a = Xor;                        // update a to be the new sum
            b = carry;                      // update b to be the new carry
        }
        return a;                           // return sum when carry is 0
    }
};
/*
Example 1:
Input: a = 1, b = 2
Output: 3

Example 2:
Input: a = 2, b = 3
Output: 5
*/


//17. NUMBER OF 1 BITS / HAMMING WEIGHT                       
//USING STL                                                        {T.C = O(LOGN), S.C = O(1)}
class Solution {
public:
    int hammingWeight(int n) {
        return __builtin_popcount(n);
    }
};
//WITHOUT STL                                                     {T.C = O(LOGN), S.C = O(1)}
class Solution {
public:
    int hammingWeight(int n) {
        int countSetBits = 0;
        while(n){
            countSetBits += n % 2;
            n /= 2;
        }
        return countSetBits;
    }
};
/*
Example 1:
Input: n = 11
Output: 3
Explanation:
The input binary string 1011 has a total of three set bits.

Example 2:
Input: n = 128
Output: 1
Explanation:
The input binary string 10000000 has a total of one set bit.

Example 3:
Input: n = 2147483645
Output: 30
Explanation:
The input binary string 1111111111111111111111111111101 has a total of thirty set bits.
*/


//18. COUNTING BITS
//BRUTE FORCE                                                      {T.C = O(N*LOGN), S.C = O(N)}
class Solution {
public:
    int countSetBits(int n){
        int count = 0;
        while(n){
            count += n % 2;
            n /= 2;
        }
        return count;
    }
    vector<int> countBits(int n) {
        vector<int>ans(n+1);
        for(int i = 0; i <= n; i++){
            ans[i] = countSetBits(i);
        }
        return ans;
    }
};
//OPTIMAL APPROACH                                                  {T.C = O(N), S.C = O(N)}
//COUNT OF BITS OF NUM IS SAME AS ITS HALF(IF ODD THEN +1)
class Solution {
public:
    vector<int> countBits(int n) {
        vector<int>ans(n+1);
        ans[0] = 0;                             
        for(int i = 1 ; i <= n ;i++){
            if(i % 2 == 0) ans[i] = ans[i/2];
            else ans[i] = ans[i/2]+1;
        }   
        return ans;
    }
};
/*
Example 1:
Input: n = 2
Output: [0,1,1]
Explanation:
0 --> 0
1 --> 1
2 --> 10

Example 2:
Input: n = 5
Output: [0,1,1,2,1,2]
Explanation:
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101
*/


//19. MISSING NUMBER 
//USING MATHS FORMULA                                             {T.C = O(N), S.C = O(1)}
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        int sum = (n*(n+1))/2;                        //0 to n numbers

        for(int i = 0 ; i < n;  i++){
            sum -= nums[i];
        }
        return sum;
    }
};
/*
Example 1:
Input: nums = [3,0,1]
Output: 2
Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the 
range since it does not appear in nums.

Example 2:
Input: nums = [0,1]
Output: 2
Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 2 is the missing number in the 
range since it does not appear in nums.

Example 3:
Input: nums = [9,6,4,2,3,5,7,0,1]
Output: 8
Explanation: n = 9 since there are 9 numbers, so all numbers are in the range [0,9]. 8 is the missing number in the 
range since it does not appear in nums.
*/


//20. REVERSE BITS                                                  {T.C = O(1), S.C = O(1)}
class Solution {
public:
    bool findKthBitSet(int k, uint32_t n){
        if( ((1 << (k-1)) & n) != 0 ){      //first reach to set bit( which is 1 by shifting k-1) then take & with n
            return true;
        }
        return false;                       //(1.0 = 0)
    }
    uint32_t reverseBits(uint32_t n) {
        uint32_t ans = 0;
        for(int i = 1; i <= 32 ; i++){      //from 1 otherwise overflow
            if(findKthBitSet(i, n)){        //find kth bit is set(1) or not set(0)
                ans = 1 << (32-i) | ans;    //second last to second position or vice versa  by shifting element and take OR (ans keep building)
            }
        }
        return ans;
    }
};
/*
Example 1:
Input: n = 00000010100101000001111010011100
Output:    964176192 (00111001011110000010100101000000)
Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.

Example 2:
Input: n = 11111111111111111111111111111101
Output:   3221225471 (10111111111111111111111111111111)
Explanation: The input binary string 11111111111111111111111111111101 represents the unsigned integer 4294967293, so return 3221225471 which its binary representation is 10111111111111111111111111111111.
*/


//21. CLIMBING STAIRS                                              {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int dp[50];
    int solveMem(int n){
        //base case
        if(n <= 1) return 1;                    //at 0 level 1 step count

        if(dp[n] != -1) return dp[n];

        int way1 = solveMem(n-1);
        int way2 = solveMem(n-2);

        return dp[n] = way1 + way2;
    }
    int climbStairs(int n) {
        memset(dp, -1, sizeof(dp));
        return solveMem(n);
    }
};
/*
Example 1:
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

Example 2:
Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
*/


//22. COIN CHANGE                                                      {T.C = O(N*TARGET), S.C = O(N*TARGET)}            
/*
{T.C = O(EXP), S.C = O(1)}
Brute force = using simple recursion handle base case , initialize ans with INT_MAX(finding minimum ans) iterate through coins , if 
amount-coins[i] >= 0 make recursive call and return min ans.
*/
class Solution {
public:
    int solve(vector<int>&coins, int amount){
        //base case
        if(amount == 0){
            return 0;
        }

        int ans = INT_MAX;
        for(int i = 0 ; i < coins.size() ; i++){
            if(amount-coins[i] >= 0){
                // ans = min(ans, solve(coins, amount-coins[i])+1);
                ans = min(ans + 0LL, solve(coins, amount-coins[i])+1LL); //LL for handle overflow

            }
        }
        return ans;
    }
    int coinChange(vector<int>& coins, int amount) {
        int ans = solve(coins, amount);
        return ans == INT_MAX ? -1 : ans;
    }
};
/*
Recursion + Memoization
new way of using dp
*/
class Solution {
public:
    int dp[10010];                                                           //1
    int solve(vector<int>&coins, int amount){
        //base case
        if(amount == 0){
            return 0;
        }

        //step3 if ans already present return it
        if(dp[amount] != -1){                                                //2
            return dp[amount];
        }

        //step2 recursive call
        int ans = INT_MAX;
        for(int i = 0 ; i < coins.size() ; i++){
            if(amount-coins[i] >= 0){
                // ans = min(ans, solve(coins, amount-coins[i])+1);
                ans = min(ans + 0LL, solve(coins, amount-coins[i])+1LL); //LL for handle overflow

            }
        }
        return dp[amount] = ans;                                             //3
    }
    int coinChange(vector<int>& coins, int amount) {
        memset(dp, -1, sizeof(dp));                                          //4
        int ans = solve(coins, amount);
        return ans == INT_MAX ? -1 : ans;
    }
};
/*
general way
*/
class Solution {
public:
    int solve(vector<int>&coins, int amount, vector<int>&dp){
        //base case
        if(amount == 0){
            return 0;
        }

        //step3 if ans already present return it
        if(dp[amount] != -1){
            return dp[amount];
        }

        //step2 recursive call
        int ans = INT_MAX;
        for(int i = 0 ; i < coins.size() ; i++){
            if(amount-coins[i] >= 0){
                // ans = min(ans, solve(coins, amount-coins[i])+1);
                ans = min(ans + 0LL, solve(coins, amount-coins[i], dp)+1LL); //LL for handle overflow

            }
        }
        return dp[amount] = ans;
    }
    int coinChange(vector<int>& coins, int amount) {
        //step1 create a dp vector
        vector<int>dp(amount+1, -1);                                             //the changing variable is amount not coins
        int ans = solve(coins, amount, dp);
        return ans == INT_MAX ? -1 : ans;
    }
};
/*
Example 1:
Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1

Example 2:
Input: coins = [2], amount = 3
Output: -1

Example 3:
Input: coins = [1], amount = 0
Output: 0
*/
    

//23. LONGEST INCREASING SUBSEQUENCE                               {T.C = O(N^2), S.C = O(N^2)}
class Solution {
public:
    int dp[2501][2501];
    int solveMem(vector<int>&nums , int i, int prev){
        int n = nums.size();
        //base case
        if(i >= n) return 0;

        if(dp[i][prev+1] != -1) return dp[i][prev+1];       //+1 for handle out of bound

        int incl = INT_MIN;                                 //max len
        if(prev == -1 || nums[i] > nums[prev]){             //1st ele or next greater then prev
            incl  = 1 + solveMem(nums, i+1, i);             //prev becomes currIdx(i)
        }
        int excl = 0 + solveMem(nums, i+1, prev);

        return dp[i][prev+1] = max(incl, excl);
    }
    int lengthOfLIS(vector<int>& nums) {
        memset(dp, -1, sizeof(dp));
        return solveMem(nums, 0, -1);           //0 = initial index, -1 = prevIdx
    }
};
/*
Example 1:
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

Example 2:
Input: nums = [0,1,0,3,2,3]
Output: 4

Example 3:
Input: nums = [7,7,7,7,7,7,7]
Output: 1
*/


//24. LONGEST COMMON SUBSEQUENCE                       
//USING TABULATION                                                 {T.C = O(N^2), S.C = O(N^2)}
class Solution {
public:
    int dp[1005][1005];
    int solveTab(string &s1, string &s2, int n, int m){
        for(int i = 1 ; i < n+1; i++){
            for(int j = 1 ;j < m+1 ; j++){
                if(s1[i-1] == s2[j-1]) dp[i][j] = 1 + dp[i-1][j-1];
                else dp[i][j] = 0 + max(dp[i-1][j], dp[i][j-1]);
            }
        }
        return dp[n][m];
    }
    int longestCommonSubsequence(string text1, string text2) {
        int n = text1.size(), m = text2.size();
        return solveTab(text1, text2, n, m);
    }
};

//USING MEMOZATION                                                {T.C = O(N^2), S.C = O(N^2)}
class Solution {
public:
    int dp[1005][1005];
    int solveMem(string &a, string &b, int i, int j){
        int n = a.length(), m = b.length();
        //base case
        if(i >= n || j >= m) return 0;

        if(dp[i][j] != -1) return dp[i][j];

        int count = 0;
        if(a[i] == b[j]){
            count = 1 + solveMem(a, b, i+1, j+1);
        }else{  //a[i] != b[i]
            count = 0 + max(solveMem(a, b, i+1, j), solveMem(a, b, i, j+1));  //incl, excl
        }
        return dp[i][j] = count;
    }
    int longestCommonSubsequence(string text1, string text2) {
        memset(dp, -1, sizeof(dp));
        return solveMem(text1, text2, 0, 0);             //0 , 0 = initial index of both text
    }
};
/*
Example 1:
Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.

Example 2:
Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.

Example 3:
Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.
*/


//25. WORD BREAK                                                  {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int dp[305];
    bool solveMem(string &s, unordered_set<string>&st, int i){
        int n = s.length();
        //base case
        if(i >= n) return true;                            //all find

        if(dp[i] != -1) return dp[i];

        string temp = "";
        for(int j = i ; j < n ; j++){                        //i not 0 (next word from currIdx)
            temp += s[j];
            if(st.count(temp)){
                if(solveMem(s, st, j+1)) return dp[i] = true;          //check of next word in dict
            }
        }
        return dp[i] = false;
    }
    bool wordBreak(string s, vector<string>& wordDict) {
        memset(dp, -1, sizeof(dp));
        unordered_set<string>st(wordDict.begin(), wordDict.end());
        return solveMem(s, st, 0);                       //0 = initial index
    }
};
/*
Example 1:
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

Example 2:
Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.

Example 3:
Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
*/


//26. COMBINATION SUM                                             {T.C = O(EXP / N^TARGET), S.C = O(N)}
class Solution {
public:
    void solve(vector<int>&cand, int target, vector<vector<int>>&ans, vector<int>&temp, int i){
        int n = cand.size();
        //base case
        if(target == 0){
            ans.push_back(temp);
            return;                                     //for next push
        }
        if(i >= n || target < 0) return;

        temp.push_back(cand[i]);
        solve(cand, target-cand[i], ans, temp, i);       //incl       // same element can use many times
        temp.pop_back();
        solve(cand, target, ans, temp, i+1);            //excl
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>>ans;
        vector<int>temp;
        sort(candidates.begin(), candidates.end());
        solve(candidates, target, ans, temp, 0);            //0 = initial index
        return ans;
    }
};
/*
Example 1:
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.

Example 2:
Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]

Example 3:
Input: candidates = [2], target = 1
Output: []
*/


//27. HOUSE ROBBER II                                              {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int dp1[105], dp2[105];
    int solveMem(vector<int>&nums, int i, int n, int *dp){      //for work both dp by refernce *
        //base case
        if(i >= n) return 0;

        if(dp[i] != -1) return dp[i];

        int incl = nums[i] + solveMem(nums, i+2, n, dp);
        int excl = 0 + solveMem(nums, i+1, n, dp);

        return dp[i] = max(incl, excl);
    }
    int rob(vector<int>& nums) {
        memset(dp1, -1, sizeof(dp1));
        memset(dp2, -1, sizeof(dp2));
        int n = nums.size();

        if(n == 1) return nums[0];
        if(n == 2) return max(nums[0], nums[1]);

        int inclZeroIdx = solveMem(nums, 0, n-1, dp1);             //here n = size, for idx = n-2
        int exclZeroIdx = solveMem(nums, 1, n, dp2);

        return max(inclZeroIdx, exclZeroIdx);
    }
};
/*
Example 1:
Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.

Example 2:
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

Example 3:
Input: nums = [1,2,3]
Output: 3
*/


//28. DECODE WAYS                                                  {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int dp[105];
    int solveMem(string &s, int i){
        int n = s.length();
        //base case
        if(i == n) return 1;                         //only 1 way to decode

        if(dp[i] != -1) return dp[i];

        int oneCharDecode = 0;                       //1 char decode (a to i)
        int twoCharDecode = 0;                       //2 char decode (11(j) to 26(z))
        if(s[i] != '0'){
            oneCharDecode = solveMem(s, i+1);
        }
        if(i+1 < n && (s[i] == '1' || (s[i] == '2' && s[i+1] <= '6'))){
            twoCharDecode = solveMem(s, i+2);        //skip two places
        }

        return dp[i] = oneCharDecode + twoCharDecode;
    }
    int numDecodings(string s) {
        memset(dp, -1, sizeof(dp));
        int n = s.length();
        //base case 
        if(s[0] == '0') return 0;                   //can't decode 05, 07...

        return solveMem(s, 0);                      //0 = initial index
    }
};
/*
Example 1:
Input: s = "12"
Output: 2
Explanation:
"12" could be decoded as "AB" (1 2) or "L" (12).

Example 2:
Input: s = "226"
Output: 3
Explanation:
"226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
*/


//29. UNIQUE PATHS                                                  {T.C = O(N^2), S.C = O(N^2)}
class Solution {
public:
    int dp[105][105];
    int solveMem(int n, int m, int i, int j){
        //base case
        if(i >= n || j >= m) return 0;
        if(i == n-1 && j == m-1) return 1;          //start == destination (1 way)

        if(dp[i][j] != -1) return dp[i][j];

        int down = solveMem(n, m, i+1, j);
        int right= solveMem(n, m, i, j+1);

        return dp[i][j] = down + right;
    }
    int uniquePaths(int m, int n) {
        memset(dp, -1, sizeof(dp));
        return solveMem(n, m, 0, 0);
    }
};
/*
Example 1:
Input: m = 3, n = 7
Output: 28

Example 2:
Input: m = 3, n = 2
Output: 3
Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down
*/


//30. JUMP GAME                                                   {T.C = O(N^2), S.C = O(N)}
class Solution {
public:
    int dp[10005];
    bool solveMem(vector<int>&nums, int i){
        int n = nums.size();
        //base case
        if(i == n-1) return true;
        if(i >= n) return false;

        if(dp[i] != -1) return dp[i];

        for(int j = 1 ; j <= nums[i] ; j++){              //1 = min jump, val = maxjump
            if(solveMem(nums, j+i) == true) return dp[i] = true;
        }
        return dp[i] = false;
    }
    bool canJump(vector<int>& nums) {
        memset(dp, -1, sizeof(dp));
        return solveMem(nums, 0);                //0 = initial index
    }
};
/*
Example 1:
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
*/


//31. JUMP GAME II                                              {T.C = O(N^2), S.C = O(N)}
class Solution {
public:
    int dp[10005];
    int solveMem(vector<int>&nums, int i){
        int n = nums.size();
        //base case
        if(i >= n-1) return 0;                   //reach end or more

        if(dp[i] != -1) return dp[i];

        int minJump = 1e5;                        //INT_MAX not working
        for(int j = 1 ; j <= nums[i] ; j++){
            int jumps = 1 + solveMem(nums, j+i);
            minJump = min(minJump, jumps);
        }
        return dp[i] = minJump;
    }
    int jump(vector<int>& nums) {
        memset(dp, -1, sizeof(dp));
        return solveMem(nums, 0);                //0 = initial index
    }
};
/*
Example 1:
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: nums = [2,3,0,1,4]
Output: 2
*/

// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
//32. CLONE GRAPH                                              {T.C = O(V+E), S.C = O(V)}         
class Solution {
public:
    void dfs(Node* root, vector<Node*>&vis, Node*node){
        vis[node->val] = node;
        for(auto it : root->neighbors){
            if(!vis[it->val]){
                Node* newNode = new Node(it->val);
                node->neighbors.push_back(newNode);
                dfs(it, vis, newNode);
            }else{
                node->neighbors.push_back(vis[it->val]);
            }
        }
    }
    Node* cloneGraph(Node* root) {
        //base case
        if(!root) return NULL;
        Node* node = new Node(root->val);
        vector<Node*>vis(100, NULL);                         //n  = 100(max nodes constraints)
        dfs(root, vis, node);

        return node;
    }
};
/*
Example 1:
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).

Example 2:
Input: adjList = [[]]
Output: [[]]
Explanation: Note that the input contains one empty list. The graph consists of only one node with val = 1 and it does not have any neighbors.

Example 3:
Input: adjList = []
Output: []
Explanation: This an empty graph, it does not have any nodes.
*/


//33. COURSE SCHEDULE                                             {T.C = O(V+E), S.C = O(V+E)}
class Solution {
public:
    vector<int>topoSort(vector<vector<int>>&edges, int n){
        unordered_map<int,vector<int>>adj;
        for(auto it : edges) adj[it[1]].push_back(it[0]);     //a <- b (b first)
        
        vector<int>inDegree(n, 0);
        for(auto it : edges){
            inDegree[it[0]]++;                                  //a <- b
        }

        queue<int>q;
        for(int i = 0 ; i < n ;i++){
            if(inDegree[i] == 0) q.push(i);                    //push index not ele
        }

        vector<int>ans;
        while(!q.empty()){
            auto frontNode = q.front();
            q.pop();
            ans.push_back(frontNode);
            for(auto it : adj[frontNode]){
                inDegree[it]--;
                if(inDegree[it] == 0) q.push(it);
            }
        }
        return ans;
    }
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int>topo = topoSort(prerequisites, numCourses);
        return topo.size() == numCourses;                   //all traverse, valid toposort
    }
};
/*
Example 1:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.

Example 2:
Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
*/


//34. PACIFIC ATLANTIC WATER FLOW                                 {T.C = O(N*M), S.C = O(N*M)}
class Solution {
public:
    vector<vector<int>>directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    bool isValid(int i, int j, int n, int m){
        return (i >= 0 && i < n && j >= 0 && j < m);
    }
    void dfs(vector<vector<int>>&grid, vector<vector<bool>>& vis, int i, int j) {
        int n = grid.size(), m = grid[0].size();
        vis[i][j] = true;
        for(auto it : directions){
            int newR = i + it[0];
            int newC = j + it[1];
 
            if(isValid(newR, newC, n, m) && !vis[newR][newC] && grid[newR][newC] >= grid[i][j]){   //new cell should high for water flow from outside to inside
                dfs(grid, vis, newR, newC);
            }
        }
    }
    /// main logic or trick for this problem : bahar se andar ki taraf jao
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        vector<vector<int>>ans;
        int n = heights.size(), m = heights[0].size();
        
        vector<vector<bool>> pacific(n, vector<bool>(m));
        vector<vector<bool>> atlantic(n, vector<bool>(m));
        
        for (int i = 0; i < n; i++) {
            dfs(heights, pacific, i, 0);               //pacific 1st col
            dfs(heights, atlantic, i, m-1);            //atlantic last col
        }
        
        for (int j = 0; j < m; j++) {
            dfs(heights, pacific, 0, j);              //pacific 1st row
            dfs(heights, atlantic, n-1, j);           //atlantic last col
        }

        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (pacific[i][j] && atlantic[i][j]) // agar uss particular point se dono oceans mai jaa paa rahe hai
                    ans.push_back({i,j});           // toh answer push kardo
            }
        }
        return ans;
    }
};
/*
Example 1:
Input: heights = [[1,2,2,3,5],
                  [3,2,3,4,4],
                  [2,4,5,3,1],
                  [6,7,1,4,5],
                  [5,1,1,2,4]]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
Explanation: The following cells can flow to the Pacific and Atlantic oceans, as shown below:
[0,4]: [0,4] -> Pacific Ocean 
       [0,4] -> Atlantic Ocean
[1,3]: [1,3] -> [0,3] -> Pacific Ocean 
       [1,3] -> [1,4] -> Atlantic Ocean
[1,4]: [1,4] -> [1,3] -> [0,3] -> Pacific Ocean 
       [1,4] -> Atlantic Ocean
[2,2]: [2,2] -> [1,2] -> [0,2] -> Pacific Ocean 
       [2,2] -> [2,3] -> [2,4] -> Atlantic Ocean
[3,0]: [3,0] -> Pacific Ocean 
       [3,0] -> [4,0] -> Atlantic Ocean
[3,1]: [3,1] -> [3,0] -> Pacific Ocean 
       [3,1] -> [4,1] -> Atlantic Ocean
[4,0]: [4,0] -> Pacific Ocean 
       [4,0] -> Atlantic Ocean
Note that there are other possible paths for these cells to flow to the Pacific and Atlantic oceans.

Example 2:
Input: heights = [[1]]
Output: [[0,0]]
Explanation: The water can flow from the only cell to the Pacific and Atlantic oceans.
*/


//35. NUMBER OF ISLANDS (IN A MATRIX)                             {T.C = O(N*M), S.C = O(N*M)}
class Solution {
public:
    vector<vector<int>>directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    bool isValid(int i, int j, int n, int m){
        return (i >= 0 && i < n && j >= 0 && j < m);
    }
    void dfs(vector<vector<char>>&grid, vector<vector<bool>>&vis, int i, int j){
        int n = grid.size(), m = grid[0].size();
        vis[i][j] = true;
        for(auto it : directions){
            int newR = i + it[0];
            int newC = j + it[1];

            if(isValid(newR, newC, n, m) && !vis[newR][newC] && grid[i][j] == '1'){
                dfs(grid, vis, newR, newC);
            } 
        }
    }
    int numIslands(vector<vector<char>>& grid) {
        int n = grid.size(), m = grid[0].size();
        vector<vector<bool>>vis(n, vector<bool>(m, false));
        int count = 0;
        for(int i = 0; i < n ; i++){
            for(int j = 0 ; j < m ; j++){
                if(!vis[i][j] && grid[i][j] == '1'){
                    count++;
                    dfs(grid, vis, i, j);
                }
            }
        }
        return count;
    }
};
/*
Example 1:
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1

Example 2:
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
*/


//36. CONNECTED COMPONENTS IN AN UNDIRECTED GRAPH              {T.C = O(V+E+CLOGC), S.C = O(V+E)}
class Solution {
  public:
    void dfs(unordered_map<int,vector<int>>&adj, vector<bool>&vis, vector<int>&components, int node){
        vis[node] = true;
        components.push_back(node);
        for(auto it : adj[node]){
            if(!vis[it]) dfs(adj, vis, components, it);
        }
    }
    vector<vector<int>> connectedcomponents(int v, vector<vector<int>>& edges) {
        unordered_map<int,vector<int>>adj;
        for(auto it : edges){
            adj[it[0]].push_back(it[1]);
            adj[it[1]].push_back(it[0]);
        }
        
        vector<bool>vis(v, false);
        // int components = 0;
        vector<vector<int>>ans;
        for(int i = 0 ; i < v ; i++){
            if(!vis[i]){
                vector<int>components;
                // components++;
                dfs(adj, vis, components, i);
                sort(components.begin(), components.end());           //sorting required in this particular ques
                ans.push_back(components);
            }
        }
        // return components;
        return ans;
    }
};
/*
Examples :
Input: e=3, v=5, edges = [{0, 1},{2, 1},{3, 4}]
Output: [[0, 1, 2],[3, 4]]
Explanation: 
Example of an undirected graph

Example 2:
Input: e=5, v=7, edges=[{0, 1},{6, 1},{2, 4},{2, 3},{3, 4}]
Output: [[0, 1, 6],[2, 3, 4],[5]]
*/


//37. LONGEST CONSECUTIVE SEQUENCE                               
//USING SET                                                       {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        int n = nums.size();
        int maxLen = 0;                           
        unordered_set<int>st(nums.begin(), nums.end());
        for(auto it : st){
            if(!st.count(it-1)){
                int count = 1;
                int nextEle = it;
                while(st.count(nextEle+1)){
                    count++;
                    nextEle++;
                }
                maxLen = max(maxLen, count);
            }
        }
        return maxLen;
    }
};
/*
Example 1:
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

Example 2:
Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
*/


//38. ALIEN DICTIONARY                                           {T.C = O(V+E), S.C = O(V+E)}
class Solution{
    public:
    vector<int>topoSort(int V , unordered_map<int, vector<int>>&adj){
        vector<int>ans;
        vector<int>inDegree(V, 0);
        
        //calculate indegrees for each nodes
        for(int i = 0 ; i < V ; i++){
            for(auto it : adj[i]){
                inDegree[it]++;
            }
        }
        
        queue<int>q;
        //push 0 indegree's in queue
        for(int i = 0 ; i < V ; i++){
            if(inDegree[i] == 0){
                q.push(i);
            }
        }
        
        //do dfs
        while(!q.empty()){
            int frontNode = q.front();
            q.pop();
            ans.push_back(frontNode);
            
            //reduce indegree of adjacent nodes after disconnection of node
            for(auto it : adj[frontNode]){
                inDegree[it]--;
                if(inDegree[it] == 0){
                    q.push(it);
                }
            }
        }
        return ans;
    }
    string findOrder(string dict[], int n, int k) {
        //create adjacency list
        unordered_map<int,vector<int>>adj;
        for(int i = 0 ; i < n-1 ; i++){             //compare till second last elemenent is possible
            string s1 = dict[i];
            string s2 = dict[i+1];
            int len = min(s1.size(), s2.size());    //compare till small string length
            for(int j = 0 ; j < len ; j++){
                if(s1[j] != s2[j]){                 //if char match skip
                    int u = s1[j]-'a';
                    int v = s2[j]-'a';
                    
                    adj[u].push_back(v);             //u before v (u -> v)
                    break;
                } 
            }
        }
        
        vector<int>topo = topoSort(k, adj);          //no. of v is k not n
        string ans = "";
        for(auto it : topo){
            ans = ans + char(it + 'a');
        }
        return ans;
    }
};
/*
Input:  n = 5, k = 4, dict = {"baa","abcd","abca","cab","cad"}
Output: 1
Explanation: Here order of characters is 'b', 'd', 'a', 'c' Note that words are sorted and in the given language "baa" comes before "abcd", therefore 'b' is before 'a' in output.
Similarly we can find other orders.

Input: n = 3, k = 3, dict = {"caa","aaa","aab"}
Output: 1
Explanation: Here order of characters is 'c', 'a', 'b' Note that words are sorted and in the given language "caa" comes before "aaa", therefore 'c' is before 'a' in output.
Similarly we can find other orders.
*/


//39. GRAPH VALID TREE                                           {T.C = O(V+E), S.C = O(V+E)}
//FOR TREE = GRAPH SHOULD = 1 COMPONENTS & NO CYCLE
bool dfsUnCycle(unordered_map<int,vector<int>>&adj, vector<bool>&vis, int node, int prev){
    vis[node] = true;
    for(auto it : adj[node]){
        if(!vis[it]){
            if(dfsUnCycle(adj, vis, it, node)) return true;              //node becomes prev
        }else if(it != prev) return true;           //visited but not parent (cycle present)
    }
    return false;
}
bool isCyclicSingleComp(unordered_map<int,vector<int>>&adj, int n){
    vector<bool>vis(n, false);
    int components = 0;
    bool hasCycle = false;
    for(int i = 0; i < n; i++){
        if(!vis[i]){
            components++;
            if(dfsUnCycle(adj, vis, i, -1)) hasCycle = true;
        } 
    }

    return (components == 1) && !hasCycle;        //tree = 1 component & no cycle
}
bool checkgraph(vector<vector<int>> edges, int n, int m){
    unordered_map<int,vector<int>>adj;
    for(auto it : edges){
        adj[it[0]].push_back(it[1]);
        adj[it[1]].push_back(it[0]);
    }
    return isCyclicSingleComp(adj, n);
}
/*
Sample Input 1 :
2
5
4
0 1
0 2
0 3
1 4
5
5
0 1
1 2
2 3
1 3
1 4
Sample Output 1 :
True
False
*/


//40. NUMBER OF CONNECTED COMPONENTS IN A GRAPH                  {T.C = O(V+E), S.C = O(V+E)}
class Solution {
  public:
    void dfs(unordered_map<int,vector<int>>&adj, vector<bool>&vis,  int node){
        vis[node] = true;
        for(auto it : adj[node]){
            if(!vis[it]) dfs(adj, vis, it);
        }
    }
    int connectedcomponents(int v, vector<vector<int>>& edges) {
        unordered_map<int,vector<int>>adj;
        for(auto it : edges){
            adj[it[0]].push_back(it[1]);
            adj[it[1]].push_back(it[0]);
        }
        
        vector<bool>vis(v, false);
        int components = 0;
        for(int i = 0 ; i < v ; i++){
            if(!vis[i]){
                components++;
                dfs(adj, vis, i);
            }
        }
        return components;
    }
};
/*
Input: e=3, v=5, edges = [{0, 1},{2, 1},{3, 4}]
Output: 2
Explanation: 
Example of an undirected graph

Example 2:
Input: e=5, v=7, edges=[{0, 1},{6, 1},{2, 4},{2, 3},{3, 4}]
Output: 3
*/


//41. INSERT INTERVAL                                             {T.C = O(N), S.C = O(N)}
class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        int n = intervals.size();
        vector<vector<int>>ans;
        int i = 0;
        while(i < n && intervals[i][1] < newInterval[0]){    //push before overlap intervals
            ans.push_back(intervals[i]);
            i++;
        }

        while(i < n && intervals[i][0] <= newInterval[1]){    //overlap intervals (update intervals then push)
            newInterval[0] = min(newInterval[0], intervals[i][0]);
            newInterval[1] = max(newInterval[1], intervals[i][1]);
            i++;
        }
        ans.push_back(newInterval);

        while(i < n){                                       //after overlap intervals (rest of intervals)
            ans.push_back(intervals[i]);
            i++;
        }
        return ans;
    }
};
/*
Example 1:
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

Example 2:
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
*/


//42. MERGE INTERVALS                                           {T.C = O(N*LOGN), S.C = O(N)}
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        int n = intervals.size();
        sort(intervals.begin(), intervals.end());
        vector<vector<int>>ans;
        vector<int>tempInterval = intervals[0];
        for(int i = 1 ; i < n ;i++){
            if(tempInterval[1] >= intervals[i][0]){           //merge (prevEnd == currStart) also
                tempInterval[0] = min(tempInterval[0], intervals[i][0]);
                tempInterval[1] = max(tempInterval[1], intervals[i][1]);
            }else{
                ans.push_back(tempInterval);
                tempInterval = intervals[i];                 //move next
            }
        }
        ans.push_back(tempInterval);                   //for last interval
        return ans;
    }
};
/*
Example 1:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

Example 2:
Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.
*/


//43. NON OVERALAPPING INTERVALS                                  {T.C = O(N*LOGN), S.C = O(1)}
//for erase overalap interval (we have to think greedly so sort on basis of ending (high have good chances to overlap more intervals))
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        int n = intervals.size();
        auto lambda = [&](auto &a, auto &b){
            return (a[1] < b[1]);
        };
        int count = 0;
        sort(intervals.begin(), intervals.end(), lambda);   //large ending time , have more chance to overlap to other(better to remove that)
        vector<int>tempInterval = intervals[0];
        // for(auto it : intervals){               //in this 1st(tempInterval) intvl is also included
        for(int i = 1; i < n; i++){
            if(intervals[i][0] < tempInterval[1]){           //currentStart < prevEnd
                count++;
            }else{
                tempInterval = intervals[i];
            }
        }
        return count;
    }
};
/*
Example 1:
Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.

Example 2:
Input: intervals = [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.

Example 3:
Input: intervals = [[1,2],[2,3]]
Output: 0
Explanation: You don't need to remove any of the intervals since they're already non-overlapping.
*/


//44. REPEATING AND MISSING                                     {T.C = O(N), S.C = O(N)} 
class Solution {
  public:
    vector<int> findTwoElement(vector<int>& arr) {
        int n = arr.size();
        unordered_map<int,int>mp;
        for(auto it : arr) mp[it]++;
        
        int repeating = -1, missing = -1;
        for(int i = 1 ; i <= n; i++){
            if(mp[i] == 2) repeating = i;
            if(mp[i] == 0) missing = i;
        }
        return {repeating, missing};
    }
};
/*
Input: arr[] = [2, 2]
Output: [2, 1]
Explanation: Repeating number is 2 and smallest positive missing number is 1.

Input: arr[] = [1, 3, 3] 
Output: [3, 2]
Explanation: Repeating number is 3 and smallest positive missing number is 2.

Input: arr[] = [4, 3, 6, 2, 1, 1]
Output: [1, 5]
Explanation: Repeating number is 1 and the missing number is 5.
*/


/*Definition of Interval:
class Interval {
public:
    int start, end;
    Interval(int start, int end) {
        this->start = start;
        this->end = end;
    }
}
*/
//45. MEETING ROOMS I                                             {T.C = O(N*LOGN), S.C = O(1)}
class Solution {
public:
    bool canAttendMeetings(vector<Interval>& intervals) {
        int n = intervals.size();
        //base case
        if(n == 0) return true;
        auto lambda = [&](auto &a, auto &b){
            return a.start < b.start;
        };
        sort(intervals.begin(), intervals.end(), lambda);           //according to start time
        Interval tempInterval = intervals[0];
        for(int i = 1 ; i < n ; i++){
            if(tempInterval.end > intervals[i].start) return false;
            else tempInterval = intervals[i];
        }
        return true;
    }
};
/*
Example 1:
Input: intervals = [(0,30),(5,10),(15,20)]
Output: false
Explanation:
(0,30) and (5,10) will conflict
(0,30) and (15,20) will conflict

Example 2:
Input: intervals = [(5,8),(9,15)]
Output: true
*/


//46. MEETING ROOMS II                                           {T.C = O(N*LOGN), S.C = O(N)}
class Solution {
public:
    int minMeetingRooms(vector<Interval>& intervals) {
        int n = intervals.size();
        //base case
        if(n == 0) return 0;                 //no interval
        
        auto lambda = [&](auto &a, auto &b){
            return a.start < b.start;
        };
        sort(intervals.begin(), intervals.end(), lambda);  //sort by start time
        
        priority_queue<int, vector<int>, greater<int>>minHeap;
        Interval tempInterval = intervals[0];
        minHeap.push(tempInterval.end);                  //push only end time of meet
        for(int i = 1 ; i < n ; i++){
            if(minHeap.top() <= intervals[i].start){       //pop curr for adjust next
                minHeap.pop();
            }
            minHeap.push(intervals[i].end);
        }
        return minHeap.size();                            //required minimum rooms
    }
};
/*
Example 1:
Input: intervals = [(0,40),(5,10),(15,20)]
Output: 2
Explanation:
day1: (0,40)
day2: (5,10),(15,20)

Example 2:
Input: intervals = [(4,9)]
Output: 1
*/


//47. LINKED LIST CYCLE                                           {T.C = O(N), S.C = O(1)}
class Solution {
public:
    bool hasCycle(ListNode *head) {
        //base case
        if(!head) return false;

        ListNode* slow = head;
        ListNode* fast = head;
        while(fast && fast->next){            //all required
            slow = slow->next;
            fast = fast->next->next;
            if(slow == fast) return true;
        }
        return false;
    }
};
/*
Example 1:
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

Example 2:
Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.

Example 3:
Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.
*/


//48. MERGE 2 SORTED LIST                                         {T.C = O(N), S.C = O(1)}
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        //base case
        if(!list1) return list2;
        if(!list2) return list1;

        if(list1->val <= list2->val){
            list1->next = mergeTwoLists(list1->next, list2);
            return list1;
        }else{
            list2->next = mergeTwoLists(list1, list2->next);
            return list2;
        }
    }
};

//ITERATIVE APPROACH
class Solution {
public:
    ListNode* merge(ListNode* first, ListNode* second){
        ListNode* curr1 = first;
        ListNode* next1 = first->next;
        ListNode* curr2 = second;
        ListNode* next2 = second->next;

        if(!first->next){
            first->next = second;
            return first;
        }

        while(next1 && curr2){
            if(curr2->val >= curr1->val && curr2->val <= next1->val){
                //insert in between 1 linkedlist
                curr1->next = curr2;
                next2 = curr2->next;
                curr2->next = next1;
                
                //update pointer
                curr1 = curr2;
                curr2 = next2;
            }else{
                //move forward
                curr1 = curr1->next;
                next1 = next1->next;

                if(!next1){
                    curr1->next = curr2;
                    return first;
                }
            }
        }
        return first;
    }
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if(!list1){
            return list2;
        }
        if(!list2){
            return list1;
        }

        if(list1->val <= list2->val){
            return merge(list1, list2);
        }else{
            return merge(list2, list1);
        }
    }
};
/*
Example 1:
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

Example 2:
Input: list1 = [], list2 = []
Output: []

Example 3:
Input: list1 = [], list2 = [0]
Output: [0]
*/


//49. MERGE K SORTED ARRAYS                                       {T.C = O(N*LOGN), S.C = O(N)},N = K^2
class Solution{
public:
    typedef pair<int,pair<int,int>>P;
    vector<int> mergeKArrays(vector<vector<int>> arr, int K){
        int n = arr.size();               //n == m == K
        vector<P>temp;
        for(int row = 0 ; row < n ; row++){               //put 1, 2, 3.. k , col element
            temp.push_back({arr[row][0], {row, 0}});
        }
        
        priority_queue<P, vector<P>, greater<P>>minHeap(temp.begin(), temp.end()); //insert all ele of temp in minHeap
        vector<int>ans;                             //extract ans from heap and push in ans vec
        while(!minHeap.empty()){
            auto topNode = minHeap.top();
            minHeap.pop();
            int val = topNode.first;
            int row = topNode.second.first;
            int col = topNode.second.second;
            
            ans.push_back(val);
            
            if(col+1 < K) minHeap.push({arr[row][col+1], {row, col+1}});
        }
        return ans;
    }
};
/*
Input: k = 3, arr[][] = {{1,2,3},{4,5,6},{7,8,9}}
Output: 1 2 3 4 5 6 7 8 9
Explanation: Above test case has 3 sorted arrays of size 3, 3, 3 arr[][] = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]. The merged list will be [1, 2, 3, 4, 5, 6, 7, 8, 9].

Input: k = 4, arr[][]={{1,2,3,4},{2,2,3,4},{5,5,6,6},{7,8,9,9}}
Output: 1 2 2 2 3 3 4 4 5 5 6 6 7 8 9 9 
Explanation: Above test case has 4 sorted arrays of size 4, 4, 4, 4 arr[][] = [[1, 2, 2, 2], [3, 3, 4, 4], [5, 5, 6, 6], [7, 8, 9, 9 ]]. The merged list will be [1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 9].
*/


//50. REMOVE NTH NODE FROM THE END OF LINKED LIST               {T.C = O(N), S.C = O(1)}
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(NULL);
        dummy->next = head;

        ListNode* slow = dummy;
        ListNode* fast = dummy;
        
        while(n--){                                  //reach to nth from start node
            fast = fast->next;
        }

        while(fast->next){                           //move until list end & move simultaneously(slow , fast)
            slow = slow->next;
            fast = fast->next;
        }
        slow->next = slow->next->next;               //slow points to nth node from end

        return dummy->next;               //head
    }
};
/*
Example 1:
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Example 2:
Input: head = [1], n = 1
Output: []

Example 3:
Input: head = [1,2], n = 1
Output: [1]
*/


//51. REMOVE NTH NODE FROM THE START OF LINKED LIST               {T.C = O(N), S.C = O(1)}
class Solution {
  public:
    Node* deleteNode(Node* head, int x) {
        Node* curr = head;
        if(x == 1){
            curr = curr->next;
            return curr;
        }
        
        for(int i = 1 ; i < x-1 ; i++){
            curr = curr->next;
        }
        curr->next = curr->next->next;
        return head;
    }
};
/*
Input: Linked list: 1 -> 3 -> 4, x = 3
Output: 1 -> 3
Explanation: After deleting the node at the 3rd position (1-base indexing), the linked list is as 1 -> 3. 

Input: Linked list: 1 -> 5 -> 2 -> 9, x = 2 
Output: 1 -> 2 -> 9
Explanation: After deleting the node at 2nd position (1-based indexing), the linked list is as 1 -> 2 -> 9.
*/


//52. REORDER LIST                                               {T.C = O(N), S.C = O(1)}
class Solution {
public:
    ListNode* reverse(ListNode* node){
        ListNode* prev = NULL;
        ListNode* curr = node;
        ListNode* forw = NULL;

        while(curr){
            forw = curr->next;
            curr->next = prev;
            prev = curr;
            curr = forw;
        }
        return prev;
    }
    void reorderList(ListNode* head) {
        //base case
        if(!head || !head->next) return;

        ListNode* slow = head;
        ListNode* fast = head;

        while(fast && fast->next){
            slow = slow->next;
            fast = fast->next->next;
        }    
        ListNode* nextHalf = reverse(slow->next);    //next start from next of slow
        slow->next = NULL;                           //break prevHalf & nexHalf
        ListNode* prevHalf = head;

        while(prevHalf && nextHalf){
            ListNode* temp1 = prevHalf->next;
            ListNode* temp2 = nextHalf->next;

            prevHalf->next = nextHalf;
            nextHalf->next = temp1;

            prevHalf = temp1;
            nextHalf = temp2;
        }
    }
};
/*
Example 1:
Input: head = [1,2,3,4]
Output: [1,4,2,3]

Example 2:
Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]
*/


//53. SET MATRIX ZEROES                                          {T.C = O(N*M), S.C = O(N+M)}
//APRROACH 1(BRUTE FORCE)                                                        {T.C = O(N^3), S.C = O(N^2)}
//use 2d matrix store ans in this then copy element in original matrix 
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int n = matrix.size();
        int m = matrix[0].size();
        vector<vector<int>>vis = matrix;

        //row 0
        for(int i = 0 ; i < n ; i++){
            for(int j = 0 ; j < m ; j++){
                if(matrix[i][j] == 0){
                    for(int k = 0; k < n ; k++){
                        vis[k][j] = 0;
                    }
                }
            }
        }

        //col 0
        for(int i = 0 ; i < n ; i++){
            for(int j = 0 ; j < m; j++){
                if(matrix[i][j] == 0){
                    for(int k = 0 ; k < m ; k++){
                        vis[i][k] = 0;
                    }
                }
            }
        }

        //now copy visited matrix to actual matrix
        for(int i = 0 ; i < n ; i++){
            for(int j = 0; j < m; j++){
                matrix[i][j] = vis[i][j];
            }
        }
    }
};
//APPROACH 2(using 2 single vectors)                                              {T.C = O(N^2), S.C = O(N)}
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int n = matrix.size();
        int m = matrix[0].size();
        vector<int>row(n+1, 0);
        vector<int>col(m+1, 0);

        for(int i = 0 ; i < n ; i++){
            for(int j = 0 ; j < m ; j++){
                if(matrix[i][j] == 0){
                    row[i] = 1;
                    col[j] = 1;
                }
            }
        }

        for(int i = 0 ; i < n ; i++){
            for(int j = 0 ; j < m; j++){
                if(row[i] || col[j]){                //if row or col updated(to 1)
                    matrix[i][j] = 0;
                }
            }
        }
    }
};

/*
Input: matrix = [
                 [1,1,1],
                 [1,0,1],
                 [1,1,1]
                ]
Output: [
         [1,0,1]
        ,[0,0,0],
         [1,0,1]
        ]
*/


//54. SPIRAL MATRIX                                               {T.C = O(N*M), S.C = O(N*M)}
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int>ans;
        int n = matrix.size(), m = matrix[0].size();
        int rStart = 0, cStart = 0;
        int rEnd = n-1, cEnd = m-1;

        while(rStart <= rEnd && cStart <= cEnd){
            //print first row
            for(int i = cStart ; i <= cEnd ; i++){
                ans.push_back(matrix[rStart][i]);
            }
            rStart++;
            //print last col
            for(int i = rStart ; i <= rEnd ; i++){
                ans.push_back(matrix[i][cEnd]);
            }
            cEnd--;
            //print last row (first check)
            if(rStart <= rEnd){
                for(int i = cEnd ; i >= cStart ; i--){
                    ans.push_back(matrix[rEnd][i]);
                }
                rEnd--;
            }
            //print 1st col
            if(cStart <= cEnd){
                for(int i = rEnd ; i >= rStart ; i--){
                    ans.push_back(matrix[i][cStart]);
                }
                cStart++;
            }
        }

        return ans;
    }
};
/*
Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]

Example 2:
Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
*/


//55. ROTATE IMAGE                                               {T.C = O(N*M), S.C = O(1)}
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();             //n = m
        for(int i = 0 ; i < n; i++){                  //transpose of matrix
            for(int j = i ; j < n ; j++){
                swap(matrix[i][j], matrix[j][i]);
            }
        }

        //reverse each row
        for(int i = 0 ; i < n; i++){
            int l = 0, r = n-1;
            while(l <= r){
                swap(matrix[i][l], matrix[i][r]);
                l++, r--;
            }
        }
    }
};
/*
Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

Example 2:
Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
*/




//56. LONGEST SUBSTRING WITHOUT REPEATING CHARACTERS             {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int n = s.length();
        int maxLen = 0;
        unordered_map<int,int>mp;
        int i = 0, j = 0;
        while(j < n){
            mp[s[j]]++;

            while(mp[s[j]] > 1){                  //if freq > 1 remove it
                mp[s[i]]--;
                if(mp[s[i]] == 0) mp.erase(s[i]);
                i++;
            }
            maxLen = max(maxLen, j-i+1);
            j++;
        }
        return maxLen;
    }
};
/*
Example 1:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Example 2:
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
*/


//57. LONGEST REPEATING CHARACTER REPLACEMENT                    {T.C = O(N), S.C = O(N)}    
class Solution {
public:
    int characterReplacement(string s, int k) {
        int n = s.length();
        unordered_map<int,int>mp;
        int maxLen = 0, maxFreq = 0;
        int i = 0, j = 0;

        while (j < n) {
            mp[s[j]]++;
            maxFreq = max(maxFreq, mp[s[j]]);

            if ((j-i+1) - maxFreq > k) {          //sz of currWin - freq (replacement required)   
                mp[s[i]]--;
                i++;
            }

            maxLen = max(maxLen, j-i+1);
            j++;
        }

        return maxLen;
    }
};
/*
Example 1:
Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.

Example 2:
Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.
There may exists other ways to achieve this answer too.
*/


//58. MINIMUM WINDOW SUBSTRING                                   {T.C = O(N), S.C = O(N)}
class Solution {
public:
    string minWindow(string s, string t) {
        int n = s.length(), m = t.length();
        unordered_map<int,int>mp;
        for(auto it : t) mp[it]++;
        int minLen = INT_MAX;
        int count = mp.size();
        int i = 0, j = 0, start = 0;
        while(j < n){
            if(mp.count(s[j])){
                mp[s[j]]--;
                if(mp[s[j]] == 0) count--;             //not erase 
            }
            while(count == 0){                    //shring window , finding minLen
                if(mp.count(s[i])){
                    mp[s[i]]++;
                    if(mp[s[i]] == 1){
                        count++;
                        if(minLen > j-i+1){
                            minLen = j-i+1;
                            start = i;           //window starting idx
                        }
                    }
                }
                i++;
            }
            j++;
        }
        return minLen == INT_MAX ? "" : s.substr(start, minLen);
    }
};
/*
Example 1:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

Example 2:
Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.

Example 3:
Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.
*/


//59. VALID ANAGRAM 
//BRUTE FORCE                                                    {T.C = O(N*LOGN), S.C = O(1)}
class Solution {
public:
    bool isAnagram(string s, string t) {
        sort(s.begin(), s.end());
        sort(t.begin(), t.end());
        return s == t;
    }
};

//USING MAP                                                     {T.C = O(N), S.C = O(N)}
class Solution {
public:
    bool isAnagram(string s, string t) {
        //base case
        if(s.length() != t.length()) return false;
        unordered_map<char,int>mp;
        for(auto it : s){
            mp[it]++;
        }
        for(auto it : t){
            if(mp.count(it)){
                mp[it]--;
                if(mp[it] == 0) mp.erase(it);
            }
        }
        if(mp.empty()) return true;
        return false;
    }
};
/*
Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

Example 2:
Input: s = "rat", t = "car"
Output: false
*/


//60. GROUP ANAGRAMS
//USING SORTING + MAP                                           {T.C = O(N*(M*LOGM)), S.C = O(N)}
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& s) {
        int n = s.size();
        unordered_map<string, vector<string>>mp;             //actual -> sorted vector push (actual)
        for(auto it : s){
            string temp = it;
            sort(temp.begin(), temp.end());
            mp[temp].push_back(it);
        }
        vector<vector<string>>ans;
        for(auto it : mp){
            ans.push_back(it.second);                      //push collected vectors
        }
        return ans;
    }
};

//WITHOUT SORTING (FREQUENCY VECTOR)                            {T.C = O(N*(M+26)), S.C = O(N)}
class Solution {
public:
    string generate(string &word){
        vector<int>freq(26, 0);
        for(auto it : word){
            freq[it-'a']++;
        }

        //search similar freq word (anagram)
        string newWord = "";
        for(int i = 0 ; i < 26 ; i++){
            if(freq[i] > 0) newWord += string(freq[i], i + 'a');  //+'a' for finding word(char)
        }
        return newWord;
    }
    vector<vector<string>> groupAnagrams(vector<string>& s) {
        int n = s.size();
        unordered_map<string, vector<string>>mp;             //actual -> sorted vector push (actual)
        for(auto it : s){
            string temp = it;
            // sort(temp.begin(), temp.end());
            // mp[temp].push_back(it);
            string newWord = generate(temp);
            mp[newWord].push_back(temp);
        }
        vector<vector<string>>ans;
        for(auto it : mp){
            ans.push_back(it.second);                      //push collected vectors
        }
        return ans;
    }
};
/*
Example 1:
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
Explanation:
There is no string in strs that can be rearranged to form "bat".
The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.

Example 2:
Input: strs = [""]
Output: [[""]]

Example 3:
Input: strs = ["a"]
Output: [["a"]]
*/


//61. VALID PARENTHESES                                           {T.C = O(N), S.C = O(N)}
class Solution {
public:
    bool isValid(string s) {
        stack<char>stk;
        for(auto it : s){
            if(it == '(' ||  it == '{' || it == '[') stk.push(it);
            else if( !stk.empty() && ( (it == ')' && stk.top() == '(' ) || (it == ']' && stk.top() == '[') || (it == '}' && stk.top() == '{') )) stk.pop();
            else return false;
        }
        return stk.empty();
    }
};
/*
Example 1:
Input: s = "()"
Output: true

Example 2:
Input: s = "()[]{}"
Output: true

Example 3:
Input: s = "(]"
Output: false

Example 4:
Input: s = "([])"
Output: true
*/


//62. CHECK PALINDROME
class Solution {
public:
    bool isPalindrome(int x) {
        string s = to_string(x);
        int n = s.length();
        int i = 0, j = n-1;
        while(i < j){
            if(s[i] != s[j]) return false;
            i++, j--;
        }
        return true;
    }
};
/*
Example 1:
Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.

Example 2:
Input: x = -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.

Example 3:
Input: x = 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
*/

//63. LONGEST PALINDROMIC SUBSTRING
//USING DP                                                        {T.C = O(N^2), S.C = O(N^2)}
class Solution {
public:
    int dp[1005][1005];
    bool solveMem(string &s, int i, int j){
        //base case
        if(i >= j) return true;               //single char, or empty string is true

        if(dp[i][j] != -1) return dp[i][j];

        if(s[i] == s[j]) return dp[i][j] = solveMem(s, i+1, j-1);      //first , last match
        return false;
    }
    string longestPalindrome(string s) {
        memset(dp, -1, sizeof(dp));
        int n = s.length();
        int maxLen = 0;
        int startIdx = 0;
        for(int i = 0 ; i < n; i++){
            for(int j = i ; j < n ; j++){                         //SUBSTRING STARTING FROM i.
                if(solveMem(s, i, j)){
                    if(j-i+1 > maxLen){
                        maxLen = j-i+1;
                        startIdx = i;
                    }
                }
            }
        }
        return s.substr(startIdx, maxLen);
    }
};

//USING BOTTOM UP DP 
//(MIK TEMPLATE FOR PALINDROMES(COUNT, LONGEST PALINDROMIC SUBSTRING OR SUBSEQUENCE)) {T.C = O(N^2), S.C = O(N^2)}
class Solution {
public:
    //USING MIK BLUEPRINT
    int countSubstrings(string s) {
        int n = s.length();
        vector<vector<int>>dp(n, vector<int>(n, false));
        int count = 0;

        for(int l = 1 ; l <= n ; l++){
            for(int i = 0 ; i+l-1 < n; i++){                  //i+L-1(curr len of substring)
                int j = i+l-1;
                if(i == j)   dp[i][j] = true;                            //1 len string
                else if(i+1 == j) dp[i][j] = (s[i] == s[j]);             //2 len string
                else{
                    dp[i][j] = (s[i] == s[j] && dp[i+1][j-1] == true);   //>2 len string
                }

                if(dp[i][j] == true) count++;
            }
        }
        return count;
    }
};

//WITHOUT DP                                                     {T.C = O(N^2), S.C = O(N)}
class Solution {
public:
    string expandFromCenter(string &s, int left, int right){
        while(left >= 0 && right < s.length() && s[left] == s[right]){
            left--;
            right++;
        }
        return s.substr(left+1, right-left-1);          //(starting index, length)
    }
    string longestPalindrome(string s) {
        int n = s.length();

        //base case
        if(s.length() <= 1) return s;

        string maxStr = s.substr(0, 1);                 //(starting index, length)
        for(int i = 0 ; i < n-1 ; i++){
            string odd = expandFromCenter(s, i, i);     //partition on char
            string eve = expandFromCenter(s, i, i+1);   //partition in between char

            if(odd.length() > maxStr.length()){
                maxStr = odd;
            }
            if(eve.length() > maxStr.length()){
                maxStr = eve;
            }
        }
        return maxStr;
    }
};
/*
Example 1:
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

Example 2:
Input: s = "cbbd"
Output: "bb"
*/


//64. PALINDROMIC SUBSTRING                                      
//BRUTE FORCE                                                     {T.C = O(N^3), S.C = O(1)}
class Solution {
public:
    bool isPalindrome(const string &s){
        int n = s.length();
        int i = 0, j = n-1;
        while(i < j){
            if(s[i] != s[j]) return false;
            i++ , j--;
        }
        return true;
    }
    int countSubstrings(string s) {
        int n = s.length();
        int count = 0;
        for(int i = 0 ; i < n ; i++){
            for(int j = i ; j < n ; j++){
                if(isPalindrome(s.substr(i, j-i+1))) count++;
            }
        }
        return count;
    }
};

//SMART APPROACH                                                  {T.C = O(N^2), S.C = O(1)}
class Solution {
public:
    void checkOddAndEven(int i, int j, string &s, int &count){
        int n = s.length();
        while(i >= 0 && j < n && s[i] == s[j]){
            count++;
            i--, j++;
        }
    }
    int countSubstrings(string s) {
        int n = s.length();
        int count = 0;
        for(int i = 0 ; i < n ; i++){
            checkOddAndEven(i, i,   s, count);
            checkOddAndEven(i, i+1, s, count);
        }
        return count;
    }
};
/*
Example 1:
Input: s = "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".

Example 2:
Input: s = "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
*/


//65. ENCODE AND DECODE STRINGS                                   {T.C = O(N), S.C = O(N)}
class Solution {
public:
    string encode(vector<string>& strs) {
        string encodeStr = "";
        for(auto it : strs){
            encodeStr += to_string(it.length()) + '#' + it;
        }
        return encodeStr;
    }

    vector<string> decode(string s) {
        vector<string>decodeStr;
        int i = 0;
        while(i < s.length()){
            int j = i;
            while(s[j] != '#') j++;

            int len = stoi(s.substr(i, j-i));
            string str = s.substr(j+1, len);
            decodeStr.push_back(str);
            i = j+1+len;
        }
        return decodeStr;
    }
};
/*
Example 1:
Input: ["neet","code","love","you"]
Output:["neet","code","love","you"]

Example 2:
Input: ["we","say",":","yes"]
Output: ["we","say",":","yes"]
*/




//66. SAME TREE                                                 {T.C = O(N), S.C = O(H)} 
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if(!p && !q) return true;
        if(!p || !q) return false;

        if(p->val != q->val) return false;

        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
};
/*
Example 1:
Input: p = [1,2,3], q = [1,2,3]
Output: true

Example 2:
Input: p = [1,2], q = [1,null,2]
Output: false

Example 3:
Input: p = [1,2,1], q = [1,1,2]
Output: false
*/


//67. INVERT BINARY TREE                                          {T.C = O(N), S.C = O(H)}
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if(!root) return NULL;

        TreeNode* temp = root->left;
        root->left = root->right;
        root->right = temp;
        invertTree(root->left);
        invertTree(root->right);

        return root;
    }
};
/*
Example 1:
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]

Example 2:
Input: root = [2,1,3]
Output: [2,3,1]

Example 3:
Input: root = []
Output: []
*/


//68. BINARY TREE MAXIMUM PATH SUM                               {T.C = O(N), S.C = O(H)}
class Solution {
public:
    int solve(TreeNode* root, int &maxi){
        //base case
        if(!root) return 0;

        int left = solve(root->left, maxi);
        int right = solve(root->right, maxi);
        
        int anyOnePath = max(left, right) + root->val;       //root->val == 1
        int onlyNode = root->val;
        int onlyPath = left + right + root->val;

        maxi = max({maxi, anyOnePath, onlyNode, onlyPath});
        
        return max(anyOnePath, onlyNode);                //onlyPath cant call for parent(its already ans)
        // return maxi;
    }
    int maxPathSum(TreeNode* root) {
        int maxi = INT_MIN;
        solve(root, maxi);
        return maxi;
    }
};
/*
Example 1:
Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.

Example 2:
Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.
*/


//69. BINARY TREE LEVEL ORDER TRAVERSAL                         {T.C = O(N), S.C = O(N)} 
class Solution {
public:
    void lvlOrder(TreeNode* root, vector<vector<int>>&ans){
        //base case
        if(!root) return;

        queue<TreeNode*>q;
        q.push(root);
        while(!q.empty()){
            int sz = q.size();
            vector<int>lvlNodes;
            for(int i = 0 ; i < sz ; i++){
                auto frontNode = q.front();
                q.pop();
                lvlNodes.push_back(frontNode->val);
                if(frontNode->left) q.push(frontNode->left);
                if(frontNode->right)q.push(frontNode->right);
            }
            ans.push_back(lvlNodes);
        }
    }
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>>ans;
        lvlOrder(root, ans);
        return ans;
    }
};
/*
Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]

Example 2:
Input: root = [1]
Output: [[1]]

Example 3:
Input: root = []
Output: []
*/


//70. SERIALIZE AND DESERIALIZE BINARY TREE                      {T.C = O(N), S.C = O(N)}
class Codec {
public:
    void btToStr(TreeNode* root, string &serAns){
        // base case
        if(!root){
            serAns += "N ";  // for NULL
            return;
        }
        // using preorder
        serAns += to_string(root->val) + ' ';
        btToStr(root->left, serAns);
        btToStr(root->right, serAns);
    }

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string serAns = "";
        btToStr(root, serAns);
        return serAns;
    }

    TreeNode* strToBt(string &data, int &i){
        if(data[i] == 'N'){
            i += 2;  // skip 'N' and space
            return NULL;
        }

        string temp = "";
        while(data[i] != ' '){
            temp += data[i];
            i++;
        }
        //preorder
        TreeNode* root = new TreeNode(stoi(temp));
        i++;  // skip space
        root->left = strToBt(data, i);
        root->right = strToBt(data, i);

        return root;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        int i = 0;
        return strToBt(data, i);
    }
};
/*
Example 1:
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]

Example 2:
Input: root = []
Output: []
*/


//71. SUBTREE OF ANOTHER TREE                                   {T.C = O(N*M), S.C = O(H)}
class Solution {
public:
    bool isIdentical(TreeNode* p, TreeNode* q){
        //base case
        if(!p && !q) return true;
        if(!p || !q) return false;
        if(p->val != q->val) return false;
        
        return (isIdentical(p->left, q->left) && isIdentical(p->right, q->right));
    }
    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
        if(!root) return false;
        if(root->val == subRoot->val){
            if(isIdentical(root, subRoot)) return true;
        }
        return (isSubtree(root->left, subRoot) || isSubtree(root->right, subRoot));  //either side subtree present
    }
};
/*
Example 1:
Input: root = [3,4,5,1,2], subRoot = [4,1,2]
Output: true

Example 2:
Input: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
Output: false
*/



//72a. CONSTRUCT BINARY TREE FROM PREORDER AND INORDER TRAVERSAL     {T.C = O(N), S.C = O(H)}
//PASS IDX BY REFERENCE , FIND ROOT IN PREORDER(START), SEARCH IN INORDER (LEFT, RIGTH SUBTREE), MAKE NEW ROOT
class Solution {
public:
    TreeNode* solve(vector<int>&preorder, vector<int>&inorder, int inOrStart, int inOrEnd , int &preOrIdx){
        //base case
        if(inOrStart > inOrEnd) return NULL;

        int rootVal = preorder[preOrIdx];
        preOrIdx++;
        int i = inOrStart;
        for(i = inOrStart ; i <= inOrEnd ; i++){           //search in inorder
            if(inorder[i] == rootVal) break;
        }

        TreeNode* root = new TreeNode(rootVal);
        //s__i-1 (i) i+1__e
        root->left = solve(preorder, inorder, inOrStart, i-1, preOrIdx);   
        root->right = solve(preorder, inorder, i+1, inOrEnd, preOrIdx);

        return root;
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = preorder.size();                  //n = m = inorder(same tree)
        int preOrIdx = 0;                              //pass by reference only
        return solve(preorder, inorder, 0, n-1, preOrIdx);    //0 = inOrStart, n-1 = inOrEnd
    }
};
/*
Example 1:
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

Example 2:
Input: preorder = [-1], inorder = [-1]
Output: [-1]
*/


//72b. CONSTRUCT BINARY TREE FROM POSTORDER AND INORDER TRAVERSAL     {T.C = O(N), S.C = O(H)}
//PASS IDX BY REFERENCE , FIND ROOT IN POSTORDER(START), SEARCH IN INORDER (LEFT, RIGTH SUBTREE), MAKE NEW ROOT
class Solution {
public:
    TreeNode* solve(vector<int>&inorder, vector<int>&postorder, int inOrStart, int inOrEnd, int &poOrIdx){
        //base case
        if(inOrStart > inOrEnd) return NULL;
        int rootVal = postorder[poOrIdx];
        poOrIdx--;
        int i = inOrStart;
        for(i = inOrStart ; i <= inOrEnd ; i++){
            if(inorder[i] == rootVal) break;
        }

        //NRL(REVERSE POST ORDER)
        TreeNode* root = new TreeNode(rootVal);
        root->right = solve(inorder, postorder, i+1, inOrEnd, poOrIdx);
        root->left = solve(inorder, postorder, inOrStart, i-1 , poOrIdx);

        return root;
    }
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        int n = inorder.size();
        int poOrIdx = n-1;
        return solve(inorder, postorder, 0, n-1 , poOrIdx);     //0 = inOrStart, n-1 = inOrEnd
    }
};
/*
Example 1:
Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
Output: [3,9,20,null,null,15,7]

Example 2:
Input: inorder = [-1], postorder = [-1]
Output: [-1]
*/


//73. VALIDATE BINARY SEARCH TREE                                {T.C = O(N), S.C = O(H)}
class Solution {
public:
    bool isValid(TreeNode* root, long long lowerBound, long long upperBound){
        //base case
        if(!root) return true;

        if(lowerBound >= root->val || root->val >= upperBound) return false;

        return isValid(root->left, lowerBound, root->val) && isValid(root->right, root->val, upperBound);
    }
    bool isValidBST(TreeNode* root) {
        return isValid(root, LLONG_MIN, LLONG_MAX);              //lower and upper bound (int wont work)
    }
};
/*
Example 1:
Input: root = [2,1,3]
Output: true

Example 2:
Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.
*/


//74. KTH SMALLEST ELEMENT IN A BST                              {T.C = O(N), S.C = O(N)}
class Solution {
public:
    void treeToArr(TreeNode* root, vector<int>&vec){
        //base case
        if(!root) return;

        //inorder
        treeToArr(root->left, vec);
        vec.push_back(root->val);
        treeToArr(root->right, vec);
    }
    int kthSmallest(TreeNode* root, int k) {
        vector<int>vec;
        treeToArr(root, vec);

        return vec[k-1];
    }
};
/*
Example 1:
Input: root = [3,1,4,null,2], k = 1
Output: 1

Example 2:
Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3
*/


//75. TOP K FREQUENT ELEMENTS                                    {T.C = O(N), S.C = O(N)}
class Solution {
public:
    //USING MAXHEAP                                      {T.C = O(N*LOGN), S.C = O(N)}
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int,int>mp;
        for(auto it : nums) mp[it]++;

        priority_queue<pair<int,int>>maxHeap;                     //FREQ, ELE (REVERSE){SORT ON BASIS OF FREQ}
        for(auto it : mp) maxHeap.push({it.second, it.first});

        vector<int>ans;
        while(k--){
            ans.push_back(maxHeap.top().second);              //second is element
            maxHeap.pop();
        }
        return ans;
    }
};
/*
Example 1:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:
Input: nums = [1], k = 1
Output: [1]
*/


//76. FIND MEDIAN FROM DATA STREAM                               {T.C = O(N), S.C = O(N)}
class MedianFinder {
public:
    priority_queue<int>leftMaxHeap;
    priority_queue<int,vector<int>, greater<int>>rightMinHeap;
    MedianFinder() {}
    
    void addNum(int num) {
        if(leftMaxHeap.empty() || leftMaxHeap.top() > num) leftMaxHeap.push(num);
        else rightMinHeap.push(num);

        int n = leftMaxHeap.size(), m = rightMinHeap.size();
        if(n < m){
            leftMaxHeap.push(rightMinHeap.top());
            rightMinHeap.pop();
        }else if(abs(n-m) > 1){
            rightMinHeap.push(leftMaxHeap.top());
            leftMaxHeap.pop();
        }
    }
    double findMedian() {
        if(leftMaxHeap.size() == rightMinHeap.size()){
            return (leftMaxHeap.top() + rightMinHeap.top()) / 2.0;
        }
        return leftMaxHeap.top();                             //extra element in maxHeap(odd len)
    }
};
/*
Example 1:
Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]
Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
*/


/*----------------------------------------------THE END----------------------------------------------------------------*/


//75 TOP INTERVIEW QUESTIONS LEETCODE

/*----------------------------------------------ARRAY/STRING------------------------------------------------------------------*/
//77. MERGE STRINGS ALTERNATELY                                                 {T.C = O(N), S.C = O(N)}
/*
simply take 2 pointers initialize to both strings if both smaller then total length ans += strings(1, 2) , if 1 is bigger(exhaust other) 
simply return rest of string.
*/
class Solution {
public:
    string mergeAlternately(string word1, string word2) {
        string ans = "";
        int n = word1.length();
        int m = word2.length();

        int i = 0, j = 0;                        //initial pointers for both strings
        while(i < n && j < m){
            ans += word1[i];
            ans += word2[j];
            i++, j++;
        }
        while(i < n){                           //if string 2 exhausted
            ans += word1[i];
            i++;
        }
        while(j < m){                           //if string 1 exhausted
            ans += word2[j];
            j++;
        }

        return ans;
    }
};
/*
Example 1:
Input: word1 = "abc", word2 = "pqr"
Output: "apbqcr"
Explanation: The merged string will be merged as so:
word1:  a   b   c
word2:    p   q   r
merged: a p b q c r

Example 2:
Input: word1 = "ab", word2 = "pqrs"
Output: "apbqrs"
Explanation: Notice that as word2 is longer, "rs" is appended to the end.
word1:  a   b 
word2:    p   q   r   s
merged: a p b q   r   s

Example 3:
Input: word1 = "abcd", word2 = "pq"
Output: "apbqcd"
Explanation: Notice that as word1 is longer, "cd" is appended to the end.
word1:  a   b   c   d
word2:    p   q 
merged: a p b q c   d
*/


//for remove error(line)
int gcd(int a, int b);
//--------------------
//78. GREATEST COMMON DIVISOR OF STRINGS                                           {T.C = O(N), S.C = O(N)}
/*
Using Commutative property of math (3+6 = 9 == 6+3 = 9) ,it implies that the strings have a common divisor ,then extracts a substring from
str1 starting from index 0 with a length determined by the gcd of the lengths of str1 and str2.
*/
class Solution {
public:
    string gcdOfStrings(string str1, string str2) {
        if(str1+str2 == str2+str1){
            return str1.substr(0, gcd(str1.size(), str2.size()));  //where gcd(3, 6) = 3
        }
        return "";
    }
};
/*
Example 1:
Input: str1 = "ABCABC", str2 = "ABC"
Output: "ABC"

Example 2:
Input: str1 = "ABABAB", str2 = "ABAB"
Output: "AB"

Example 3:
Input: str1 = "LEET", str2 = "CODE"
Output: ""
*/


//79. KIDS WITH THE GREATEST NUMBER OF CANDIES                                 {T.C = O(N), S.C = O(N)}
/*
First find maximum element of vector of candies then iterate again and check if candies[i] + extracandies >= maxi then push true else false
to ans vector.
*/
class Solution {
public:
    vector<bool> kidsWithCandies(vector<int>& candies, int extraCandies) {
        vector<bool>ans;
        int maxi = INT_MIN;
        for(int i = 0 ; i < candies.size() ; i++){
            maxi = max(maxi, candies[i]);
        }
        for(int i = 0 ; i < candies.size() ; i++){
            if(candies[i]+extraCandies >= maxi){
                ans.push_back(true);
            }else{
                ans.push_back(false);
            }
        }
        return ans;
    }
};
/*
Example 1:
Input: candies = [2,3,5,1,3], extraCandies = 3
Output: [true,true,true,false,true] 
Explanation: If you give all extraCandies to:
- Kid 1, they will have 2 + 3 = 5 candies, which is the greatest among the kids.
- Kid 2, they will have 3 + 3 = 6 candies, which is the greatest among the kids.
- Kid 3, they will have 5 + 3 = 8 candies, which is the greatest among the kids.
- Kid 4, they will have 1 + 3 = 4 candies, which is not the greatest among the kids.
- Kid 5, they will have 3 + 3 = 6 candies, which is the greatest among the kids.

Example 2:
Input: candies = [4,2,1,1,2], extraCandies = 1
Output: [true,false,false,false,false] 
Explanation: There is only 1 extra candy.
Kid 1 will always have the greatest number of candies, even if a different kid is given the extra candy.
*/


//80. CAN PLACE FLOWERS                                                       {T.C = O(N), S.C = O(1)}
/*
First check base case if n == 0 then always true, check left and right empty with boundary condition if its empty we can place flower else not.
*/
class Solution {
public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        int l = flowerbed.size();

        if(n == 0){                                  //if flower 0 then always true
            return true;
        }

        for(int i = 0 ; i < l ; i++){
            if(flowerbed[i] == 0){
                bool left = (i == 0) || (flowerbed[i-1] == 0);
                bool right = (i == l-1) || (flowerbed[i+1] == 0);

                if(left && right){
                    flowerbed[i] = 1;                   //place flower and decrease count of flower
                    n--;
                    if(n == 0){
                        return true;
                    }
                }
            }
        }
        return false;
    }
};
/*
Example 1:
Input: flowerbed = [1,0,0,0,1], n = 1
Output: true

Example 2:
Input: flowerbed = [1,0,0,0,1], n = 2
Output: false
*/


//81. REVERSE VOWELS OF A STRING                                              {T.C = O(N), S.C = O(1){constant space(10 char)}}
/*
First store all the vowels in a set then initialize two pointers (start, end) while(i < j) , inner i < j && vowels not find move i and j
if find then swap initial and final characters(vowel).
*/
class Solution {
public:
    string reverseVowels(string s) {
        int i = 0;
        int n = s.size();
        int j = n-1;

        unordered_set<char>st;
        st.insert('a');
        st.insert('e');
        st.insert('i');
        st.insert('o');
        st.insert('u');
        st.insert('A');
        st.insert('E');
        st.insert('I');
        st.insert('O');
        st.insert('U');

        while(i < j){
            while(i < j && st.find(s[i]) == st.end()){
                i++;
            }
            while(i < j && st.find(s[j]) == st.end()){
                j--;
            }
            if(i < j){
                swap(s[i], s[j]);
                i++, j--;
            }
        }
        return s;
    }
};
/*
Example 1:
Input: s = "hello"
Output: "holle"

Example 2:
Input: s = "leetcode"
Output: "leotcede"
*/


//82. REVERSE WORDS IN A STRING
class Solution {
public:
    string reverseWords(string s) {
        int n = s.length();
        string ans;
        int i = 0;
        while(i < n){
            while(i < n && s[i] == ' '){               //continue beginning spaces
                i++;
            }
            if(i >= n){
                break;
            }
            int j = i+1;                              //just next space's word
            while(j < n && s[j] != ' '){              //word encounter
                j++;
            }
            string sub = s.substr(i, j-i);            //find substring = substr(initial, lenght (substring))
            if(ans.empty()){
                ans = sub;
            }else{
                ans = sub + " " + ans;                //reverse push 
            }
            i = j+1;
        }
        return ans;
    }
};
/*
Example 1:
Input: s = "the sky is blue"
Output: "blue is sky the"

Example 2:
Input: s = "  hello world  "
Output: "world hello"
Explanation: Your reversed string should not contain leading or trailing spaces.

Example 3:
Input: s = "a good   example"
Output: "example good a"
Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.
*/


//83. INCREASING TRIPLET SUBSEQUENCE                                 {T.C = O(N), S.C = O(1)}
/*
First check the base case if n < 3 then return false(not possible triplet) initialize firstMin and secondMin and
store the first min and second min element and 3rd element return true , else false.
*/
class Solution {
public:
    bool increasingTriplet(vector<int>& nums) {
        int n = nums.size();
        //base case
        if(n < 3){
            return false;
        }
        int firstMin = INT_MAX;
        int secondMin = INT_MAX;
        for(int i = 0 ; i < n ; i++){
            if(nums[i] <= firstMin){
                firstMin = nums[i];
            }else if(nums[i] <= secondMin){       //check for second min element
                secondMin = nums[i];
            }else{
                return true;                      //third element is greater
            }
        }
        return false;
    }
};
/*
Example 1:
Input: nums = [1,2,3,4,5]
Output: true
Explanation: Any triplet where i < j < k is valid.

Example 2:
Input: nums = [5,4,3,2,1]
Output: false
Explanation: No triplet exists.

Example 3:
Input: nums = [2,1,5,0,4,6]
Output: true
Explanation: The triplet (3, 4, 5) is valid because nums[3] == 0 < nums[4] == 4 < nums[5] == 6.
*/


//84. STRING COMPRESSION                                               {T.C = O(N), S.C = O(1)}
/*
First find the count if char is same to next char, then handle case count > 1 && count > 9 {10 = '1' '0'} put it into string.
*/
class Solution {
public:
    int compress(vector<char>& chars) {
        int n = chars.size();
        int i = 0;
        int index = 0;
        
        while(i < n){
            //find count
            int count = 0;
            char currChar = chars[i];
            while(i < n && chars[i] == currChar ){
                count++;
                i++;
            }

            //now do asssign operation
            chars[index] = currChar;
            index++;
            // chars[index] = count;                 //we cant use directly handle (10 = '1' '0' & count > 1)
            if(count > 1){
                string countStr = to_string(count);          //10 = '1' '0'
                for(char it : countStr){
                    chars[index] = it;
                    index++;
                }
            }
        }
        return index;                               //index points to end of required (compressed)string (length)
    }
};
/*
Example 1:
Input: chars = ["a","a","b","b","c","c","c"]
Output: Return 6, and the first 6 characters of the input array should be: ["a","2","b","2","c","3"]
Explanation: The groups are "aa", "bb", and "ccc". This compresses to "a2b2c3".

Example 2:
Input: chars = ["a"]
Output: Return 1, and the first character of the input array should be: ["a"]
Explanation: The only group is "a", which remains uncompressed since it's a single character.

Example 3:
Input: chars = ["a","b","b","b","b","b","b","b","b","b","b","b","b"]
Output: Return 4, and the first 4 characters of the input array should be: ["a","b","1","2"].
Explanation: The groups are "a" and "bbbbbbbbbbbb". This compresses to "ab12".
*/


/*---------------------------------------- TWO POINTERS --------------------------------------------------------*/
//85. MOVE ZEROES                                                          {T.C = O(N), S.C = O(1)}
/*
Two pointer appraoch 1st pointer use to keep track of non zero element another when is simple iteration , if(nums[j] is not 0) then it will swap
with nums[i] it ensures that nonzero element shift ot front.
*/
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int n = nums.size();
        int i = 0;                                  //placing non zero element to front
        int j = 0;                                  //for iteration
        while(j < n){
            if(nums[j] != 0){                      
                swap(nums[i], nums[j]);            //nonzero element shift to front
                i++;
            }
            j++;
        }
    }
};
/*
Example 1:
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]

Example 2:
Input: nums = [0]
Output: [0]
*/


//86. IS SUBSEQUENCE                                                    {T.C = O(N), S.C = O(1)}
/*
Initialize pointer to both string and iterate if same char then move first and second pointer else move second pointer at last check the
first string is completed or not.
*/
class Solution {
public:
    bool isSubsequence(string s, string t) {
        int n = s.size();
        int m = t.size();
        int i = 0;                      //initial pointer of s
        int j = 0;                      //initial pointer of t
        while(i < n && j < m){
            if(s[i] == t[j]){
                i++;
            }
            j++;
        }
        return i == s.size();           //string s is exhausted or not
    }
};
/*
Example 1:
Input: s = "abc", t = "ahbgdc"
Output: true

Example 2:
Input: s = "axc", t = "ahbgdc"
Output: false
*/


//87. MAX NUMBER OF K-SUM PAIRS                                            {T.C = O(N*LOGN), S.C = O(1)}
/*
First sort the array then use two pointer approach if sum of nums[i]+nums[j] == k then increase count else if less then i++ else j--.
*/
class Solution {
public:
    int maxOperations(vector<int>& nums, int k) {
        int n = nums.size();
        int i = 0; 
        int j = n-1;
        int opCount = 0;

        sort(nums.begin(), nums.end());
        
        while(i < j){
            if(nums[i]+nums[j] == k){
                opCount++;
                i++, j--;
            }else if(nums[i]+nums[j] < k){
                i++;
            }else{
                j--;
            }
        }
        return opCount;
    }
};
/*
Example 1:
Input: nums = [1,2,3,4], k = 5
Output: 2
Explanation: Starting with nums = [1,2,3,4]:
- Remove numbers 1 and 4, then nums = [2,3]
- Remove numbers 2 and 3, then nums = []
There are no more pairs that sum up to 5, hence a total of 2 operations.

Example 2:
Input: nums = [3,1,3,4,3], k = 6
Output: 1
Explanation: Starting with nums = [3,1,3,4,3]:
- Remove the first two 3's, then nums = [1,4,3]
There are no more pairs that sum up to 6, hence a total of 1 operation.
*/


/*-------------------------------------------SLIDING WINDOW ------------------------------------------*/
//88. MAXIMUM AVERAGE SUBARRAY I                                            {T.C = O(N), S.C = O(1)}
/*
First find the window then maxAvg = max(maxAvg, sum/k(avg)) and shrink move window by shrinking from left(sum -= nums[i])
and increasing right side (j++).
*/
class Solution {
public:
    double findMaxAverage(vector<int>& nums, int k) {
        int n = nums.size();
        int i = 0;
        int j = 0;
        double maxiAvg = INT_MIN;
        double sum = 0;
        double avg = 0;
        while(j < n){
            sum += nums[j];
            if(j-i+1 < k){                  //find k size window
                j++;
            }else if(j-i+1 == k){
                maxiAvg = max(maxiAvg, sum/k);
                sum -= nums[i];            //shrink window
                i++, j++;
            }
        }
        return maxiAvg;
    }
};
/*
Example 1:
Input: nums = [1,12,-5,-6,50,3], k = 4
Output: 12.75000
Explanation: Maximum average is (12 - 5 - 6 + 50) / 4 = 51 / 4 = 12.75

Example 2:
Input: nums = [5], k = 1
Output: 5.00000
*/


//89. MAXIMUM NUMBER OF VOWELS IN A SUBSTRING OF GIVEN LENGTH                      {T.C = O(N), S.C = O(1)}
/*
First store the all vowels in map(char, bool) , then first check if char matches to mp's char then count++, then expend the window,
when the window length reach to k then shrink window from left and match the char to mp if same the count--. return maxi(window).
*/
class Solution {
public:
    int maxVowels(string s, int k) {
        int n = s.length();
        //unordered_map<char, bool> isVowel = {{'a', true}, {'e', true}, {'i', true}, {'o', true}, {'u', true}};
        unordered_map<char, bool>mp;
        mp['a'] = true;
        mp['e'] = true;
        mp['i'] = true;
        mp['o'] = true;
        mp['u'] = true;

        int i = 0, j = 0;
        int maxi = 0;
        int count = 0;
        while(j < n){
            if(mp.find(s[j]) != mp.end()){
                if(mp[s[j]]){                         //if char is found(true)
                    count++;
                }
            }
            if(j-i+1 < k){
                j++;
            }else if(j-i+1 == k){
                maxi = max(maxi, count);
                if(mp.find(s[i]) != mp.end()){             //shrinking window
                    if(mp[s[i]]){
                        count--;
                    }
                }
                i++, j++;
            }
        }
        return maxi;
    }
};
/*
Example 1:
Input: s = "abciiidef", k = 3
Output: 3
Explanation: The substring "iii" contains 3 vowel letters.

Example 2:
Input: s = "aeiou", k = 2
Output: 2
Explanation: Any substring of length 2 contains 2 vowels.

Example 3:
Input: s = "leetcode", k = 3
Output: 2
Explanation: "lee", "eet" and "ode" contain 2 vowels.
*/


//90. MAX CONSECUTIVE ONES III                                                  {T.C = O(N), S.C = O(1)}
/*
Using Variable Size Sliding window approach, take variable of zeroCount and store it if it is > K then shrink the window and at last 
return max length.(we can swap only those 0s whose count is less then k only)
*/
class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {
        int n = nums.size();
        int i = 0, j = 0;
        int zeroCount = 0;
        int maxi = 0;

        while (j < n) {
            if (nums[j] == 0) {
                zeroCount++;
            }
            while (zeroCount > k) {         // If zeroCount exceeds k, move the window's starting position (i)
                if (nums[i] == 0) {
                    zeroCount--;
                }
                i++;
            }
            maxi = max(maxi, j - i + 1);   // Update the maximum length of the subarray
            j++;
        }

        return maxi;
    }
};
/*
Example 1:
Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
Explanation: [1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.

Example 2:
Input: nums = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], k = 3
Output: 10
Explanation: [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.
*/


//91. LONGEST SUBARRAY OF 1'S AFTER DELETING ONE ELEMENT                          {T.C = O(N), S.C = O(1)}
/*
Using Variable Size Sliding window, take zero count for first zero occur in subarray zeroCount++, then for next shrink window(j-i) 1 is deleted
always, after that return maxi(length of subarray).
*/
class Solution {
public:
    int longestSubarray(vector<int>& nums) {
        int n = nums.size();
        int i = 0, j = 0;
        int maxi = 0;
        int zeroCount = 0;

        while(j < n){
            if(nums[j] == 0){
                zeroCount++;
            }
            while(zeroCount > 1){
                if(nums[i] == 0){
                    zeroCount--;
                }
                i++;
            }
            maxi = max(maxi, j-i);          //j-i (1 deleted {j-i+1 - 1 = j-i})
            j++;
        }
        return maxi;
    }
};
/*
Example 1:
Input: nums = [1,1,0,1]
Output: 3
Explanation: After deleting the number in position 2, [1,1,1] contains 3 numbers with value of 1's.

Example 2:
Input: nums = [0,1,1,1,0,1,1,0,1]
Output: 5
Explanation: After deleting the number in position 4, [0,1,1,1,1,1,0,1] longest subarray with value of 1's is [1,1,1,1,1].

Example 3:
Input: nums = [1,1,1]
Output: 2
Explanation: You must delete one element.
*/


/*------------------------------------------------------------PREFIX SUM--------------------------------------------------------------*/
//92. FIND THE HIGHEST ALTITUDE                                                  {T.C = O(N), S.C = O(N)}
/*
Take a vector then store prefix sum of there by pushing element and prefix.back() after that return max of the prefix vector
*/
class Solution {
public:
    int largestAltitude(vector<int>& gain) {
        int n = gain.size();
        vector<int>prefixSum(n, 0);
        for(auto it : gain){
            prefixSum.push_back(prefixSum.back() + it);
        }
        return *max_element(prefixSum.begin(), prefixSum.end());
    }
};
/*
Example 1:
Input: gain = [-5,1,5,0,-7]
Output: 1
Explanation: The altitudes are [0,-5,-4,1,1,-6]. The highest is 1.

Example 2:
Input: gain = [-4,-3,-2,-1,4,3,2]
Output: 0
Explanation: The altitudes are [0,-4,-7,-9,-10,-6,-3,-1]. The highest is 0.
*/


//93. FIND PIVOT INDEX                                                {T.C = O(N), S.C = O(1)}
/*
First find the total sum of the array, then iterate again decrease sum = sum-a[i] if it is equal to prefix sum then return the index else
increase the prefix sum else return -1.
*/
class Solution {
public:
    int pivotIndex(vector<int>& nums) {
        int n = nums.size();
        int sum = 0;
        int prefixSum = 0;

        for(int i = 0 ; i < n ; i++){
            sum += nums[i];
        }

        for(int i = 0 ; i < n ; i++){
            sum = sum-nums[i];
            if(sum == prefixSum){                   //left sum == right sum (return index)
                return i;
            }
            prefixSum += nums[i];
        }
        return -1;
    }
};
/*
Example 1:
Input: nums = [1,7,3,6,5,6]
Output: 3
Explanation:
The pivot index is 3.
Left sum = nums[0] + nums[1] + nums[2] = 1 + 7 + 3 = 11
Right sum = nums[4] + nums[5] = 5 + 6 = 11

Example 2:
Input: nums = [1,2,3]
Output: -1
Explanation:
There is no index that satisfies the conditions in the problem statement.

Example 3:
Input: nums = [2,1,-1]
Output: 0
Explanation:
The pivot index is 0.
Left sum = 0 (no elements to the left of index 0)
Right sum = nums[1] + nums[2] = 1 + -1 = 0
*/


/*------------------------------------------------------ HASH MAP / SET ----------------------------------------------------*/
//94. FIND THE DIFFERENCE OF TWO ARRAYS                                   {T.C = O(N), S.C = O(N)}
/*
Take 2 vectors and 2 sets , push given vector's in set then iterate the set and push in vector the non match element same to next set
finally return {ans1, ans2}.
*/
class Solution {
public:
    vector<vector<int>> findDifference(vector<int>& nums1, vector<int>& nums2) {
        vector<int>ans1;
        vector<int>ans2;
        unordered_set<int>st1(nums1.begin(), nums1.end());        //insert vector in a set
        unordered_set<int>st2(nums2.begin(), nums2.end());

        for(auto it : st1){                      //search 1st set's element in 2nd sets then push element and viceversa
            if(st2.count(it) == 0){        //or if (st2.find(it) == st2.end())
                ans1.push_back(it);
            }
        }
        for(auto it : st2){
            if(st1.count(it) == 0){
                ans2.push_back(it);
            }
        }
        
        return {ans1, ans2};
    }
};
/*
Example 1:
Input: nums1 = [1,2,3], nums2 = [2,4,6]
Output: [[1,3],[4,6]]
Explanation:
For nums1, nums1[1] = 2 is present at index 0 of nums2, whereas nums1[0] = 1 and nums1[2] = 3 are not present in nums2. Therefore, answer[0] = [1,3].
For nums2, nums2[0] = 2 is present at index 1 of nums1, whereas nums2[1] = 4 and nums2[2] = 6 are not present in nums2. Therefore, answer[1] = [4,6].

Example 2:
Input: nums1 = [1,2,3,3], nums2 = [1,1,2,2]
Output: [[3],[]]
Explanation:
For nums1, nums1[2] and nums1[3] are not present in nums2. Since nums1[2] == nums1[3], their value is only included once and answer[0] = [3].
Every integer in nums2 is present in nums1. Therefore, answer[1] = [].
*/


//95. UNIQUE NUMBER OF OCCURENCES                                    {T.C = O(N), S.C = O(N)}
/*
Using Map (number, count) & set (unique count) , first store arr in map then , iterate map and check if count is unique
after inserting count in set.
*/
class Solution {
public:
    bool uniqueOccurrences(vector<int>& arr) {
        unordered_map<int ,int>mp;             //store <number, count>
        unordered_set<int>unq;                 //store<unique-count>
        for(auto it : arr){                    //insert arr element in mp
            mp[it]++;
        }
        for(auto it : mp){
            // if(unq.count(it.second) > 0){        //or
            if(unq.find(it.second) != unq.end()){        //check for unique count
                return false;
            }
            unq.insert(it.second);
        }
        return true;
    }
};
/*
Example 1:
Input: arr = [1,2,2,1,1,3]
Output: true
Explanation: The value 1 has 3 occurrences, 2 has 2 and 3 has 1. No two values have the same number of occurrences.

Example 2:
Input: arr = [1,2]
Output: false

Example 3:
Input: arr = [-3,0,1,-3,1,1,1,-3,10,0]
Output: true
*/


//96. DETERMINE IF TWO STRINGS ARE CLOSE                                          {T.C = O(N*LOGN), S.C = O(N)}
/*
Take 2 vectors and 2 map ,store word1 and word2 in a map, now iterate 1st map and check if all char is matched with mp2 then push all mp1 and mp2
char in vectors then sort the vector and check v1 == v2.
*/
class Solution {
public:
    bool closeStrings(string word1, string word2) {
        vector<int>v1,v2;
        unordered_map<char, int>mp1;
        for(auto it : word1){
            mp1[it]++;
        }
        unordered_map<char, int>mp2;
        for(auto it : word2){
            mp2[it]++;
        }

        for(auto it : mp1){
            if(mp2.find(it.first) == mp2.end()){           //mp1 char not found in mp2
                return false;
            }
            v1.push_back(mp1[it.first]);
            v2.push_back(mp2[it.first]);
        }
        sort(v1.begin(), v1.end());
        sort(v2.begin(), v2.end());

        return v1 == v2;
    }
};
/*
Example 1:
Input: word1 = "abc", word2 = "bca"
Output: true
Explanation: You can attain word2 from word1 in 2 operations.
Apply Operation 1: "abc" -> "acb"
Apply Operation 1: "acb" -> "bca"

Example 2:
Input: word1 = "a", word2 = "aa"
Output: false
Explanation: It is impossible to attain word2 from word1, or vice versa, in any number of operations.

Example 3:
Input: word1 = "cabbba", word2 = "abbccc"
Output: true
Explanation: You can attain word2 from word1 in 3 operations.
Apply Operation 1: "cabbba" -> "caabbb"
Apply Operation 2: "caabbb" -> "baaccc"
Apply Operation 2: "baaccc" -> "abbccc"
*/


//97. EQUAL ROW AND COLUMN PAIRS                                             {T.C = O(N^2), S.C = O(N)}
/*
First convert row vector element into a string then , mp<string, count> , now initialize count = 0 check particular col with matching row
then increase count.
*/
class Solution {
public: 
    string mpStr(vector<int>&n){               //convert vector of number to string
        string str = "";
        for(auto it : n){
            str += to_string(it);
            str += "#";                        //for creating uniquness of particular string
        }
        return str;
    }
    int equalPairs(vector<vector<int>>& grid) {
        int n = grid.size();
        unordered_map<string, int>mp;

        for(auto it : grid){                   //inserting <string(row), count>
            mp[mpStr(it)]++;
        }

        int count = 0;
        for(int j = 0 ; j < n ; j++){                   //now check for particluar col with row
            string finder = "";
            for(int i = 0 ; i < n ; i++){
                finder += to_string(grid[i][j]);
                finder += "#";
            }        
            count += mp[finder];
        }
        return count;               
    }
};
/*
Example 1:
Input: grid = [[3,2,1],[1,7,6],[2,7,7]]
Output: 1
Explanation: There is 1 equal row and column pair:
- (Row 2, Column 1): [2,7,7]

Example 2:
Input: grid = [[3,1,2,2],[1,4,4,5],[2,4,2,2],[2,4,2,2]]
Output: 3
Explanation: There are 3 equal row and column pairs:
- (Row 0, Column 0): [3,1,2,2]
- (Row 2, Column 2): [2,4,2,2]
- (Row 3, Column 2): [2,4,2,2]
*/


/*---------------------------------------------------------- STACK ----------------------------------------------*/
//98. REMOVING STARS FROM A STRING                                     {T.C = O(N), S.C = O(N)}
/*
Take a stack and push the element != * otherwise pop (!stk.empty) after that pop element from stack and push to string after that reverse the
string(stack is LIFO) then finally return ans string.
*/
class Solution {
public:
    string removeStars(string s) {
        string ans;
        stack<char>stk;

        for(int i = 0 ; i < s.length() ; i++){
            if(s[i] == '*' && !stk.empty()){
                stk.pop();
            }else{
                stk.push(s[i]);
            }
        }
        while(!stk.empty()){
            ans.push_back(stk.top());
            stk.pop();
        }
        reverse(ans.begin(), ans.end());           //Stack Using LIFO (reverse printing)
        return ans;
    }
};
/*
Example 1:
Input: s = "leet**cod*e"
Output: "lecoe"
Explanation: Performing the removals from left to right:
- The closest character to the 1st star is 't' in "leet**cod*e". s becomes "lee*cod*e".
- The closest character to the 2nd star is 'e' in "lee*cod*e". s becomes "lecod*e".
- The closest character to the 3rd star is 'd' in "lecod*e". s becomes "lecoe".
There are no more stars, so we return "lecoe".

Example 2:
Input: s = "erase*****"
Output: ""
Explanation: The entire string is removed, so we return an empty string.
*/


//99. ASTEROID COLLISION                                           {T.C = O(N), S.C = O(N)}
/*
Check the condition (collision case) if stack not empty, it < 0 and st.top > 0 find sum of each iteration, check if 
sum > 0{Stack's asteroid is big, reinitialse it = 0 }, sum < 0 {incoming asteroid is big(destroys other) pop up the elemnt}
sum == 0 both pop element and reinitialize it , and push element into stack after that return stack's popped elemnt in
vector and reverse answer printed.
*/
class Solution {
public:
    vector<int> asteroidCollision(vector<int>& asteroids) {
        vector<int>ans;
        stack<int>stk;

        for(auto it : asteroids){
            while(!stk.empty() && it < 0 && stk.top() > 0){       //collision case (only one case -><-)
                int sum = it + stk.top();
                if(sum < 0){               //incoming asteroid is big
                    stk.pop();
                }else if(sum > 0){         //stack's asteroid is big
                    it = 0;
                }else{ //sum == 0
                    stk.pop();
                    it = 0;
                }
            }
            if(it != 0){
                stk.push(it);
            }
        }

        while(!stk.empty()){
            ans.push_back(stk.top());
            stk.pop();
        }
        reverse (ans.begin(), ans.end());

        return ans;
    }
};
/*
Example 1:
Input: asteroids = [5,10,-5]
Output: [5,10]
Explanation: The 10 and -5 collide resulting in 10. The 5 and 10 never collide.

Example 2:
Input: asteroids = [8,-8]
Output: []
Explanation: The 8 and -8 collide exploding each other.

Example 3:
Input: asteroids = [10,2,-5]
Output: [10]
Explanation: The 2 and -5 collide resulting in -5. The 10 and -5 collide resulting in 10.
*/


//100. DECODE STRINGS                                                 {T.C = O(N), S.C = O(N)}
/*
Use two stacks(string, int), 2 var(string, int) iterate string , handle cases opening bracket= push curr string and 
curr number in respective stacks, closing brackets = pops the previous string and number from their respective stacks
isdigit() = currNum = currNum * 10 + (it - '0'), else add rest of char and return currStr.
*/
class Solution {
public:
    string decodeString(string s) {
        string currStr="";
        int currNum = 0;
        stack<string>strStk;
        stack<int>numStk;

        for(auto it : s){
            if(it == '['){
                strStk.push(currStr);
                numStk.push(currNum);
                currStr = "";              //reset after pushing
                currNum = 0;               //reset after pushing
            }else if(it == ']'){           //pops the previous string and number from their respective stacks
                int num = numStk.top();    
                numStk.pop();
                string outPut = "";
                while(num--){
                    outPut += currStr;
                }

                string prevStr = strStk.top();
                strStk.pop();
                currStr = prevStr + outPut;
            }else if(isdigit(it)){
                currNum = currNum * 10 + (it - '0');
            }else{
                currStr += it;
            }
        }
        return currStr;
    }
};
/*
Example 1:
Input: s = "3[a]2[bc]"
Output: "aaabcbc"

Example 2:
Input: s = "3[a2[c]]"
Output: "accaccacc"

Example 3:
Input: s = "2[abc]3[cd]ef"
Output: "abcabccdcdcdef"
*/


/*--------------------------------------------------- QUEUE ----------------------------------------------------*/
//101. NUMBER OF RECENT CALLS                                             {T.C = O(N), S.C = O(1)}
/*
Take an queue, first find the margin(t-3000) push the element(t) in queue, pop the element from queue until top element lesser then margin
after that return size of queue.
*/
class RecentCounter {
public:
    queue<int>q;
    RecentCounter() {}

    int ping(int t) {
        int margin = t - 3000;
        q.push(t);
        while(q.front() < margin){
            q.pop();
        }
        return q.size();
    }
};
/*
Example 1:

Input
["RecentCounter", "ping", "ping", "ping", "ping"]
[[], [1], [100], [3001], [3002]]
Output
[null, 1, 2, 3, 3]

Explanation
RecentCounter recentCounter = new RecentCounter();
recentCounter.ping(1);     // requests = [1], range is [-2999,1], return 1
recentCounter.ping(100);   // requests = [1, 100], range is [-2900,100], return 2
recentCounter.ping(3001);  // requests = [1, 100, 3001], range is [1,3001], return 3
recentCounter.ping(3002);  // requests = [1, 100, 3001, 3002], range is [2,3002], return 3
*/


//102. DOTA2 SENAT
//BRUTE FORCE                                                         {T.C = O(N^2), S.C = O(1)}
/*
First store count of R and D, initialize idx = 0, if (R) remove nearest D vice versa,
idx-- when we deleted someone at index < idx.
idx+1 % (length of updated strings) for round.
*/
class Solution {
public:
    bool removeSenator(string &senate, char ch, int i){
        bool checkRemoveLeftSide = false;

        while(true){
            if(i == 0){              //1 round completed
                checkRemoveLeftSide = true;
            }
            if(senate[i] == ch){
                senate.erase(senate.begin() + i);  //shift
                break;
            }
            (i+1) % senate.length();
        }
        return checkRemoveLeftSide;
    }
    string predictPartyVictory(string senate) {
        int rCount = count(senate.begin(), senate.end(), 'R');
        int dCount = senate.length()-rCount;

        int idx = 0;
        while(rCount > 0 && dCount > 0){
            if(senate[idx] == 'R'){
                bool checkRemoveLeftSide = removeSenator(senate, 'D',(idx+1) % senate.length());
                dCount--;
                if(checkRemoveLeftSide){
                    idx--;
                }
            }else{ //(senate[idx] == 'D')
                bool checkRemoveLeftSide = removeSenator(senate, 'R', (idx+1) % senate.length());
                rCount--;
                if(checkRemoveLeftSide){
                    idx--;
                }
            }
            idx = (idx+1) % senate.length();
        }

        return rCount == 0 ? "Dire" : "Radiant";
    }
};

//OPTIMIZED APPROACH                                                       {T.C = O(N), S.C = O(N)}
/*
Similar to above just not shifting element,by taking extra space of vector <bool>removed.
*/
class Solution {
public:
    void removeSenator(string &senate, char ch, int i, vector<bool>&removed){
        while(true){
            if(senate[i] == ch && removed[i] == false){
                removed[i] = true;
                break;
            }
            i = (i+1) % senate.length();
        }
    }
    string predictPartyVictory(string senate) {
        int rCount = count(senate.begin(), senate.end(), 'R');
        int dCount = senate.length()-rCount;
        int n = senate.length();
        vector<bool>removed(n, false);

        int idx = 0;
        while(rCount > 0 && dCount > 0){
            if(removed[idx] == false){
                if(senate[idx] == 'R'){
                    removeSenator(senate, 'D',(idx+1) % senate.length(), removed);
                    dCount--;
                }else{ //(senate[idx] == 'D')
                    removeSenator(senate, 'R', (idx+1) % senate.length(), removed);
                    rCount--;
                }
            }
            idx = (idx+1) % senate.length();
        }

        return rCount == 0 ? "Dire" : "Radiant";
    }
};
/*
Example 1:
Input: senate = "RD"
Output: "Radiant"
Explanation: 
The first senator comes from Radiant and he can just ban the next senator's right in round 1. 
And the second senator can't exercise any rights anymore since his right has been banned. 
And in round 2, the first senator can just announce the victory since he is the only guy in the senate who can vote.

Example 2:
Input: senate = "RDD"
Output: "Dire"
Explanation: 
The first senator comes from Radiant and he can just ban the next senator's right in round 1. 
And the second senator can't exercise any rights anymore since his right has been banned. 
And the third senator comes from Dire and he can ban the first senator's right in round 1. 
And in round 2, the third senator can just announce the victory since he is the only guy in the senate who can vote.
*/


/*----------------------------------------------------------------- LINKED LIST --------------------------------------------*/
// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};
//103. DELETE THE MIDDLE NODE OF A LINKED LIST                                  {T.C = O(N), S.C = O(1)}
/*
Using slow and fast pointer approach first find the middle element then remove middle element(temp->next = slow->next {slow (middle element skipped)})
after that delete slow and return head;
*/
class Solution {
public:
    ListNode* deleteMiddle(ListNode* head) {
        if(!head || !head->next){
            return NULL;
        }
        ListNode* slow = head;
        ListNode* fast = head;
        ListNode* temp = NULL;

        while(fast && fast->next){                     //finds middle element
            temp = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        temp->next = slow->next;                       //middle element removed
        delete slow;
        return head;
    }
};
/*
Example 1:
Input: head = [1,3,4,7,1,2,6]
Output: [1,3,4,1,2,6]
Explanation:
The above figure represents the given linked list. The indices of the nodes are written below.
Since n = 7, node 3 with value 7 is the middle node, which is marked in red.
We return the new list after removing this node. 

Example 2:
Input: head = [1,2,3,4]
Output: [1,2,4]
Explanation:
The above figure represents the given linked list.
For n = 4, node 2 with value 3 is the middle node, which is marked in red.

Example 3:
Input: head = [2,1]
Output: [2]
Explanation:
The above figure represents the given linked list.
For n = 2, node 1 with value 1 is the middle node, which is marked in red.
Node 0 with value 2 is the only node remaining after removing node 1.
*/


//104. ODD EVEN LINKED LIST                                                  {T.C = O(N), S.C = O(1)}
/*
Using two pointers one from head and another from head->next now traverse ll while(even && even->next) increase both pointers after that
connect odd->next = evenhead (connect odd and even ll).
*/
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        //base case
        if(!head || !head->next){
            return head;
        }

        ListNode* odd = head;
        ListNode* even = head->next;
        ListNode* temp = even;                 //store head of even ll


        while(even && even->next){             //check second or next element 
            odd->next = even->next;
            odd = odd->next;
            even->next = odd->next;
            even = even->next;
        }
        odd->next = temp;                     //connect odd->even list

        return head;
    }
};
/*
Example 1:
Input: head = [1,2,3,4,5]
Output: [1,3,5,2,4]

Example 2:
Input: head = [2,1,3,5,6,4,7]
Output: [2,3,6,7,1,5,4]
*/


//105. MAXIMUM TWIN SUM OF A LINKED LIST                                           {T.C = O(N), S.C = O(N)}
// APPROACH 1(By converting LL into vector)
/*
First Store all ll element into a vector then initialize 2 pointers i = 0 and j = n-1 , take twinSum = arr[i]+arr[j] and update max 
after that retun max.
*/
class Solution {
public:
    /*
    ListNode* ArrToLL(vector<int>&arr){
        int n = arr.size();
        //base case
        if(n == 0){
            return NULL;
        }

        ListNode* head = new ListNode(arr[0]);
        ListNode* curr = head;
        for(int i = 1 ; i < n ; i++){
            curr->next = new ListNode(arr[i]);
            curr = curr->next;
        }
        return head;
    }
    */
    void llToArr(ListNode* head, vector<int>&arr){
        ListNode* temp = head;
        while(temp){
            arr.push_back(temp->val);
            temp = temp->next;
        }
    }
    int pairSum(ListNode* head) {
        vector<int>arr;
        llToArr(head, arr);
        int twinSum = 0;
        int n = arr.size();
        int maxi = INT_MIN;
        int i = 0 , j = n-1;
        while(i < j){
            twinSum = arr[i] + arr[j];
            maxi = max(maxi, twinSum);
            i++;
            j--;
        }
        return maxi;
    }
};

// APPROACH 2 (USING STACK)                                                 {T.C = O(N), S.C = O(N)}
/*
Take Stack put ll element into stack, then traverse ll till n/2 and add st.top().
*/
class Solution {
public:
    int pairSum(ListNode* head) {
        stack<int>stk;
        ListNode* temp = head;
        while(temp){
            stk.push(temp->val);
            temp = temp->next;
        }

        temp = head;                    //reinitialize temp with head
        int n = stk.size();
        int twinSum = 0;
        int maxi = INT_MIN;
        int i = 0;
        while(i < n/2 && temp){
            twinSum = temp->val + stk.top();
            maxi = max(maxi, twinSum);
            stk.pop();
            temp = temp->next;
            i++;
        }
        return maxi;
    }
};

// APPROCH 3 (USING LL{reverse 2nd half(mid)})                              {T.C = O(N), S.C = O(1)}
/*
Finds the mid node, reverse after the mid node then take two pointers on start and mid and add
the two pointers return max of it.
*/
class Solution {
public:
    int pairSum(ListNode* head) {
        //Finding mid
        ListNode* slow = head;
        ListNode* fast = head;
        ListNode* temp = NULL;

        while(fast && fast->next){
            temp = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        ListNode* mid = slow;


        //Reverse 2nd half of ll

        ListNode* prev = NULL;
        // ListNode* mid = slow;
        ListNode* next = NULL;
        
        while(mid){
            next = mid->next;
            mid->next = prev;
            prev = mid;
            mid = next;
        }

        //Find max result 
        temp = head;                    //reinitialize temp with head
        int twinSum = 0;
        int maxi = INT_MIN;
        int i = 0;
        while(prev){                                 //2nd half start from prev
            twinSum = temp->val + prev->val;
            maxi = max(maxi, twinSum);
            temp = temp->next;
            prev = prev->next;
        }
        return maxi;
        

    }
};
/*
Example 1:
Input: head = [5,4,2,1]
Output: 6
Explanation:
Nodes 0 and 1 are the twins of nodes 3 and 2, respectively. All have twin sum = 6.
There are no other nodes with twins in the linked list.
Thus, the maximum twin sum of the linked list is 6. 

Example 2:
Input: head = [4,2,2,3]
Output: 7
Explanation:
The nodes with twins present in this linked list are:
- Node 0 is the twin of node 3 having a twin sum of 4 + 3 = 7.
- Node 1 is the twin of node 2 having a twin sum of 2 + 2 = 4.
Thus, the maximum twin sum of the linked list is max(7, 4) = 7. 

Example 3:
Input: head = [1,100000]
Output: 100001
Explanation:
There is only one node with a twin in the linked list having twin sum of 1 + 100000 = 100001.
*/


/*-------------------------------------------------------- BINARY TREE - DFS ------------------------------------------------------*/
//106. LEAF SIMILAR TREES                                                    {T.C = O(N), S.C = O(N)}
/*
Take 2 vectors for storing leaf nodes, then call inorder , search for leaf node and push in vector then make recursive call of left and right
if v1 == v2 ? true : false.
*/
class Solution {
public:
    void inOrder(TreeNode* root, vector<int>&v){
        //base case
        if(!root){
            return;
        }

        //NLR
        if(!root->left && !root->right){
            v.push_back(root->val);
        }
        inOrder(root->left, v);
        inOrder(root->right, v);
    }
    bool leafSimilar(TreeNode* root1, TreeNode* root2) {
        vector<int>v1, v2;
        inOrder(root1, v1);
        inOrder(root2, v2);

        return v1 == v2 ? true : false;
    }
};
/*
Example 1:
Input: root1 = [3,5,1,6,2,9,8,null,null,7,4], root2 = [3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]
Output: true

Example 2:
Input: root1 = [1,2,3], root2 = [1,3,2]
Output: false
*/


//107. COUNT GOOD NODES IN BINARY TREE                                {T.C = O(N), S.C = O(H)}
/*
Take count variable, make dfs calls(root, root->val, count) , if(root->val > maxVal)  update maxValue and make left
and right calls. after that return count.
*/
class Solution {
public:
    void dfs(TreeNode* root, int maxValue , int &count){
        //base case
        if(!root){
            return;
        }
        if(root->val >= maxValue){
            count++;
            maxValue = root->val;
        }
        dfs(root->left, maxValue, count);
        dfs(root->right, maxValue, count);
    }
    int goodNodes(TreeNode* root) {
        int count = 0;
        dfs(root, root->val, count);                    //root->val = max(current)
        return count;
    }
};
/*
Example 1:
Input: root = [3,1,4,3,null,1,5]
Output: 4
Explanation: Nodes in blue are good.
Root Node (3) is always a good node.
Node 4 -> (3,4) is the maximum value in the path starting from the root.
Node 5 -> (3,4,5) is the maximum value in the path
Node 3 -> (3,1,3) is the maximum value in the path.

Example 2:
Input: root = [3,3,null,4,2]
Output: 3
Explanation: Node 2 -> (3, 3, 2) is not good, because "3" is higher than it.

Example 3:
Input: root = [1]
Output: 1
Explanation: Root is considered as good.
*/


//108. PATH SUM III                                          {T.C = O(N^2), S.C = O(1)}
/*
First we have to find the paths from Node , then make recursive call for left and right subtree, for finding count of 
paths we have another function in which if root->val == target sum then count path++ and make recursive call for 
left and right subtree by targetSum-root->val.
*/
class Solution {
public:
    int countPaths(TreeNode* root, long long targetSum){
        int count = 0;
        //base case
        if(!root){
            return 0;
        }

        if(root->val == targetSum){
            count++;
        }
        //recursive call(backtracking)
        count += countPaths(root->left, targetSum - root->val);  //if there's a path with the remaining target sum starting from the left and right subtrees
        count += countPaths(root->right, targetSum - root->val);

        return count;
    }
    int pathSum(TreeNode* root, int targetSum) {
        //base case
        if(!root){
            return 0;
        }

        int pathSumFromNode = countPaths(root, targetSum);

        //left and right recursive call
        int pathSumFromLeft = pathSum(root->left, targetSum);
        int pathSumFromRight = pathSum(root->right, targetSum);

        return pathSumFromNode + pathSumFromLeft + pathSumFromRight;
    }
};
/*
Example 1:
Input: root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
Output: 3
Explanation: The paths that sum to 8 are shown.

Example 2:
Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: 3
*/


//109. LONGEST ZIGZAG PATH IN A BINARY TREE                           {T.C = O(N), S.C = O(H)}
/*
we have to check both the left and rigth subtree for that make two fun call each, then in func update maxi = max(maxi, steps)
if(left) check for 2 condition if left to right (steps+1) or left to left(steps == 1) simialar for right subtree.
*/
class Solution {
public:
    void maxZigZag(TreeNode* root, bool goLeft, int steps, int &maxi){
        //base case
        if(!root){
            return;
        }
        maxi = max(maxi, steps);

        if(goLeft){
            maxZigZag(root->left, false, steps+1, maxi);   //if go left then next step is right with steps++.
            maxZigZag(root->right, true, 1, maxi);         //if go left and still go left then path reinitialize( steps == 1)
        }else{  //(goRight)
            maxZigZag(root->right, true, steps+1, maxi);   //if go right then next step is left(true) with steps++.
            maxZigZag(root->left, false, 1, maxi);         //if go right and still go rigth(false) then path reinitialize(steps == 1).
        }
    }
    int longestZigZag(TreeNode* root) {
        int maxi = 0;
        maxZigZag(root, true, 0, maxi);          //0 = steps    , goLeft (first check for left subtree)
        maxZigZag(root, false, 0, maxi);          //0 = steps   , !goLeft(first check for right subtree)

        return maxi;
    }
};
/*
Example 1:
Input: root = [1,null,1,1,1,null,null,1,1,null,1,null,null,null,1]
Output: 3
Explanation: Longest ZigZag path in blue nodes (right -> left -> right).

Example 2:
Input: root = [1,1,1,null,1,null,null,1,1,null,1]
Output: 4
Explanation: Longest ZigZag path in blue nodes (left -> right -> left -> right).

Example 3:
Input: root = [1]
Output: 0
*/


/*--------------------------------------------------------------- BINARY TREE BFS -----------------------------------------------*/
//110. BINARY TREE RIGHT SIDE VIEW                                          {T.C = O(N), S.C = O(N)}
/*
Take vector and call solve function in which NRL(Right view) {NLR(Left View)} if(level == ans.size(){size reached change level}), push root->val
in vector and level++.
*/
class Solution {
public:
    void solve(TreeNode* root, vector<int>&ans, int level){
        //base case
        if(!root){
            return;
        }

        //NRL        (for right view)  //for left view (NLR)
        if(level == ans.size()){                   //if level reach ends then change the level and push root->data in vector
            ans.push_back(root->val);           
        }
        solve(root->right, ans, level+1);
        solve(root->left, ans, level+1);
    }
    vector<int> rightSideView(TreeNode* root) {
        vector<int>ans;
        solve(root, ans, 0);                      //0 == initial level
        return ans;
    }
};

//another way using BFS (level order traversal)
class Solution {
public:
    void lvlOrder(TreeNode* root, vector<vector<int>>&ans){
        //base case
        if(!root) return;

        queue<TreeNode*>q;
        q.push(root);
        while(!q.empty()){
            int sz = q.size();
            vector<int>temp;
            while(sz--){
                auto frontNode = q.front();
                q.pop();
                temp.push_back(frontNode->val);
                if(frontNode->left) q.push(frontNode->left);
                if(frontNode->right)q.push(frontNode->right);
            }
            ans.push_back(temp);
        }
    }
    vector<int> rightSideView(TreeNode* root) {
        vector<vector<int>>ans;
        lvlOrder(root, ans);
        vector<int>res;
        for(auto it : ans){
            res.push_back(it.back());
        }
        return res;
    }
};
/*
Example 1:
Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]

Example 2:
Input: root = [1,null,3]
Output: [1,3]

Example 3:
Input: root = []
Output: []
*/


//110. MAXIMUM LEVEL SUM OF A BINARY TREE                              {T.C = O(N), S.C = O(N)}
/*
Take resultLvl, currLvl, maxiSum and queue(bfs) push root in queue, then bfs(while(!q.empty())) traverse in queue
and update sum also check for left and right subtree after that currLvl++ and update result lvl.
*/
class Solution {
public:
    int maxLevelSum(TreeNode* root) {
        int resultLvl = 0;
        int currLvl = 1;
        int maxiSum = INT_MIN;
        queue<TreeNode*>q;                             //bfs
        q.push(root);
        
        while(!q.empty()){
            int n = q.size();
            int sum = 0;

            while(n--){                              //traverse in queue(stores level's nodes)
                TreeNode* frontNode = q.front();
                q.pop();
                sum += frontNode->val;

                if(frontNode->left){
                    q.push(frontNode->left);
                }
                if(frontNode->right){
                    q.push(frontNode->right);
                }
            }
            if(sum > maxiSum){
                maxiSum = sum;
                resultLvl = currLvl;
            }
       
            currLvl++;
        }
        return resultLvl;
    }
};
/*
Example 1:
Input: root = [1,7,0,7,-8,null,null]
Output: 2
Explanation: 
Level 1 sum = 1.
Level 2 sum = 7 + 0 = 7.
Level 3 sum = 7 + -8 = -1.
So we return the level with the maximum sum which is level 2.

Example 2:
Input: root = [989,null,10250,98693,-89388,null,null,null,-32127]
Output: 2
*/


/*-----------------------------------------------------------BINARY SEARCH TREE(BST) -----------------------------------------------*/
//111. SEARCH IN A BINARY SEARCH TREE                                      {T.C = O(N), S.C = O(H)}
/*
Simply check base case if root not exist return null if (root->val == val) return root(return corresponding tree).
if(root->val > val) recursion right else left.
*/
class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
        if(!root){
            return NULL;
        }
        if(root->val == val){
            return root;
        }

        if(root->val > val){
            return searchBST(root->left, val);
        }else{
            return searchBST(root->right, val);
        }
    }
};
/*
Example 1:
Input: root = [4,2,7,1,3], val = 2
Output: [2,1,3]

Example 2:
Input: root 
Output : []
*/


//112. DELETE NODE IN A BST                                            {T.C = O(N), S.C = O(H)}
/*
Before deleting the node we have to chech number or child does deleting node have, if zero child simply delete and return 
NULL, if 1 child (left or right) return non deleted (root->left) or root->right, if 2 child then we have to check minimum
value (which find in right side of BST), if(root->val > key) left call else right call.
*/
class Solution {
public:
    TreeNode* minValue(TreeNode* root){
        TreeNode* temp = root;
        while(temp->left){
            temp = temp->left;
        }
        return temp;
    }
    TreeNode* deleteNode(TreeNode* root, int key) {
        //base case
        if(!root){
            return NULL;
        }

        if(root->val == key){
            //0 child /leaf node delete
            if(!root->left && !root->right){
                delete root;
                return NULL;
            }

            //1 child either left or right node delete
            if(root->left && !root->right){
                TreeNode* temp = root->left;
                delete root;
                return temp;
            }
            if(!root->left && root->right){
                TreeNode* temp = root->right;
                delete root;
                return temp;
            }

            //2 child node delete
            TreeNode* temp = minValue(root->right);
            root->val = temp->val;
            root->right = deleteNode(root->right, temp->val);
            return root;
        }
        else if(root->val > key){           //val present in left subtree
            root->left = deleteNode(root->left, key);
            return root;
        }else{ //root->val < key            //val present in right subtree
            root->right = deleteNode(root->right,  key);
            return root;
        }
        return root;
    }
};
/*
Example 1:
Input: root = [5,3,6,2,4,null,7], key = 3
Output: [5,4,6,2,null,null,7]
Explanation: Given key to delete is 3. So we find the node with value 3 and delete it.
One valid answer is [5,4,6,2,null,null,7], shown in the above BST.
Please notice that another valid answer is [5,2,6,null,4,null,7] and it's also accepted.

Example 2:
Input: root = [5,3,6,2,4,null,7], key = 0
Output: [5,3,6,2,4,null,7]
Explanation: The tree does not contain a node with value = 0.

Example 3:
Input: root = [], key = 0
Output: []
*/


/*------------------------------------------------------- GRAPHS GENERAL ------------------------------------------------*/
//GRAPH
//MINIMUM DISTANCE FINDING ALGORITHM
/*
1. DIJKSTRA'S ALGO                  {not work for -ve weighted graph}
2. BELLMANFORD ALGO                 {work for -ve weighted graph}
3. FLOYD WARSHALL ALGO              {brute force algo}
*/
//MINIMUM SPANNING TREE ALGORITHM
/*
1. KRUSKAL'S ALGO          {intermediate result may be or may be not connected}
2. PRIM'S ALGO             {intermediate result always connected} 
*/

//113. BFS OF GRAPH                                                      {T.C = O(N+M/ V+E), S.C = O(N+M / V+E)}   //N = nodes, E = edges
/*
We required 3 DS (ans, vis(bool), queue) and also initialize nodeindex = 0, push node in queue and mark vis[node] = 1 and push node in ans
now traverse queue, take out frontNode and push in ans, after that traverse adj[frontNode] , if (!vis[it]) then push it and mark vis[it]=1.
*/
class Solution {
  public:
    void BFS(vector<int>adj[], vector<int>&ans, vector<bool>&vis, int node){
        queue<int>q;
        q.push(node);
        vis[node] = 1;
        while(!q.empty()){
            auto frontNode = q.front();
            q.pop();
            ans.push_back(frontNode);
            for(auto it : adj[frontNode]){
                if(!vis[it]){
                    q.push(it);
                    vis[it] = 1;
                }
            }
        }
    }
    vector<int> bfsOfGraph(int V, vector<int> adj[]) {
        vector<int>ans;
        vector<bool>vis(V, 0);
        BFS(adj, ans, vis, 0);                        //node index = 0
        return ans;
    }
};
/*
Example 1:
Input:
V = 5, E = 4
adj = {{1,2,3},{},{4},{},{}}
Output: 
0 1 2 3 4
Explanation: 
0 is connected to 1 , 2 , 3.
2 is connected to 4.
so starting from 0, it will go to 1 then 2
then 3. After this 2 to 4, thus bfs will be
0 1 2 3 4.

Example 2:
Input:
V = 3, E = 2
adj = {{1,2},{},{}}
Output: 
0 1 2
Explanation:
0 is connected to 1 , 2.
so starting from 0, it will go to 1 then 2,
thus bfs will be 0 1 2. 
*/


//114. DFS OF GRAPH                                                        {T.C = O(N+M / V+E), S.C = O(N+M / V+E)}
/*
We required 2 DS (ans, vis(bool)) and also initialize nodeindex = 0,  mark vis[node] = 1 and push node in ans,  now traverse adj[node],
if (!vis[it]) then push it and mark vis[it]=1 and make recursive call to DFS.
*/
class Solution {
  public:
    void DFS(vector<int>adj[], vector<int>&ans, vector<bool>&vis, int node){
        vis[node] = 1;
        ans.push_back(node);
        for(auto it : adj[node]){
            if(!vis[it]){
                DFS(adj, ans, vis, it);
            }
        }
    }
    vector<int> dfsOfGraph(int V, vector<int> adj[]) {
        vector<int>ans;
        vector<bool>vis(V, 0);
        DFS(adj, ans, vis, 0);                       //0 = nodeindex
        return ans;

        /*
        //handle disconnected components
        for(int i = 0 ; i < V ; i++){
            dfs(0, adj, ans, vis);                                  //0 = starting index or node
        }
        */
    }
};
/*
Example 1:
Input: V = 5 , adj = [[2,3,1] , [0], [0,4], [0], [2]]
Output: 0 2 4 3 1
Explanation: 
0 is connected to 2, 3, 1.
1 is connected to 0.
2 is connected to 0 and 4.
3 is connected to 0.
4 is connected to 2.
so starting from 0, it will go to 2 then 4,
and then 3 and 1.
Thus dfs will be 0 2 4 3 1.

Example 2:
Input: V = 4, adj = [[1,3], [2,0], [1], [0]]
Output: 0 1 2 3
Explanation:
0 is connected to 1 , 3.
1 is connected to 0, 2. 
2 is connected to 1.
3 is connected to 0. 
so starting from 0, it will go to 1 then 2
then back to 0 then 0 to 3
thus dfs will be 0 1 2 3. 
*/


/*----------------------------------------------------- GRAPHS DFS ------------------------------------------------*/
//115. KEYS AND ROOMS                                             {T.C = O(N+M), S.C = O(N+M)}
/*
Simple dfs , take bool vector and call dfs if after dfs any room is left or(vis == false) then return false else true.
*/
class Solution {
public:
    void dfs(vector<vector<int>>&rooms, vector<bool>&vis, int node){
        vis[node] = true;
        for(auto it : rooms[node]){
            if(!vis[it]){
                dfs(rooms, vis, it);
            }
        }
    }
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        int n = rooms.size();
        vector<bool>vis(n, false);
        dfs(rooms, vis, 0);                      //0 = nodeindex
        for(auto it : vis){                      //if there is single false then ans not possible
            if(it ==  false){
                return false;
            }
        }
        return true;
    }
};
/*
Example 1:
Input: rooms = [[1],[2],[3],[]]
Output: true
Explanation: 
We visit room 0 and pick up key 1.
We then visit room 1 and pick up key 2.
We then visit room 2 and pick up key 3.
We then visit room 3.
Since we were able to visit every room, we return true.

Example 2:
Input: rooms = [[1,3],[3,0,1],[2],[0]]
Output: false
Explanation: We can not enter room number 2 since the only key that unlocks it is in that room.
*/


//116. NUMBER OF PROVINCES                                       {T.C = O(N^2), S.C = O(N+M)}
/*
Actually We are finding the components(islands) in this , we use simple dfs and take for loop for handling disconnected
components and in dfs Call we checks for index(for(int i = 0 ; i < n ; i++)) not for value(auto it : isConnected[node])
*/
class Solution {
public:
    void dfs(vector<vector<int>>&isConnected, vector<bool>&vis, int node){
        int n = isConnected.size();
        vis[node] = 1;
        for(int i = 0 ; i < n ; i++){                           //checks for index not value(auto it : isConnected[node])
            if(!vis[i] && isConnected[node][i] == 1){           //checks there is direct connnection.
                dfs(isConnected, vis, i);
            }
        }
    }
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();
        int uniqueNodes = 0;
        vector<bool>vis(n, 0);
        for(int i = 0 ; i < n ; i++){                       //for disconnected components
            if(!vis[i]){
                dfs(isConnected, vis, i);                           //0 = nodeIndex
                uniqueNodes++;
            }
        }
        return uniqueNodes;
    }
};
/*
Example 1:
Input: isConnected = [[1,1,0],[1,1,0],[0,0,1]]
Output: 2

Example 2:
Input: isConnected = [[1,0,0],[0,1,0],[0,0,1]]
Output: 3
*/


//117. REORDER ROUTES TO MAKE ALL PATHS LEAD TO THE CITY ZERO              {T.C = O(N+M), S.C = O(N+M)}
/*
Initialize a count variable = 0, make and adj list, with pair<value, 1or0(real path or imaginary path)> and vised array
then make dfs call in dfs call simple dfs mark vis[node] = 1 then iterate over adjcency list and if not vis[node / v] ,
if(it.second == 1) real path(means it is real and moving away from 0 so we have to flip) count++ finally return count.
*/
class Solution {
public:
    void dfs(vector<vector<pair<int, int>>>&adj, vector<bool>&vis, int &count, int node){
        vis[node] = 1;
        for(auto it : adj[node]){              //pair<int, int>it    {value, (1 or 0)}
            int v = it.first;
            int check = it.second;             //1 = real, 0 = imaginary
            if(!vis[v]){                       //node's value or node is not visited
                if(check == 1){                //means it is real and moving away from 0 so we have to flip
                    count++;
                }
                dfs(adj, vis, count, v);
            }
        }
    }
    int minReorder(int n, vector<vector<int>>& connections) {
        int count = 0;                                //no. of flips
        vector<vector<pair<int, int>>>adj(n);
        vector<bool>vis(n, 0);
        for(auto it : connections){
            int u = it[0];
            int v = it[1];

            adj[u].push_back({v, 1});                 //real path(1)
            adj[v].push_back({u, 0});                 //imaginary path(0)
        }

        dfs(adj, vis, count, 0);                             //0 = starting/node index
        return count;
    }
};
/*
Example 1:
Input: n = 6, connections = [[0,1],[1,3],[2,3],[4,0],[4,5]]
Output: 3
Explanation: Change the direction of edges show in red such that each node can reach the node 0 (capital).

Example 2:
Input: n = 5, connections = [[1,0],[1,2],[3,2],[3,4]]
Output: 2
Explanation: Change the direction of edges show in red such that each node can reach the node 0 (capital).

Example 3:
Input: n = 3, connections = [[1,0],[2,0]]
Output: 0
*/


//118. EVALUATE DIVISION                                       {T.C = O(N+M), S.C = O(N+M)}
/*
similar question to graph with with weight or graph traversal
*/
class Solution {
public:
    void dfs(unordered_map<string, vector<pair<string, double>>> &adj, string src, string dst, double &ans, double prod, unordered_set<string>&vis){
        if(vis.find(src) != vis.end()){
            return;
        }
        vis.insert(src);       //mark visited
        if(src == dst){
            ans = prod;
            return;
        }
        for(auto it : adj[src]){
            string v = it.first;
            double val = it.second;
            dfs(adj, v, dst, ans, prod*val, vis);
        }
    }
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        //similar question to graph with with weight or graph traversal
        int n = equations.size();

        //create adjacency list
        // unordered_map<int, int>mp;
        unordered_map<string, vector<pair<string, double>>>adj;
        for(int i = 0 ; i < n ; i++){
            string u = equations[i][0];       //"a"
            string v = equations[i][1];       //"b"
            double val = values[i];            //2

            adj[u].push_back({v, val});     //a/b
            adj[v].push_back({u, 1.0/val}); //b/a
        }

        vector<double>result;

        for(auto it : queries){
            string src = it[0];
            string dst = it[1];

            double ans = -1.0;
            double prod = 1.0;


            if(adj.find(src) != adj.end()){
                unordered_set<string>vis;                 //vis set created each time 
                dfs(adj, src, dst, ans, prod, vis);
            }
            result.push_back(ans);
        }
        return result;
    }
};
/*
Example 1:
Input: equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
Output: [6.00000,0.50000,-1.00000,1.00000,-1.00000]
Explanation: 
Given: a / b = 2.0, b / c = 3.0
queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? 
return: [6.0, 0.5, -1.0, 1.0, -1.0 ]
note: x is undefined => -1.0

Example 2:
Input: equations = [["a","b"],["b","c"],["bc","cd"]], values = [1.5,2.5,5.0], queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
Output: [3.75000,0.40000,5.00000,0.20000]

Example 3:
Input: equations = [["a","b"]], values = [0.5], queries = [["a","b"],["b","a"],["a","c"],["x","y"]]
Output: [0.50000,2.00000,-1.00000,-1.00000]
*/


/*------------------------------------------------------- GRAPH BFS ----------------------------------------------------------*/
//119. NEAREST EXIT FROM ENTRANCE IN MAZE                                   {T.C = O(N*M), S.C = O(N)}
/*
We use simple BFS approach, take queue initialize with entrance position , take count of minSteps =0, mark grid visited (grid[en[0]][en[1]] = '+')
while !q.empty() take sz of queue in that while (que sze--) simply extract top element and check the condition if exit find return minSteps else
return -1;
*/
class Solution {
public:
    int nearestExit(vector<vector<char>>& grid, vector<int>& entrance) {
        int n = grid.size();
        int m = grid[0].size();

        queue<pair<int, int>> q;
        q.push({entrance[0], entrance[1]});                 //push initial position(entrance of maze)
        int minSteps = 0;

        // mark the entrance cell as visited
        grid[entrance[0]][entrance[1]] = '+';

        while(!q.empty()){
            int sz = q.size();
            while(sz--){
                auto frontNode = q.front();
                q.pop();
                int i = frontNode.first;
                int j = frontNode.second;

                // if exit cell is found return minSteps
                if(!(i == entrance[0] && j == entrance[1]) && (i == 0 || j == 0 || i == n - 1 || j == m - 1)){
                    return minSteps;
                }

                //explore all neighbours
                vector<vector<int>>directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
                for(auto it : directions){
                    int new_i = i + it[0];
                    int new_j = j + it[1];

                    if(new_i >= 0 && new_i < n && new_j >= 0 && new_j < m && grid[new_i][new_j] != '+'){
                        grid[new_i][new_j] = '+';
                        q.push({new_i, new_j});
                    }
                }
            }
            minSteps++;
        }
        return -1;
    }
};
/*
Example 1:
Input: maze = [["+","+",".","+"],
               [".",".",".","+"],
               ["+","+","+","."]], 
       entrance = [1,2]
Output: 1
Explanation: There are 3 exits in this maze at [1,0], [0,2], and [2,3].
Initially, you are at the entrance cell [1,2].
- You can reach [1,0] by moving 2 steps left.
- You can reach [0,2] by moving 1 step up.
It is impossible to reach [2,3] from the entrance.
Thus, the nearest exit is [0,2], which is 1 step away.

Example 2:
Input: maze = [["+","+","+"],
               [".",".","."],
               ["+","+","+"]],
       entrance = [1,0]
Output: 2
Explanation: There is 1 exit in this maze at [1,2].
[1,0] does not count as an exit since it is the entrance cell.
Initially, you are at the entrance cell [1,0].
- You can reach [1,2] by moving 2 steps right.
Thus, the nearest exit is [1,2], which is 2 steps away.

Example 3:
Input: maze = [[".","+"]],
       entrance = [0,0]
Output: -1
Explanation: There are no exits in this maze.
*/


//120. ROTTING ORANGES                                                         {T.C = O(N*M), S.C = O(N)}
/*
Take queue{{row, col}, time} traverse in a grid if its 2 (rotten) push in queue with 0 time and mark vis = 2, else vis = 0(not rotten) , if element
is 1 then cntFreshOranges++, intitialze variable time and count = 0, apply bfs and check for neighbours , finally return time.
*/
class Solution {
  public:
    int orangesRotting(vector<vector<int>>&grid) {
      int n = grid.size();
      int m = grid[0].size();

      queue<pair<pair<int, int>,int>>q;                // store {{row, column}, time}
      int vis[n][m];
      int cntFresh = 0;
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
          if (grid[i][j] == 2) {                      // if cell contains rotten orange push in queue with time 0 and mark vis(rotten {2})
            q.push({{i, j}, 0}); 
            vis[i][j] = 2;                            
          }
          else {                                      // if not rotten
            vis[i][j] = 0;
          }
          if (grid[i][j] == 1){                       // count fresh oranges
            cntFresh++;
          } 
        }
      }

      int tm = 0;
      int cnt = 0;
      vector<vector<int>>directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

      // bfs traversal (until the queue becomes empty)
      while (!q.empty()) {
        int r = q.front().first.first;
        int c = q.front().first.second;
        int t = q.front().second;
        tm = max(tm, t);
        q.pop();

        //explore neighbours
        for (auto it : directions) {
          int nrow = r + it[0];
          int ncol = c + it[1];
          if (nrow >= 0 && nrow < n && ncol >= 0 && ncol < m && vis[nrow][ncol] == 0 && grid[nrow][ncol] == 1) {
            q.push({{nrow, ncol}, t + 1});                // push in queue with timer increased
            vis[nrow][ncol] = 2;                          // mark as rotten
            cnt++;
          }
        }
      }
      
      if (cnt != cntFresh){                              // if all oranges are not rotten
        return -1;                     
      } 
      return tm;
    }
};
/*
Example 1:
Input: grid = [[2,1,1],
               [1,1,0],
               [0,1,1]]
Output: 4

Example 2:
Input: grid = [[2,1,1],
               [0,1,1],
               [1,0,1]]
Output: -1
Explanation: The orange in the bottom left corner (row 2, column 0) is never rotten, because rotting only happens 4-directionally.

Example 3:
Input: grid = [[0,2]]
Output: 0
Explanation: Since there are already no fresh oranges at minute 0, the answer is just 0.
*/


/*----------------------------------------------------- HEAP / PRIORITY QUEUE -----------------------------------------*/
//121. KTH LARGEST ELEMENT IN AN ARRAY                                      {T.C = O(N*LOGK), S.C = O(K)}
/*
Using min Heap put first k elements in heap, then for rest of element check if curr element is bigger than minHeap.top()
then pop top element and push curr element to minheap, finally return minHeap.top( k th largest element).
*/
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        //creating min heap (kth largest element on top of minHeap others node are greater then k)
        priority_queue<int, vector<int>, greater<int>>minHeap;

        //put first k elements to min heap
        for(int i = 0 ; i < k ; i++){
            minHeap.push(nums[i]);
        }

        //for rest of elements
        for(int i = k ; i < nums.size() ; i++){
            if(nums[i] > minHeap.top()){
                minHeap.pop();
                minHeap.push(nums[i]);
            }
        }
        return minHeap.top();
    }
};
/*
Example 1:
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
Example 2:

Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4
*/


//122. SMALLEST NUMBER IN INIFINITE SET                                   {T.C = O(N), S.C = O(N)}
class SmallestInfiniteSet {
public:
    priority_queue<int, vector<int>, greater<int>>minHeap;
    unordered_set<int>st;
    int currSmallest;
    SmallestInfiniteSet() {
        currSmallest = 1;
    }
    
    int popSmallest() {
        int ans;
        if(!minHeap.empty()){
            ans = minHeap.top();
            minHeap.pop();
            st.erase(ans);
        }else{
            ans = currSmallest;
            currSmallest++;
        }
        return ans;
    }
    
    void addBack(int num) {
        if(num >= currSmallest || st.find(num) != st.end()){
            return;
        }
        minHeap.push(num);
        st.insert(num);
    }
};
/*
Example 1:

Input
["SmallestInfiniteSet", "addBack", "popSmallest", "popSmallest", "popSmallest", "addBack", "popSmallest", "popSmallest", "popSmallest"]
[[], [2], [], [], [], [1], [], [], []]
Output
[null, null, 1, 2, 3, null, 1, 4, 5]

Explanation
SmallestInfiniteSet smallestInfiniteSet = new SmallestInfiniteSet();
smallestInfiniteSet.addBack(2);    // 2 is already in the set, so no change is made.
smallestInfiniteSet.popSmallest(); // return 1, since 1 is the smallest number, and remove it from the set.
smallestInfiniteSet.popSmallest(); // return 2, and remove it from the set.
smallestInfiniteSet.popSmallest(); // return 3, and remove it from the set.
smallestInfiniteSet.addBack(1);    // 1 is added back to the set.
smallestInfiniteSet.popSmallest(); // return 1, since 1 was added back to the set and
                                   // is the smallest number, and remove it from the set.
smallestInfiniteSet.popSmallest(); // return 4, and remove it from the set.
smallestInfiniteSet.popSmallest(); // return 5, and remove it from the set.
*/


//123. MAXIMUM SUBSEQUENCE SCORE                                 {T.C = O(N*LOGN), S.C = O(N)}
/*
First store the vectors in single vector<pair> then sort in decreasing order on the basis of second vector, then make
minHeap and variable kSum(store first k elment sum). take maxi (final ans variable) = kSum * v[k-1].second(mini element)
after that traverse from k to n update kSum = v[i].first-minHeap.top() {add new element and remove top element(minimum)}
in ans = max(ans, kSum*v[i].second), finally return ans.
*/
class Solution {
public:
    long long maxScore(vector<int>& nums1, vector<int>& nums2, int k) {
        int n = nums1.size();                    //size is same for both nums1 and nums2
        vector<pair<int, int>>v(n);                 //{nums1[i], nums2[i]}
        for(int i = 0 ; i < n ; i++){
            v.push_back({nums1[i], nums2[i]});
        }

        auto lambda = [&](auto &pair1, auto &pair2){
            return pair1.second > pair2.second;
        };
        sort(v.begin(), v.end(), lambda);       //lambda = decreasing sort on the basis of second element in pair

        priority_queue<int, vector<int>, greater<int>>minHeap;
        long long kSum = 0;
        for(int i = 0 ; i < k ; i++){
            kSum += v[i].first;
            minHeap.push(v[i].first);
        }

        long long maxiAns = kSum * v[k-1].second;    //sum (nums1 range) * max(nums2 range which is k-1 th after sorting)

        for(int i = k ; i < n ; i++){
            kSum += v[i].first - minHeap.top();        //add next element and remove kth mini element
            minHeap.pop();
            minHeap.push(v[i].first);
            maxiAns = max(maxiAns, kSum * v[i].second);
        }

        return maxiAns;
    }
};
/*
Example 1:
Input: nums1 = [1,3,3,2], nums2 = [2,1,3,4], k = 3
Output: 12
Explanation: 
The four possible subsequence scores are:
- We choose the indices 0, 1, and 2 with score = (1+3+3) * min(2,1,3) = 7.
- We choose the indices 0, 1, and 3 with score = (1+3+2) * min(2,1,4) = 6. 
- We choose the indices 0, 2, and 3 with score = (1+3+2) * min(2,3,4) = 12. 
- We choose the indices 1, 2, and 3 with score = (3+3+2) * min(1,3,4) = 8.
Therefore, we return the max score, which is 12.

Example 2:
Input: nums1 = [4,2,3,1,1], nums2 = [7,5,10,9,6], k = 1
Output: 30
Explanation: 
Choosing index 2 is optimal: nums1[2] * nums2[2] = 3 * 10 = 30 is the maximum possible score.
*/


//124. TOTAL COST TO HIRE K WORKERS                                 {T.C = O(K*LOGN), S.C = O(N)}
/*
Using 2 minHeap first for 1st k costs another for last k costs , while count < k , push costs in minHeap < candidates 
and if both pq is not empty then compare there top element and push the lowest cost in ans , finally return ans.
*/
class Solution {
public:
    long long totalCost(vector<int>& costs, int k, int candidates) {
        long long ans = 0;
        int n = costs.size();
        int count = 0;
        priority_queue<int, vector<int>, greater<int>>minHeapFirst, minHeapLast;

        int i = 0;
        int j = n-1;
        while(count < k){
            while(minHeapFirst.size() < candidates && i <= j){    //push element in pq1 and pq2
                minHeapFirst.push(costs[i++]);
            }
            while(minHeapLast.size() < candidates && j >= i){
                minHeapLast.push(costs[j--]);
            }

            int cost1 = INT_MAX;
            int cost2 = INT_MAX;
            if(minHeapFirst.size() > 0){
                cost1 = minHeapFirst.top();
            }
            if(minHeapLast.size() > 0){
                cost2 = minHeapLast.top();
            }

            if(cost1 <= cost2){                                //add minimum cost to ans
                ans += cost1;
                minHeapFirst.pop();
            }else{
                ans += cost2;
                minHeapLast.pop();
            }
            count++;
        }
        return ans;
    }
};
/*
Example 1:
Input: costs = [17,12,10,2,7,2,11,20,8], k = 3, candidates = 4
Output: 11
Explanation: We hire 3 workers in total. The total cost is initially 0.
- In the first hiring round we choose the worker from [17,12,10,2,7,2,11,20,8]. The lowest cost is 2, and we break the tie by the smallest index, which is 3. The total cost = 0 + 2 = 2.
- In the second hiring round we choose the worker from [17,12,10,7,2,11,20,8]. The lowest cost is 2 (index 4). The total cost = 2 + 2 = 4.
- In the third hiring round we choose the worker from [17,12,10,7,11,20,8]. The lowest cost is 7 (index 3). The total cost = 4 + 7 = 11. Notice that the worker with index 3 was common in the first and last four workers.
The total hiring cost is 11.

Example 2:
Input: costs = [1,2,4,1], k = 3, candidates = 3
Output: 4
Explanation: We hire 3 workers in total. The total cost is initially 0.
- In the first hiring round we choose the worker from [1,2,4,1]. The lowest cost is 1, and we break the tie by the smallest index, which is 0. The total cost = 0 + 1 = 1. Notice that workers with index 1 and 2 are common in the first and last 3 workers.
- In the second hiring round we choose the worker from [2,4,1]. The lowest cost is 1 (index 2). The total cost = 1 + 1 = 2.
- In the third hiring round there are less than three candidates. We choose the worker from the remaining workers [2,4]. The lowest cost is 2 (index 0). The total cost = 2 + 2 = 4.
The total hiring cost is 4.
*/


/*---------------------------------------------------- BINARY SEARCH -------------------------------------------------*/
//for remove error(line)
int guess(int a);
//--------------------

//125. GUESS NUMBER HIGHER OR LOWER                                  {T.C = O(LOGN), S.C = O(1)}
/*
First there is Pre defined guess function with output(0, 1, -1) , use binary search and find the location of guess number
and update accordingly finally return pick.
*/
class Solution {
public:
    int guessNumber(int n) {
        int pick;
        int start = 1;                          //range [1, n]
        int end = n;
        while(start <= end){
            int mid = start + (end-start)/2;
            if(guess(mid) == 0){
                pick = mid;
                break;
            }else if(guess(mid) == 1){
                start = mid+1;
            }else{ //guess(mid) == -1
                end = mid-1;
            }
        }
        return pick;
    }
};
/*
Example 1:
Input: n = 10, pick = 6
Output: 6

Example 2:
Input: n = 1, pick = 1
Output: 1

Example 3:
Input: n = 2, pick = 1
Output: 1
*/


//126. SUCCESSFUL PAIRS OF SPELLS AND POTIONS
//BRUTE FORCE                                                 {T.C = O(N*M), S.C = O(1)} 
/*
Use 2 loops first loop traverse on spells another on potions in which store products of both vector and if the product 
is equal or greater then success then increase the pairCount after that push paircount in ans and return ans.
*/
class Solution {
public:
    vector<int> successfulPairs(vector<int>& spells, vector<int>& potions, long long success) {
        vector<int>ans;
        for(int i = 0 ; i < spells.size() ; i++){
            int pairCount = 0;
            for(int j = 0 ; j < potions.size() ; j++){
                // long long product = (long long)(spells[i]) * (long long)(potions[j]);
                int product = spells[i]*potions[j];
                if(product >= success){
                    pairCount++;
                }
            }
            ans.push_back(pairCount);
        }
        return ans;
    }
};

//OPTIMIZED APPROACH (BINARY SEARCH)                                           {T.C = O(N+M)LOGN, S.C = O(N)}
/*
First sort the potions vector then iteratre in spells and apply binary search by taking pointer on start(0) and end(n(potions)-1)
take long long x = potions[mid](value) if it * value < success start = mid+1 else end = mid-1 , after that push the count
of successful pairs(n-start), and return ans;
*/
class Solution {
public:
    vector<int> successfulPairs(vector<int>& spells, vector<int>& potions, long long success) {
        int m = spells.size();
        int n = potions.size();
        vector<int>ans;
        sort(potions.begin(), potions.end());
        for(auto it : spells){
            int start = 0;
            int end = n-1;         
            while(start <= end){
                int mid = end - (end-start)/2;
                long long x = potions[mid];
                if(it * x < success){
                    start = mid+1;
                }else{
                    end = mid-1;
                }
            }
            ans.push_back(n-start);                      //return difference
        }
        return ans;
    }
};
/*
Example 1:
Input: spells = [5,1,3], potions = [1,2,3,4,5], success = 7
Output: [4,0,3]
Explanation:
- 0th spell: 5 * [1,2,3,4,5] = [5,10,15,20,25]. 4 pairs are successful.
- 1st spell: 1 * [1,2,3,4,5] = [1,2,3,4,5]. 0 pairs are successful.
- 2nd spell: 3 * [1,2,3,4,5] = [3,6,9,12,15]. 3 pairs are successful.
Thus, [4,0,3] is returned.

Example 2:
Input: spells = [3,1,2], potions = [8,5,8], success = 16
Output: [2,0,2]
Explanation:
- 0th spell: 3 * [8,5,8] = [24,15,24]. 2 pairs are successful.
- 1st spell: 1 * [8,5,8] = [8,5,8]. 0 pairs are successful. 
- 2nd spell: 2 * [8,5,8] = [16,10,16]. 2 pairs are successful. 
Thus, [2,0,2] is returned.
*/


//127. FIND PEAK ELEMENT                                              {T.C = O(LOGN), S.C = O(1)}
//BRUTE FORCE                                                  
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        return max_element(nums.begin(), nums.end()) - nums.begin();
    }
};
/*OPTIMIZED APPROACH (BINARY SEARCH)                                {T.C = O(LOGN), S.C = O(1)}
search for peak we have to check adjacent element (nums[mid] < nums[mid+1]) then peak may be in right side else
peak in left side after breaking loop return start or low index.
*/
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int n = nums.size();
        int start = 0;
        int end = n - 1;

        while(start < end){
            int mid = start + (end-start)/2;
            if(nums[mid] < nums[mid+1]){              //peak may be in right side
                start = mid+1;
            }else{ //(nums[mid] > nums[start])        //peak may be in left side
                end = mid;
            }
        }
        return start;
    }
};
/*
Example 1:
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.

Example 2:
Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.
*/

//128. KOKO EATING BANANAS                                        {T.C = O(LOGN), S.C = O(1)}
/*
Iniitialize two pointer one is one start(piles) another on max element of piles vector , now apply binary search
if(hour >= counthours of getHours(piles, mid)) ans = mid, end = mid-1 else start = mid+1, and return ans, In get hour func
initialize count = 0, now iterate on piles count += it/mid for remaining(it%mid != 0) count++ finally return count.
*/
class Solution {
public:
    long long getHours(vector<int>&piles, int mid){
        long long count = 0;
        for(auto it : piles){
            count += it/mid;                    //4/3 for 3/3 count++
            if(it % mid != 0){                    //4%3 for rest 1 banana count++
                count++;
            }
        }
        return count;
    }
    int minEatingSpeed(vector<int>& piles, int h) {
        int start = 1;
        int end = *max_element(piles.begin(), piles.end());
        int ans = start;

        while(start <= end){
            int mid = start + (end-start)/2;
            if(h >= getHours(piles, mid)){
                ans = mid;
                end = mid-1;
            }else{
                start = mid+1;
            }
        }
        return ans;
    }
};
/*
Example 1:
Input: piles = [3,6,7,11], h = 8
Output: 4

Example 2:
Input: piles = [30,11,23,4,20], h = 5
Output: 30

Example 3:
Input: piles = [30,11,23,4,20], h = 6
Output: 23
*/


/*----------------------------------------------------- BACKTRACKING --------------------------------------------------*/
//BACKTRACKING = POSSIBLE / PERMUTATIONS / COMBINATIONS / SUBSETS ETC.
/*
STEP  1. DO SOMETHING OR EXPLORE 
STEP  2. EXPLORE
STEP  3. REVERT STEP-1 AND FURTHER EXPLORE*/
//129. LETTER COMBINATIONS OF A PHONE NUMBER                             {T.C = O(EXP), S.C = O(N)}
/*
Use brute force recursive solution and apply backtracking to it. first store all the strings of keypad in a map, then
in solve function if index >= digits.length() . ans.push_back(output) , extract number (digits[index]-'0') now traverse value
(mapping[number]) push value[i] i output then make recursive call and pop back(backtrack).
*/
class Solution {
private : 
    void solve(string digits ,vector<string> &ans ,string output ,string mapping[], int index){
        //base case
        if(index >= digits.length()){
            ans.push_back(output);
            return;
        }

        int number = digits[index] - '0';      //char to number
        string value = mapping[number];

        for(int i = 0 ; i <  value.length() ; i++){
            output.push_back(value[i]);
            solve(digits ,ans, output, mapping, index+1);
            output.pop_back();                //to backtrack and explore other combos
        }

    }
public:
    vector<string> letterCombinations(string digits) {
        vector<string> ans;
        string output = "";

        if(digits.length() == 0){
            return ans;
        }

        string mapping[10] = {"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};   //0 to 9 (10)
        
        solve(digits ,ans ,output ,mapping, 0);         //0 = starting index
        return ans;
    }
};
/*
Example 1:
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

Example 2:
Input: digits = ""
Output: []

Example 3:
Input: digits = "2"
Output: ["a","b","c"]
*/


//130. COMBINATION SUM III                                       {T.C = O(EXP), S.C = O(N^2)}
/*
Take combo(combination) vector and ans(vector of vector) and make solve call(recursion) with 1 = idx and 0 = sum , in solve
if(k == 0 && sum == n) ans.push_back combo, take a for loop from currindex to 9 and push curr i and make recursive call
and pop back for backtracking.
*/
class Solution {
public:
    void solve(int k ,int n, vector<int>&combo, vector<vector<int>>&ans, int idx, int sum){
        //base case
        if(k == 0 && sum == n){
            ans.push_back(combo);
            return;
        }
        if(sum > n){
            return;
        }
        for(int i = idx ; i <= 9 ; i++){                    //i = idx (unique number required)
            combo.push_back(i);
            solve(k-1, n, combo, ans, i+1, sum+i);
            combo.pop_back();                                  //backtracking
        }
    }
    vector<vector<int>> combinationSum3(int k, int n) {     //n = required sum
        vector<int>combo;
        vector<vector<int>>ans;
        solve(k, n, combo, ans, 1, 0);                 //1 = index(1 to 9) , 0 = sum
        return ans;
    }
};
/*
Example 1:
Input: k = 3, n = 7
Output: [[1,2,4]]
Explanation:
1 + 2 + 4 = 7
There are no other valid combinations.

Example 2:
Input: k = 3, n = 9
Output: [[1,2,6],[1,3,5],[2,3,4]]
Explanation:
1 + 2 + 6 = 9
1 + 3 + 5 = 9
2 + 3 + 4 = 9
There are no other valid combinations.

Example 3:
Input: k = 4, n = 1
Output: []
Explanation: There are no valid combinations.
Using 4 different numbers in the range [1,9], the smallest sum we can get is 1+2+3+4 = 10 and since 10 > 1, there are no valid combination.
*/


/*---------------------------------------- DYNAMIC PROGRAMMING (DP) --------------------------------------------------*/
//131. NTH TRIBONACCI NUMBER                                         {T.C = O(N), S.C = O(N)}
/*
Bottom up(initial index = n) approach in TOP DOWN (DP) first check for base cases , then follow standardise dp steps
and store(dp[i] = solve(i-3)+solve(i-2)+solve(i-1)).
*/
class Solution {
public:
    int solve(int n, vector<int>&dp, int i){
        //base case
        if(i == 0){
            return 0;
        }
        if(i == 1 || i == 2){
            return 1;
        }
        //step3 if ans already present return it
        if(dp[i] != -1){
            return dp[i];
        }

        //step2 store ans in dp
        dp[i] = solve(n, dp, i-3)+solve(n, dp, i-2) + solve(n, dp, i-1);
        return dp[i];
    }
    int tribonacci(int n) {
        if(n == 0){
            return 0;
        }
        vector<int>dp(n+1, -1);
        int ans = solve(n, dp, n);                         //n = initial index = n
        return ans;
    }
};
/*
Example 1:
Input: n = 4
Output: 4
Explanation:
T_3 = 0 + 1 + 1 = 2
T_4 = 1 + 1 + 2 = 4

Example 2:
Input: n = 25
Output: 1389537
*/


//132. MIN COST CLIMBING STAIRS                                       {T.C = O(N), S.C = O(N)}
/*
Use dp and take min(n-1 and n-2) , in solveMem handle base case if(n == 0) return cost[0] similarly for 1 , store ans in
dp = cost[i] + min(solveMem(n-1), solveMem(n-2))
*/
class Solution {
public:
    int solveMem(vector<int>&cost , int n , vector<int>&dp){
        //base case
        if(n == 0){
            return cost[0];
        }
        if(n == 1){
            return cost[1];
        }

        //step-3 if answer already present retur it.
        if(dp[n] != -1){
            return dp[n];
        }

        //step2 store ans in dp
        dp[n] = cost[n] + min(solveMem(cost, n-1, dp), solveMem(cost, n-2, dp));
        return dp[n];
    }
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();

        //step-1 create dp array
        vector<int>dp(n+1, -1);
        int ans = min(solveMem(cost, n-1, dp), solveMem(cost, n-2, dp));
        return ans;
    }
};
/*
Example 1:
Input: cost = [10,15,20]
Output: 15
Explanation: You will start at index 1.
- Pay 15 and climb two steps to reach the top.
The total cost is 15.

Example 2:
Input: cost = [1,100,1,1,1,100,1,1,100,1]
Output: 6
Explanation: You will start at index 0.
- Pay 1 and climb two steps to reach index 2.
- Pay 1 and climb two steps to reach index 4.
- Pay 1 and climb two steps to reach index 6.
- Pay 1 and climb one step to reach index 7.
- Pay 1 and climb two steps to reach index 9.
- Pay 1 and climb one step to reach the top.
The total cost is 6.
*/


//133. DOMINO AND TROMINO TILING                                           {T.C = O(N), S.C = O(N)}
/*
We have to use formula ans = 2*(n-1) + (n-3) and store ans n dp , finally return it.
*/
class Solution {
public:
    int solveMem(int n, vector<int>&dp){
        int mod = 1e9+7;
        //base case
        if(n == 1 || n == 2){
            return n;
        }
        if(n == 3){
            return 5;
        }

        //step3 if ans already present return it
        if(dp[n] != -1){
            return dp[n];
        }

        //step2 store ans in dp
        dp[n] = (2*solveMem(n-1, dp)%mod + solveMem(n-3, dp)%mod ) % mod;    //formula = 2*(n-1) + (n-3)

        return dp[n];
    }
    int numTilings(int n) {
        //step1 create a dp vector
        vector<int>dp(n+1, -1);
        return solveMem(n, dp);                         //n = traverse from last
    }
};
/*
Example 1:
Input: n = 3
Output: 5
Explanation: The five different ways are show above.

Example 2:
Input: n = 1
Output: 1
*/


/*----------------------------------------------- DP MULTIDIMENSIONAL ------------------------------------------------*/
//134. EDIT DISTANCE                                                      {T.C = O(N*M), S.C = O(N*M)}
/*
First check base cases then if word1[i] == word[j] then move both pointers else perform insert(i, j+1{add}),
delete(i+1{skip}, j), replace operations(i+1, j+1).
*/
class Solution {
public:
    int solveMem(string &word1, string &word2, vector<vector<int>>& dp, int i, int j) {
        int m = word1.length();
        int n = word2.length();

        // Base case
        if (i == m) {                      //string1 is fully traversed now rest(n-j length is ans)
            return n - j;
        }
        if (j == n) {                      //string2 is fully traversed now rest(m-i length is ans)
            return m - i;
        }

        //step3 if answer already present, return it
        if (dp[i][j] != -1) {
            return dp[i][j];
        }

        int mini = INT_MAX;
        // Recursive call
        if (word1[i] == word2[j]) {
            mini = solveMem(word1, word2, dp, i + 1, j + 1);
        } else { // Min of insert, delete, replace
            int insertOp = solveMem(word1, word2, dp, i, j + 1);   //It increments the index j of word2 while keeping the index i of word1 the same
            int deleteOp = solveMem(word1, word2, dp, i + 1, j);   //It increments the index i of word1 while keeping the index j of word2 the same
            int replacOp = solveMem(word1, word2, dp, i + 1, j + 1);//It increments the index i of word1 as well as index j of word2

            mini = 1 + min({insertOp, deleteOp, replacOp});         //1 (current operation + other)         
        }

        //step2 store ans in dp
        return dp[i][j] = mini;
    }

    int minDistance(string word1, string word2) {
        int n = word1.length();
        int m = word2.length();

        //step1 create a dp vector
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, -1));
        return solveMem(word1, word2, dp, 0, 0);                  //0, 0 = starting index of both strings
    }
};
/*
Example 1:
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Example 2:
Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')
*/


/*--------------------------------------------------- BIT MANIPULATION -----------------------------------------------*/
//135. SINGLE NUMBER (ALL TWICE EXCEPT ONE(ANS))                                                {T.C = O(N), S.C = O(1)}
// (1 element appear once and other twice, find 1 elements which appears once)                                                {T.C = O(N), S.C = O(1)}
/*
xor of each element (1^1 = 0 and 1^0 = 1) so if the element is same of pairwise then it will cancel or and remaining element is ans.
*/
class Solution {
public:
    int singleNumber(vector<int>& nums) {             //xor = different element != 0 else 0 
        int xorAns = 0;
        for(int i = 0 ; i < nums.size(); i++){
            xorAns = xorAns^nums[i];
        }
        return xorAns;
    }
};
/*
Example 1:
Input: nums = [2,2,1]
Output: 1

Example 2:
Input: nums = [4,1,2,1,2]
Output: 4
*/


//136. SINGLE NUMBER II   (1 element appear once and other thrice , find 1 elements which appears once)
/*
brute force = use unordered map store all values in map and return whose count is 1.   {T.C = O(N*LOGN), S.C = O(N)}
*/
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        unordered_map<int, int>mp;
        for(int i = 0 ; i < nums.size(); i++){
            mp[nums[i]]++;
        }
        for(auto it : mp){
            if(it.second == 1){
                return it.first;
            }
        }
        return -1;
    }
};
/*
optimized approach - bitwise                                                               {T.C = O(N), S.C = O(1)}
ex = [2 , 2, 2 , 3]
ones = 0  2  0  0  3(ans)
twos = 0  0  2  0  0
*/
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ones = 0, twos = 0;           //ones = stores element which repeating once and similary twos 
        
        for (int num : nums) {
            ones = (ones ^ num) & ~twos;
            twos = (twos ^ num) & ~ones;
        }
        
        return ones;
    }
};
/*
Example 1:
Input: nums = [2,2,3,2]
Output: 3

Example 2:
Input: nums = [0,1,0,1,0,1,99]
Output: 99
*/


//137. SINGLE NUMBER III  (2 elements appears once and other twice, find 2 elements which appears once)
//{T.C = O(N), S.C = O(1)}
/*
*/
class Solution {
public:
    vector<int> singleNumber(vector<int>& nums) {
        vector<int> ans(2, 0); // Initialize the answer vector with two elements

        int xorResult = 0;
        
        // Step 1: Get the XOR of all elements
        for (int num : nums) {
            xorResult ^= num;
        }
        
        // Step 2: Find the rightmost set bit
        int rightmostSetBit = 1;
        while ((rightmostSetBit & xorResult) == 0) {
            rightmostSetBit <<= 1;
        }
        
        // Step 3 & 4: Divide into two groups and XOR separately
        for (int num : nums) {
            if ((num & rightmostSetBit) == 0) {
                ans[0] ^= num; // XOR for the group where the bit is not set
            } else {
                ans[1] ^= num; // XOR for the group where the bit is set
            }
        }
        
        return ans;
    }
};
/*
Example 1:
Input: nums = [1,2,1,3,2,5]
Output: [3,5]
Explanation:  [5, 3] is also a valid answer.

Example 2:
Input: nums = [-1,0]
Output: [-1,0]
*/


//138. MINIMUM FLIPS TO MAKE (A) OR (B) EQUAL TO (C)
//BRUTE FORCE APPROACH(INTUTIVE)                                            {T.C = O(N), S.C = O(1)}
/*
Check the bits one by one whether they need to be flipped. basically we have to check for all the numbers untill all will 0
and check for right most bit of c if it is 1 then flips++ (one of a or b should 0)  else it is 0, then (both should 0) and
make right shift of all a, b, c , finally return flips.
*/
class Solution {
public:
    int minFlips(int a, int b, int c) {
        int flips = 0;
        while( a != 0 || b != 0 || c != 0){        //break only a,b,c=0,0,0
            if((c & 1) == 1){                      //for c's right most bit 1 (one of  a or b should 0)
                if((a & 1) == 0 && (b & 1) == 0){
                    flips++;
                }
            }else if((c & 1) == 0){                //for c's right most bit 0 (both should be 0)
                if((a & 1) == 1){
                    flips++;
                }
                if((b & 1) == 1){
                    flips++;
                }
            }
            a >>= 1;
            b >>= 1;
            c >>= 1;
        }
        return flips;
    }
};

//USING STL                                            {T.C = O(N), S.C = O(1)}
class Solution {
public:
    int minFlips(int a, int b, int c) {
        int temp = (a|b)^c;                //flips
        int extra = ((a&b) & temp);          //handle extra case
        return __builtin_popcount(temp) + __builtin_popcount(extra);   //gives count of 1 bits(ans)
    }
};
/*
Example 1:
Input: a = 2, b = 6, c = 5
Output: 3
Explanation: After flips a = 1 , b = 4 , c = 5 such that (a OR b == c)

Example 2:
Input: a = 4, b = 2, c = 7
Output: 1

Example 3:
Input: a = 1, b = 2, c = 3
Output: 0
*/


/*--------------------------------------------------- TRIE ------------------------------------------------------------*/
//139.  IMPLEMENT TRIE (PREFIX TREE)                    {T.C = O(N + N), S.C = O(N*M {no. of word , lenghth of word})}
/*
//trie's main functions = insert, search, delete
//first we have to create a struct (isEndOfWord, child array)
//second we have to write function for creating new node for an trie
//initialize trieNode with root
//write for insert and with some upgradation search and startswith fucntion made
*/
class Trie {
public:
    struct trieNode{                   //basic requirement of trie is endofword and child
        bool isEndOfWord;
        trieNode *child[26];
    };

    trieNode *getNode(){              //for making newnode in trie
        trieNode* newNode = new trieNode();

        newNode->isEndOfWord = false;     //endofword should be false for new node
        for(int i = 0 ; i < 26 ; i++){
            newNode->child[i] = NULL;     //creating empty vector
        }
        return newNode;
    }

    trieNode* root;

    Trie() {
        root = getNode();
    }
    
    //if we create insert then search and starts with is just slight updation of code
    void insert(string word) {  //apple
        trieNode* crawler = root;             //crawler work as iterator
        for(int i = 0 ; i < word.length() ; i++){
            char ch = word[i];
            int idx = ch-'a';

            if(crawler->child[idx] == NULL){
                crawler->child[idx] = getNode();  //'a'
            }
            crawler = crawler->child[idx];       //move forward
        }
        crawler->isEndOfWord = true;      //'e' reach end
    }
    
    bool search(string word) {  //'apple'
        trieNode* crawler = root;             //crawler work as iterator
        for(int i = 0 ; i < word.length() ; i++){
            char ch = word[i];
            int idx = ch-'a';

            if(crawler->child[idx] == NULL){    //character in the word being searched for doesn't exist in the trie
                return false;
            }
            crawler = crawler->child[idx];       //move forward
        }
        if(crawler != NULL && crawler->isEndOfWord == true){  //'e'
            return true;
        }
        return false;
    }
    
    bool startsWith(string prefix) { //'app'
        trieNode* crawler = root;             //crawler work as iterator
        int i = 0;
        for(i = 0 ; i < prefix.length() ; i++){    //same as search just word => prefix
            char ch = prefix[i];
            int idx = ch-'a';

            if(crawler->child[idx] == NULL){    //character in the word being searched for doesn't exist in the trie
                return false;
            }
            crawler = crawler->child[idx];       //move forward
        }
        if(i == prefix.length()){
            return true;
        }
        return false;
    }
};
/*
Example 1:
Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]
Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True
*/


//140. SEARCH SUGGESTIONS SYSTEMS
//USING SORTING AND BINARY SEARCH                                           {T.C = O(N*LOGN), S.C = O(N)}
/*
First sort the products, then iterate to searchword string prefix += it, find the index of lower_bound and store in start, push empty vector in ans
then iterate with condition(start ; atmost 3times && prefix match ; i++ ) , ans.back().push_back(prod) , initialize bsStart with start(for efficency)'
finally return ans.
*/
class Solution {
public:
    vector<vector<string>> suggestedProducts(vector<string>& products, string searchWord) {
        vector<vector<string>>ans;
        sort(products.begin(), products.end());
        int n = products.size();
        int start = 0, bsStart = 0;
        string prefix;
        for(auto it : searchWord){
            prefix += it;
            start = lower_bound(products.begin()+bsStart, products.end(), prefix) - products.begin();  //gives index
            ans.push_back({});
            // it iterates at most three times && prefix of the product matches the current value of prefix
            for(int i = start ; i < min(start+3, n) && !products[i].compare(0, prefix.length(), prefix); i++){
                ans.back().push_back(products[i]);
            }
            bsStart = start;
        }
        return ans;
    }
};
/*
Example 1:
Input: products = ["mobile","mouse","moneypot","monitor","mousepad"], searchWord = "mouse"
Output: [["mobile","moneypot","monitor"],["mobile","moneypot","monitor"],["mouse","mousepad"],["mouse","mousepad"],["mouse","mousepad"]]
Explanation: products sorted lexicographically = ["mobile","moneypot","monitor","mouse","mousepad"].
After typing m and mo all products match and we show user ["mobile","moneypot","monitor"].
After typing mou, mous and mouse the system suggests ["mouse","mousepad"].

Example 2:
Input: products = ["havana"], searchWord = "havana"
Output: [["havana"],["havana"],["havana"],["havana"],["havana"],["havana"]]
Explanation: The only word "havana" will be always suggested while typing the search word.
*/


/*------------------------------------------------- INTERVALS ---------------------------------------------------------*/
//141. MINIMUM NUMBER OF ARROWS TO BURST BALLOONS                                  {T.C = O(N*LOGN), S.C = O(1)}
bool comp(vector<int>&a, vector<int>&b){
    return a[1] < b[1];                         
}
class Solution {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        //base case
        if(points.size() == 0){
            return 0;
        }
        sort(points.begin(), points.end(), comp);     //sort on the basis of upper bound of each interval

        int arrow = 1;                                   //1 arrow always required to burst 1 or > ballon
        int end = points[0][1];                          //upper bound 1st interval

        for(int i = 1 ; i < points.size() ; i++){
            if(points[i][0] > end){                     //lower bound of interval  > end
                arrow++;
                end = points[i][1];
            }
        }
        return arrow;
    }
};
/*
Input: points = [[10,16],[2,8],[1,6],[7,12]]
Output: 2
Explanation: The balloons can be burst by 2 arrows:
- Shoot an arrow at x = 6, bursting the balloons [2,8] and [1,6].
- Shoot an arrow at x = 11, bursting the balloons [10,16] and [7,12].
*/
   


/*---------------------------------------- MONOTONIC STACK(INCREASING STACK) -------------------------------------------*/
//142. DAILY TEMPRATURES                                          {T.C = O(N*LOGN), S.C = O(N)}
/*
Take an stack, traverse the vector(tempratures) , while(st.empty() && element > element[stk.top{index}]) , prevIdx = st.top()
put ans[prevIdx] = i-prevIdx, then push i in stack finally return ans.
*/
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        vector<int>ans(n);
        int count = 0;
        stack<int>stk;
        for(int i = 0 ; i < n ; i++){
            while(!stk.empty() && temperatures[i] > temperatures[stk.top()]){
                int prevIdx = stk.top();
                stk.pop();
                ans[prevIdx] = i - prevIdx;
            }
            stk.push(i);
        }
        return ans;
    }
};
/*
Example 1:
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]

Example 2:
Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]

Example 3:
Input: temperatures = [30,60,90]
Output: [1,1,0]
*/


//143. ONLINE STOCK SPAN                                        {T.C = O(N), S.C = O(N)}
/*
Take an stack(pair{price, span}) initialize span 1 (always) then traverse stack untill the !st.empty() && top <= price
span += top.second(curr span) , push {price , span } in stack and finally return span.
*/
class StockSpanner {
public:
    stack<pair<int, int>>stk;                          //{price, span}
    StockSpanner() {
        
    }
    
    int next(int price) {
        int span = 1;                   //current span(minimum value before curr) is 1 
        while(!stk.empty() && stk.top().first <= price){
            span += stk.top().second;
            stk.pop();
        }
        stk.push({price, span});
        return span;
    }
};
/*
Example 1:
Input
["StockSpanner", "next", "next", "next", "next", "next", "next", "next"]
[[], [100], [80], [60], [70], [60], [75], [85]]
Output
[null, 1, 1, 1, 2, 1, 4, 6]

Explanation
StockSpanner stockSpanner = new StockSpanner();
stockSpanner.next(100); // return 1
stockSpanner.next(80);  // return 1
stockSpanner.next(60);  // return 1
stockSpanner.next(70);  // return 2
stockSpanner.next(60);  // return 1
stockSpanner.next(75);  // return 4, because the last 4 prices (including today's price of 75) were less than or equal to today's price.
stockSpanner.next(85);  // return 6
*/


/*------------------------------------------------== THE END ---------------------------------------------------------*/

//150 TOP INTERVIEW QUESTIONS (LEETCODE)

//144. MERGE SORTED ARRAY                                                           {T.C = O(N), S.C = O(1)}
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i = m-1, j = n-1, k = n+m-1;
        while(i >= 0 && j >= 0){
            if(nums1[i] > nums2[j]){
                nums1[k] = nums1[i];             //first fill last index(greater element)
                k--, i--;
            }else{
                nums1[k] = nums2[j];
                k--, j--;
            }
        }
        while(i >= 0){
            nums1[k] = nums1[i];
            k--, i--;
        }
        while(j >= 0){
            nums1[k] = nums2[j];
            k--, j--;
        }
    }
};
/*
Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]
Explanation: The arrays we are merging are [1] and [].
The result of the merge is [1].
*/


//145. REMOVE ELEMENT                                                           {T.C = O(N), S.C = O(1)}
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int count = 0;
        for(int i = 0 ; i < nums.size() ; i++){
            if(nums[i] != val){
                nums[count] = nums[i];
                count++;
            }
        }
        return count;
    }
};
/*
Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums containing 0, 0, 1, 3, and 4.
Note that the five elements can be returned in any order.
It does not matter what you leave beyond the returned k (hence they are underscores).
*/


//146. REMOVE DUPLICATES FROM SORTED ARRAY                                            {T.C = O(N), S.C = O(1)}
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int count = 1;
        int n = nums.size();
        if(n <= 1){
            return n;
        }
        for(int i = 1 ; i < nums.size() ; i++){
            if(nums[i] != nums[count-1]){
                nums[count] = nums[i];
                count++;
            }
        }
        return count;
    }
};
/*
Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
*/


//147. REMOVE DUPLICATES FROM SORTED ARRAY II                                {T.C = O(N), S.C = O(1)}
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int count = 2;
        int n = nums.size();
        if(n <= 2){
            return n;
        }
        for(int i = 2 ; i < n ; i++){
            if(nums[i] != nums[count-2]){
                nums[count] = nums[i];
                count++;
            }

        }
        return count;
    }
};
/*
Input: nums = [0,0,1,1,1,1,2,3,3]
Output: 7, nums = [0,0,1,1,2,3,3,_,_]
Explanation: Your function should return k = 7, with the first seven elements of nums being 0, 0, 1, 1, 2, 3 and 3 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
*/


//148. MAJORITY ELEMENT
//BRUTE FORCE (HASHMAP)                                                      {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int>mp;
        for(auto it : nums){
            mp[it]++;
        }
        int x = n/2;
        for(auto it : mp){
            if(it.second > x){
                return it.first;
            }
        }
        return -1;
    }
};

//OPTIMIZED APPROACH
//MOORE VOTING ALGORITHM                                                         {T.C = O(N), S.C = O(1)}
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int count = 0;
        int element = 0;

        for(auto it : nums){
            if(count == 0){
                element = it;
            }
            if(it == element){
                count++;
            }else{
                count--;
            }
        }
        return element;
    }
};
/*
Input: nums = [2,2,1,1,1,2,2]
Output: 2
*/


//149. ROTATE ARRAY
//BRUTE FORCE                                                                  {T.C = O(N), S.C = O(N)}
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        if(n == 0 || k == 0){              //size = 0 or k = 0 / same index do nothing
            return;
        }
        vector<int>rotatedArray(n);
        k = k % n ;                        //if k > n then take modulo to match index
        for(int i = 0 ; i < n ; i++){
            rotatedArray[(i+k) % n] = nums[i];
        }
        nums = rotatedArray;              //copy vector in ans
    }
};

//OPTIMIZED APPROACH                                                        {T.C = O(N), S.C = O(1)}
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        int n = nums.size();

        k = k % n;                         //if k > n then modulo for match index
                                                    //example 123456 , k = 3 => 456123
        reverse(nums.begin(), nums.begin()+(n-k));  //321456        
        reverse(nums.begin()+(n-k), nums.end());    //321654
        reverse(nums.begin(), nums.end());          //456123
    }
};
/*
Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]
*/


//150. H-INDEX                                                                  {T.C = O(N*LOGN), S.C = O(1)}
//BRUTE FORCE(WORKING SOLUTION)
class Solution {
public:
/* h value for a particular index will be mininmum value of the difference between the total number of publications and i th index and number of number of citations of the ith index.*/
    int hIndex(vector<int>& citations) {
        int n = citations.size();
        sort(citations.begin(), citations.end());
        int ans = 0;
        for(int i = 0 ; i < n ; i++){
            ans = max(ans, min(citations[i], (n-i)) );          
        }
        return ans;
    }
};

//OPTIMIZED APPROACH
//BUCKET SORT(FREQUENCY ARRAY)                                                  {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int hIndex(vector<int>& citations) {
        int n = citations.size();
        vector<int>bucket(n+1);

        for(int i = 0 ; i < n ; i++){
            if(citations[i] >= n){
                bucket[n]++;
            }else{
                bucket[citations[i]]++;
            }
        }

        int count = 0;
        for(int i = n ; i >= 0 ; i--){
            count += bucket[i];
            if(count >= i){
                return i;
            }
        }
        return 0;
    }
};
/*
Input: citations = [3,0,6,1,5]
Output: 3
Explanation: [3,0,6,1,5] means the researcher has 5 papers in total and each of them had received 3, 0, 6, 1, 5 citations respectively.
Since the researcher has 3 papers with at least 3 citations each and the remaining two with no more than 3 citations each, their h-index is 3.
*/


//151. INSERT DELETE GETRANDOM O(1)
class RandomizedSet {
    vector<int>v;                              //to store index
    unordered_map<int, int>mp;                      //<value, index>
public:
    RandomizedSet() {
        //no need
    }
    
    bool insert(int val) {
        if(mp.find(val) != mp.end()){
            return false;
        }
        //insert it in a array/vector
        v.push_back(val);
        mp[val] = v.size() - 1;

        return true;
    }
    
    bool remove(int val) {
        if(!mp.count(val)){                     //val not present in mp
            return false;
        }

        mp[v.back()] = mp[val];
        swap(v.back(), v[mp[val]]);
        v.pop_back();
        mp.erase(val);

        return true;
    }
    
    int getRandom() {
        return v[rand()% v.size()];
    }
};
/*
Input
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
Output
[null, true, false, true, 2, true, false, 2]

Explanation
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomizedSet.remove(2); // Returns false as 2 does not exist in the set.
randomizedSet.insert(2); // Inserts 2 to the set, returns true. Set now contains [1,2].
randomizedSet.getRandom(); // getRandom() should return either 1 or 2 randomly.
randomizedSet.remove(1); // Removes 1 from the set, returns true. Set now contains [2].
randomizedSet.insert(2); // 2 was already in the set, so return false.
randomizedSet.getRandom(); // Since 2 is the only number in the set, getRandom() will always return 2.
*/


//152. GAS STATION                                                               {T.C = O(N), S.C = O(1)}
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int totalDiff = 0, n = gas.size(), fuel= 0, index = 0; // or cost.size() both same
        for(int i = 0 ; i < n ; i++){
            int diff = gas[i]-cost[i];
            totalDiff += diff;
            fuel += diff;
            if(fuel < 0){               //for next search
                index = i+1;
                fuel = 0;
            }
        }
        return totalDiff < 0 ? -1 : index;
    }
};
/*
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3
Explanation:
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.
*/


//153. CANDY                                                                    {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int candy(vector<int>& ratings) {
        int n = ratings.size();
        if(n <= 1){
            return n;
        }
        vector<int>ans(n, 1);
        //left traversal
        for(int i = 1 ; i < n ; i++){
            if(ratings[i] > ratings[i-1]){
                ans[i] = ans[i-1] + 1;
            }
        }
        //right traversal
        for(int i = n-1 ; i > 0 ; i--){
            if(ratings[i] < ratings[i-1]){
                ans[i-1] = max(ans[i] + 1, ans[i-1]);         //compares both cases 
            }
        }
        int res = 0;
        for(int i = 0 ; i < n ; i++){
            res += ans[i];
        }
        return res;
    }
};
/*
Input: ratings = [1,2,2]
Output: 4
Explanation: You can allocate to the first, second and third child with 1, 2, 1 candies respectively.
The third child gets 1 candy because it satisfies the above two conditions.
*/


//154. TRAPPING RAIN WATER                                                        {T.C = O(N), S.C =  O(1)}
class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        int left = 0, right = n-1, leftMax = 0, rightMax = 0;
        int res = 0;
        while(left <= right){
            if(height[left] <= height[right]){
                if(height[left] > leftMax){
                    leftMax = height[left];
                }else{
                    res += leftMax - height[left];
                }
                left++;
            }else{
                if(height[right] > rightMax){
                    rightMax = height[right];
                }else{
                    res += rightMax - height[right];
                }
                right--;
            }
        }
        return res;
    }
};
/*
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. 
In this case, 6 units of rain water (blue section) are being trapped.
*/


//155. ZIGZAG CONVERSION                                                        {T.C = O(N), S.C = O(N)}
class Solution {
public:
    string convert(string s, int numRows) {
        vector<string>ans(numRows);                      //for storing pattern
        if(numRows == 1){
            return s;
        }
        bool flag = false;
        int i = 0;
        for(auto ch : s){
            ans[i] += ch;
            if(i == 0 || i == numRows-1){                 //switching direction up <-> down
                flag = !flag;
            }
            if(flag){
                i++;
            }else{
                i--;
            }
        }
        string zigzag = "";
        // for(int i = 0 ; i < ans.size() ; i++){
        //     zigzag += ans[i];
        // }
        for(auto it : ans){
            zigzag += it;
        }
        return zigzag;
    }
};
/*
Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:
P     I    N
A   L S  I G
Y A   H R
P     I
*/


//156. FIND THE INDEX OF THE FIRST OCCURANCE IN A STRING
//BRUTE FORCE (STL)
class Solution {
public:
    int strStr(string haystack, string needle) {
        return haystack.find(needle);
    }
};

//WITHOUT STL APPROACH                                                             {T.C = O(N), S.C = O(M)}
class Solution {
public:
    int strStr(string haystack, string needle) {
        int n = haystack.length();
        int m = needle.length();
        for(int i = 0 ; i < n-m+1 ; i++){              //no need to till last, check till from that where a valid substr can be made
            if(haystack[i] == needle[0]){
                string sub = haystack.substr(i, needle.length());     //substr(initial, length)
                if(sub == needle){
                    return i;
                }
            }
        }
        return -1;
    }
};
/*
Input: haystack = "leetcode", needle = "leeto"
Output: -1
Explanation: "leeto" did not occur in "leetcode", so we return -1.
*/


//157. TEXT JUSTIFICATION                                                          {T.C = O(N), S.C = O(N)}
class Solution {
public:
    vector<string> fullJustify(vector<string>& words, int maxWidth) {
        vector<string>ans;
        int n = words.size();
        int i = 0;
        while(i < n){
            int lineLen = words[i].size();
            int j = i+1;
            while(j < n && lineLen + words[j].size() + (j-i) <= maxWidth){          //keep adding words untill line exceeds maxWidth
                lineLen += words[j].size();
                j++;
            }
            int numOfWords = j-i;
            int numOfSpace = maxWidth - lineLen;
            //for constructing justified line
            string line;
            //handle only 1 word or last word
            if(numOfWords == 1 || j == n){
                line = words[i];
                for(int k = i+1 ; k < j ; k++){
                    line += ' ' + words[k];
                }
                line += string(maxWidth-line.size(), ' ');    // Add spaces to reach maxWidth.
            }else{
                int spaceBwWords = numOfSpace / (numOfWords - 1);
                int extraSpace = numOfSpace % (numOfWords-1);

                line = words[i];
                for(int k = i+1; k < j ; k++){
                    line += string(spaceBwWords, ' ');
                    if(extraSpace > 0){
                        line += ' ';                         // Add extra space if available.
                        extraSpace--;
                    }
                    line += words[k];
                }
            }
            ans.push_back(line);
            i = j;                                          // Move to the next set of words.
        }
        return ans;
    }
};
/*
Input: words = ["What","must","be","acknowledgment","shall","be"], maxWidth = 16
Output:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
Explanation: Note that the last line is "shall be    " instead of "shall     be", because the last line must be left-justified instead of fully-justified.
Note that the second line is also left-justified because it contains only one word.
*/


//158. VALID PALINDROME 
//BRUTE FORCE APPROACH                                                                     {T.C = O(N), S.C = O(N)}
class Solution {
public:
    bool isPalindrome(string s) {
        int n = s.size();
        string ans;
        for(int i = 0 ; i < n; i++){
            if(s[i] >= 'A' && s[i] <= 'Z'){
                s[i] = tolower(s[i]);
            }
            if((s[i] >= 'a' && s[i] <= 'z') || (s[i] >= '0' && s[i] <= '9')){
                ans += s[i];
            }else{
                continue;
            }
        }
        int m = ans.size();
        for(int i = 0 ; i < m/2; i++){
            if(ans[i] != ans[(m-1)-i]){
                return false;
            }
        }
        return true;
    }
};

//USING TWO POINTERS                                                         {T.C = O(N), S.C = O(1)}
class Solution {
public:
    bool isPalindrome(string s) {
        int n = s.size();
        int i = 0;
        int j = n-1;
        while(i <= j){
            if(!isalnum(s[i])){
                i++;
            }
            else if(!isalnum(s[j])){
                j--;
            }else{
                if(tolower(s[i]) != tolower(s[j])){
                    return false;
                }else{
                    i++;
                    j--;
                }
            }
        }
        return true;
    }
};
/*
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
*/


//159 SLIDING WINDOW MAXIMUM
//BRUTE FORCE  (TLE)                                                                     {T.C = O(N^2), S.C = O(N)}
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int>ans;
        for(int i = 0 ; i <= n-k ; i++){
            int maxi = INT_MIN;
            for(int j = i ; j < i+k ; j++){
                maxi = max(maxi, nums[j]);
            }
            ans.push_back(maxi);
        }
        return ans;
    }
};

//OPTIMIZED APPROACH USING DEQUEUE                                                 {T.C = O(N), S.C = O(N)}
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int>ans;
        deque<int>dq;                        //store ans in decreasing order
        //for first window
        for(int i = 0 ; i < k ; i++){
            while(!dq.empty() && dq.back() < nums[i]){
                dq.pop_back();
            }
            dq.push_back(nums[i]);
        }
        ans.push_back(dq.front());
        //for all next windows
        for(int i = k ; i < n ; i++){
            //remove previous element
            if(dq.front() == nums[i-k]){
                dq.pop_front();
            }

            //same as above
            while(!dq.empty() && dq.back() < nums[i]){
                dq.pop_back();
            }
            dq.push_back(nums[i]);
            ans.push_back(dq.front());                  //push all top elements
        }
        return ans;
    }
};
/*
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
 */


//160. MINIMUM SIZE SUBARRAY SUM                                                   {T.C = O(N), S.C = O(1)}
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int n = nums.size();
        int i = 0, j = 0;
        int sum = 0;
        int mini = INT_MAX;                          //length of window
        while(j < n){
            sum += nums[j];                          //calculation
            while(sum >= target){
                mini = min(mini, j-i+1);             //calculation
                sum -= nums[i];                      //slide window
                i++;
            }
            j++;
        }
        return mini == INT_MAX ? 0 : mini;
    }
};
/*
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.
*/


//161. SUBSTRING WITH CONCATENATION OF ALL WORDS                                      {T.C = O(N*M), S.C = O(N)}
//COPY PASTE
class Solution {
public:
    vector<int> findSubstring(string s, vector<string>& words) {
        vector<int> ans;
        int n = words.size();   // Number of words
        int m = words[0].size(); // Length of each word
        int len = n * m;        // Total length of all words
        unordered_map<string, int> mp;

        // Count the frequency of each word
        for (const string& word : words) {
            mp[word]++;
        }

        if (len > s.size()) {
            return ans;
        }

        for (int i = 0; i < m; i++) {
            unordered_map<string, int> mp2;
            int left = i;

            for (int j = i; j <= s.size() - m; j += m) { // Changed i to j here
                string temp = s.substr(j, m);

                if (mp.find(temp) != mp.end()) {
                    mp2[temp]++;
                    while (mp2[temp] > mp[temp]) {
                        mp2[s.substr(left, m)]--;
                        left += m;
                    }

                    if (j - left + m == len) {
                        ans.push_back(left);
                        mp2[s.substr(left, m)]--;
                        left += m;
                    }
                } else {
                    mp2.clear();
                    left = j + m;
                }
            }
        }
        return ans;
    }
};
/*
Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
Explanation: Since words.length == 2 and words[i].length == 3, the concatenated substring has to be of length 6.
The substring starting at 0 is "barfoo". It is the concatenation of ["bar","foo"] which is a permutation of words.
The substring starting at 9 is "foobar". It is the concatenation of ["foo","bar"] which is a permutation of words.
The output order does not matter. Returning [9,0] is fine too.
*/


//162. VALID SUDOKU                                                               {T.C = O(1), S.C = O(1)}
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        unordered_set<string>st;
        for(int i = 0 ; i < 9 ; i++){                    //i == row, j == col
            for(int j = 0 ; j < 9 ; j++){
                int boxIndex = (i/3)*3 + j/3;
                if(board[i][j] != '.'){
                    string row = "r" + to_string(i) + board[i][j];
                    string col = "c" + to_string(j) + board[i][j];
                    string box = "b" + to_string(boxIndex) + board[i][j];
                    //already string is present in set
                    if(st.count(row) || st.count(col) || st.count(box)){
                        return false;
                    }
                    st.insert(row);
                    st.insert(col);
                    st.insert(box);
                }
            }
        }
        return true;
    }
};
/*
Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true
*/


//163. GAME OF LIFE                                                          {T.C = O(N*M), S.C = O(1)}
//APPROACH
//store all cordinates in array then check 4 cases according to quesiton
class Solution {
public:
    bool isValidNbr(int x, int y , vector<vector<int>>&board){
        return ( (x >= 0 && x < board.size()) && (y >= 0 && y < board[0].size()) );
    }
    void gameOfLife(vector<vector<int>>& board) {
        int n = board.size();
        int m = board[0].size();
        //all 8 coordinates
        int delRow[8] = {0, 1, 0, -1, 1, 1, -1, -1};
        int delCol[8] = {1, 0, -1, 0, -1, 1, 1, -1};

        for(int row = 0; row < n ; row++){
            for(int col = 0; col < m ; col++){
                int countNbr = 0;

                for(int i = 0 ; i < 8 ; i++){
                    int currRow = row + delRow[i];
                    int currCol = col + delCol[i];
                    
                    if(isValidNbr(currRow, currCol, board) && abs(board[currRow][currCol]) == 1){
                        countNbr++;
                    }
                }
                if(board[row][col] == 1 && (countNbr < 2 || countNbr > 3)){
                    board[row][col] = -1;
                }else if(board[row][col] == 0 && countNbr == 3){
                    board[row][col] = 2;
                }
            }
        } 

        for(int row = 0 ; row < n ; row++){
            for(int col = 0; col < m ; col++){
                if(board[row][col] >= 1){
                    board[row][col] = 1;
                }else{
                    board[row][col] = 0;
                }
            }
        }       
    }
};
/*
Input: board = [
                [0,1,0],
                [0,0,1],
                [1,1,1],
                [0,0,0]
               ]
Output: [ 
         [0,0,0],
         [1,0,1],
         [0,1,1],
         [0,1,0]
        ]
*/


//164. RANSOME NOTE
//APPROACH 1
//USING UNORDERED MAP / HASH MAP                                       {T.C = O(N), S.C = O(N)}
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        unordered_map<char, int>mp;
        for(auto ch : magazine){
            mp[ch]++;
        }
        for(auto ch : ransomNote){
            if(mp.find(ch) != mp.end() && mp[ch] > 0){
                mp[ch]--;                      // Decrement the count of the character
            } else {
                return false;                  // Not enough characters in the magazine
            }
        }
        return true;                           // All characters in ransomNote can be constructed
    }
};

//APPROACH 2(OPTIMIZED)
//using constant size vector                                          {T.C = O(N), S.C = O(1)}
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        vector<int>v(26);

        for(auto ch : magazine){
            v[ch-97]++;
        }
        for(auto ch : ransomNote){
            v[ch-97]--;
        }
        
        for(auto it : v){
            if(it < 0){
                return false;
            }
        }
        return true;
    }
};
/*
Input: ransomNote = "aa", magazine = "aab"
Output: true
*/


//165. ISOMORPHIC STRINGS                                                         {T.C = O(N), S.C = O(N)}
//APPROACH
//first fill map (check already present char & replace char) then check correct mapping
class Solution {
public:
    bool isIsomorphic(string s, string t) {
        unordered_map<char,char>mp;
        unordered_set<int>st;

        //base case
        if(s.length() != t.length()){
            return false;
        }

        for(int i = 0 ; i < s.length() ; i++){
            if(mp.find(s[i]) == mp.end()){            //if original is not already present in map / fill map
                if(st.find(t[i]) == st.end()){        //replacement char also not present
                    mp[s[i]] = t[i];                  //mapping original->replacement
                    st.insert(t[i]);
                }else{
                    return false;
                }
            }else{  //mp.find(s[i] != mp.end())      //orginal is already present in map
                if(mp[s[i]] != t[i]){                //check valid mapping
                    return false;
                }                       
            } 
        }
        return true;
    }
};
/*
Input: s = "egg", t = "add"
Output: true
*/


//166. WORD PATTERN                                                                 {T.C = O(N+M), S.C = O(N+M)}
//APPROACH
//first divide long string into string array then map each char to string then check for valid mapping 
class Solution {
public:
    bool wordPattern(string pattern, string s) {
        vector<string>v;
        string temp = "";
        for(int i = 0 ; i < s.length() ; i++){
            if(s[i] == ' '){
                v.push_back(temp);
                temp = "";                            //reset temp string
            }else{
                temp += s[i];
            }
        }
        v.push_back(temp);

        //base case
        if(pattern.length() != v.size()){
            return false;
        }

        unordered_map<char ,string>mp;
        unordered_set<string>st;

        for(int i = 0 ; i < pattern.length() ; i++){
            if(mp.find(pattern[i]) == mp.end()){          //if original is not present in map
                if(st.find(v[i]) == st.end()){            //replace char(string) is also not present
                    mp[pattern[i]] = v[i];                //mapping pattern->s
                    st.insert(v[i]);                      
                }else{
                    return false;
                }
            }else{//mp.find(pattern[i]) != mp.end()       //if original is present in map
                if(mp[pattern[i]] != v[i]){               //check for valid mapping
                    return false;
                }
            }
        }
        return true;
    }
};
/*
Input: pattern = "abba", s = "dog cat cat dog"
Output: true
*/


//167. HAPPY NUMBER                                                               {T.C = O(LOGN), S.C = O(LOGN)}
//APPROACH
//run infinite loop store sum in set (unique elements only) if duplicate entry return false else find sum = 1 then true
class Solution {
public:
    bool isHappy(int n) {
        unordered_set<int>st;
        while(true){                                         //infinite loop
            int sum = 0;
            while(n != 0){
                sum += pow(n%10 , 2);
                n = n/10;                                        //extract digit next operation on rest of digits
            }

            if(sum == 1){
                return true;
            }

            n = sum;               //update number
            if(st.find(n) != st.end()){                     //set have already present (infinit loop)
                return false;
            }
            st.insert(n);
        }
    }
};
/*
Input: n = 19
Output: true
Explanation:
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
*/


//168. CONTAINS NUMBER II
//BRUTE FORCE                                                                    {T.C = O(N^2), S.C = O(1)}
//use 2 for loop and match condition(nums[i] == nums[j] and abs(i - j) <= k) according to question
class Solution {
public:
    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        int n = nums.size();
        for(int i = 0 ; i < n ; i++){
            for(int j = i+1 ; j < n ; j++){
                if(nums[i] == nums[j] && abs(i-j) <= k){
                    return true;
                }
            }
        }
        return false;
    }
};

//OPTIMIZED APPROACH                                                            {T.C = O(N), S.C = O(N)}
//using unordered map 
class Solution {
public:
    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        int n = nums.size();
        unordered_map<int,int>mp;

        for(int i = 0 ; i < n ; i++){
            if(mp.count(nums[i])){                            //duplicate element found
                if(abs(i-mp[nums[i]]) <= k){
                    return true;
                }
            }
            mp[nums[i]] = i;                                   //mapping element with index
        }
        return false;
    }
};
/*
Input: nums = [1,2,3,1], k = 3
Output: true
*/


//169. SUMMARY RANGES                                                       {T.C = O(N), S.C = O(N)}
class Solution {
public:
    vector<string> summaryRanges(vector<int>& nums) {
        int n = nums.size();
        vector<string>ans;

        string temp = "";

        int i = 0, j = 0;
        for(i = 0 ; i < n ; i++){
            j = i;             //i is used to keep track of the beginning of a potential range, and j is used to find the end of that range while iterating through the consecutive numbers. 
            while(j + 1 < n && nums[j+1] == nums[j] + 1){
                j++;
            }
            if(j > i){
                temp += to_string(nums[i]);
                temp += "->";
                temp += to_string(nums[j]);
            }else{
                temp += to_string(nums[i]);
            }

            ans.push_back(temp);
            temp = "";
            i = j;                                       //move i to j for fresh start (next range)
        }
        return ans;
    }
};
/*
Input: nums = [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]
Explanation: The ranges are:
[0,2] --> "0->2"
[4,5] --> "4->5"
[7,7] --> "7"
*/


//170. SIMPLIFY PATH                                                              {T.C = O(N), S.C = O(N)}
//APPROACH
//take a stack and res string traverse string check for following condition
/*
The path starts with a single slash '/'.
Any two directories are separated by a single slash '/'.
The path does not end with a trailing '/'.
The path only contains the directories on the path from the root directory to the target file or directory (i.e., no period '.' or double period '..')
Return the 
*/
//remember res poped element is reverse answer so  (/+st.top()+res)
class Solution {
public:
    string simplifyPath(string path) {
        stack<string>st;
        string res;

        for(int i = 0 ; i < path.size() ; i++){
            if(path[i] == '/'){
                continue;
            }
            string temp;
            while(i < path.size() && path[i] != '/'){
                temp += path[i];
                i++;
            }
            if(temp == "."){
                continue;
            }
            else if(temp == ".."){
                if(!st.empty()){
                    st.pop();   
                }
            }else{
                st.push(temp);
            }
        }
        while(!st.empty()){
            res = '/'+ st.top()+ res;
            st.pop();
        }

        if(res.size() == 0){
            return "/";
        }
        return res;
    }
};
/*
Input: path = "/home/"
Output: "/home"
Explanation: Note that there is no trailing slash after the last directory name.

Input: path = "/../"
Output: "/"
Explanation: Going one level up from the root directory is a no-op, as the root level is the highest level you can go.

Input: path = "/home//foo/"
Output: "/home/foo"
Explanation: In the canonical path, multiple consecutive slashes are replaced by a single one.
*/


//171. MIN STACK                                                                             {T.C = O(1), S.C = O(N)}
//APPROACH
/*
Initialize two stacks, arr to store the actual elements and min_arr to store the minimum elements.
When pushing an element from the stack:
Push the element into the arr stack.
Check if the min_arr stack is not empty. If it is, simply push the element into the min_arr stack.
If the min_arr stack is not empty, calculate the minimum between the current element and the top of the min_arr stack and 
push the minimum value into the min_arr stack. This ensures that min_arr always contains the minimum value for the stack.

When popping an element from the stack:
Pop the element from both the arr and min_arr stacks. This maintains the consistency of the two stacks.

To get the top element of the stack, simply return the top element from the arr stack.

To get the minimum value of the stack, return the top element from the min_arr stack. This provides a constant time operation for retrieving the minimum.
*/
class MinStack {
public:
    stack<int>s, b;                      //s = stores original elements, b = stores minimum elements
    MinStack() {}
    
    void push(int val) {
        s.push(val);
        if(!b.empty()){
            val = min(val, b.top());
        }
        b.push(val);
    }
    
    void pop() {                                     //for maintaining consistancy pop from both
        s.pop();
        b.pop();
    }
    
    int top() {
        return s.top();
    }
    
    int getMin() {
        return b.top();
    }
};
/*
Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
*/


//172. EVALUATE REVERSE POLISH NOTATION                                                 {T.C = O(N), S.C = O(N)}
//APPROACH
//simple implement postfix operation 
class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int>st;
        for(auto i : tokens){
            if(i == "+" || i == "-" || i == "*" || i == "/"){
                int op2 = st.top();                        //in postfix first pop element = op2
                st.pop();
                int op1 = st.top();
                st.pop();

                if(i == "+"){
                    st.push(op1 + op2);
                }
                if(i == "-"){
                    st.push(op1 - op2);
                }
                if(i == "*"){
                    st.push(op1 * op2);
                }
                if(i == "/"){
                    st.push(op1 / op2);
                }
            }else{
                int data = stoi(i);                            //string to integer
                st.push(data);
            }
        }
        return st.top();
    }
};
/*
Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
Output: 22
Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
*/


//173. BASIC CALCULATOR                                                              {T.C = O(N), S.C = O(N)}
//APPROACH
//take 2 stacks (1 = integer, 2 = sign), now calculate ans
class Solution {
public:
int calculate(string s) {
            
    int value = 0;                // Current integer value
    int res = 0;                // Running result
    int sign = 1;
    stack<int> st;              // Stores intermediate results
    stack<int> stSign;          // Stores signs

    
    for(char c : s){
        if(c >= '0' && c <= '9'){
            value = value* 10 + (c-'0');
        }else if(c == '+'){
            res += sign * value;
            value = 0;
            sign = 1;
        }else if(c == '-'){
            res += sign * value;
            value = 0;
            sign = -1;
        }else if(c == '('){
            st.push(res);
            stSign.push(sign);

            res = 0; 
            sign = 1; 
        }else if(c == ')'){
            res += sign * value;
            res *= stSign.top(); stSign.pop();
            res += st.top(); st.pop();
            value = 0; 

        }
    }
    return res + sign * value;
}
};
/*
Input: s = "(1+(4+5+2)-3)+(6+8)"
Output: 23
*/


//ALWAYS USE DUMMY NODE WHERE WE HAVE TO DISLINK OR DELETE THE NODE
//174. ADD 2 NUMBERS                                                                    {T.C = O(N), S.C = O(MAX(N,M)}
class Solution {
public:
    /* already reverse order have given in question
    ListNode* reverse(ListNode* head){
        ListNode* prev = NULL;
        ListNode* curr = head;
        ListNode* forward = NULL;

        while(curr){                                  //curr == curr != NULL
            forward = curr->next;
            curr->next = prev;
            prev = curr;
            curr = forward;
        }
        return prev;
    }
    */
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        // l1 = reverse(l1);
        // l2 = reverse(l2);
        int carry = 0;
        ListNode *dummy = new ListNode(NULL);
        ListNode *temp = dummy;

        while(l1 || l2 || carry){
            int sum = 0;
            if(l1){
                sum += l1->val;
                l1 = l1->next;
            }
            if(l2){
                sum += l2->val;
                l2 = l2->next;
            }
            sum += carry;
            carry = sum/10;
            ListNode* extra = new ListNode(sum % 10);
            temp->next = extra;
            temp = temp->next;
        }
        // return reverse(dummy->next);
        return dummy->next;                        //actual result
    }
};
/*
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.
*/


// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
//175. COPY LIST WITH RANDOM POINTERS                                                 {T.C = O(N), S.C = O(N)}
//APPROACH
//first copy the original ll to new ll then stores its value and random pontier in unordered map then with the help
//of map we put random pointer to new ll
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(!head){
            return head;
        }
        Node* newHead = new Node(0);
        Node* newCurr = newHead;
        Node* curr = head;

        unordered_map<Node*, Node*>mp;
        //copying simple pointers or ll to new ll
        while(curr){
            Node* temp = new Node(curr->val);
            mp.insert({curr, temp});                        //{oldNode, newNode}

            newCurr->next = temp;
            newCurr = newCurr->next;
            curr = curr->next;  
        }

        //reset
        curr = head;
        newCurr = newHead->next;

        //copying random pointers
        while(curr){
            Node* random = curr->random;
            Node* newNode = mp[random];
            newCurr->random = newNode;

            newCurr = newCurr->next;
            curr = curr->next;
        }

        return newHead->next;
    }
};
/*
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
*/


//176. REVERSE A LINKED LIST II                                                                  {T.C = O(N), S.C = O(1)}
//APPROACH
//divide in 3 subproblem first traverse till left then reverse left to right at last join remaining from right to last in ll
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {

        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* temp = dummy;

        ListNode* prev = NULL;
        ListNode* curr = head;
        ListNode* forward = NULL;

        //move to before reversing position
        for(int i = 0 ; i < left-1 ; i++){
            temp = temp->next;
            curr = curr->next;
        }

        //from where we start reversing
        ListNode* subListed = curr;

        //same logic as reverse ll
        for(int i = 0 ; i < right-left+1 ; i++){
            forward = curr->next;
            curr->next = prev;
            prev = curr;
            curr = forward;
        }

        //join the pieces (remaining node after {right-left+1} nodes)
        temp->next = prev;
        subListed->next = curr;

        return dummy->next;
    }
};
/*
Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]
*/


//177. REVERSE NODES IN K-GROUP                                                       {T.C = O(N), S.C = O(1)}
//APPROACH
//simple recursive solution first reverse k nodes then put recursion
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        //if size is less then k then not reverse ll
        int sz = 0;
        ListNode* temp = head;

        while(temp){
            temp = temp->next;
            sz++;
        }
        //---------------------------------------------

        //base case
        if(!head || sz < k){
            return head;
        }

        ListNode* prev = NULL;
        ListNode* curr = head;
        ListNode* forward = NULL;
        int count = 0;
        //reverse 1 k elements
        while(curr && count < k){
            forward = curr->next;
            curr->next = prev;
            prev = curr;
            curr = forward;
            count++;
        }

        //after k elements lets recursion do its work
        if(forward){
            head->next = reverseKGroup(forward, k);         //point recursive ans to head->next
        }
        return prev;
    }
};
/*
Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]
*/


//178. REMOVE DUPLICATES FROM SORTED LINKED LIST II                                     {T.C = O(N), S.C = O(1)}
//APPROACH
//make a dummy node (for dislinking the link) then untill a val is same remove pointers after that simple return dummy->next
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* temp = dummy;

        while(head){
            if(head->next && head->val == head->next->val){
                while(head->next && head->val == head->next->val){
                    head = head->next;
                    temp->next = head->next;                        //removing link
                }
            }else{
                temp = temp->next;
            }
            head = head->next;
        }
        return dummy->next;
    }
};
/*
Input: head = [1,2,3,3,4,4,5]
Output: [1,2,5]
*/


//179. ROTATE LIST                                                                  {T.C = O(N), S.C = O(1)}
//APPROACH
//first find len of list then effective rotation (k % n) from start(k = n-k) traversal find new head and update it
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if(!head || k == 0){
            return head;
        }
        ListNode* curr = head;        
        int len = 1;                            // Initialize the length of the list.

        while(curr->next != NULL){
            curr = curr->next;
            len++;
        }
        curr->next = head;

        k = k % len;                           //effective rotation
        k = len - k;

        
        while (k--) {                         // Traverse the list to find the new head after rotation.
            curr = curr->next;
        }

        // Update the head and break the circular connection to complete the rotation.
        head = curr->next;
        curr->next = nullptr;

        return head;
    }
};
/*
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]
*/


//180. PARTITION LIST                                                           {T.C = O(N), S.C = O(1)}
//APPROACH
//take 2 dummy nodes or list 1st list have have store smaller element then x 2nd list rest of element then return ll
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode* list1 = new ListNode(0);
        ListNode* list2 = new ListNode(0);
        ListNode* l1 = list1;
        ListNode* l2 = list2;

        while(head){
            if(head->val < x){
                l1->next = head;
                l1 = l1->next;
            }else{
                l2->next = head;
                l2 = l2->next;
            }
            head = head->next;
        }
        
        l1->next = list2->next;                  //join 1 and 2 second list
        l2->next = NULL;                         //point last node to null

        return list1->next;                      //list1 = 0, list1->next = actual head
    }
};
/*
Input: head = [1,4,3,2,5,2], x = 3
Output: [1,2,2,4,3,5]

Example 2:
Input: head = [2,1], x = 2
Output: [1,2]
*/


//181. LRU CACHE                                                            {T.C = O(1), S.C = O(CAP)}
//APPROACH
//The LRUCache class uses a doubly linked list and a hash map to implement a fixed-size cache, allowing for efficient O(1)
//retrieval and insertion of key-value pairs while maintaining the least recently used (LRU) item at the tail of the list. 
//When the cache reaches its capacity, it removes the LRU item to accommodate new entries.
class LRUCache {
public:
    class Node {
    public:
        int key;
        int val;
        Node* prev;
        Node* next;

        Node(int key, int val) {
            this->key = key;
            this->val = val;
        }
    };

    Node* head = new Node(-1, -1);  // Head sentinel node with minimum values
    Node* tail = new Node(-1, -1);  // Tail sentinel node with minimum values

    int cap;                        // Maximum capacity of the cache
    unordered_map<int, Node*> mp;   // A map to store key-node pairs for quick access

    LRUCache(int capacity) {
        cap = capacity;
        head->next = tail;           // Initialize the linked list with head and tail sentinels
        tail->prev = head;
    }

    // Function to delete a specific node from the linked list
    void deleteNode(Node* delNode) {
        Node* delPrev = delNode->prev;
        Node* delNext = delNode->next;

        delPrev->next = delNext;
        delNext->prev = delPrev;
    }

    // Function to add a new node to the front of the linked list
    void addNode(Node* newNode) {
        Node* temp = head->next;
        newNode->next = temp;
        newNode->prev = head;

        head->next = newNode;
        temp->prev = newNode;
    }

    // Get the value for a given key and move the corresponding node to the front
    int get(int key) {
        if (mp.find(key) != mp.end()) {
            Node* ansNode = mp[key];
            int ans = ansNode->val;

            deleteNode(ansNode);
            addNode(ansNode);

            mp[key] = head->next;  // Update the map with the new position of the node
            return ans;
        }
        return -1;  // Key not found in the cache
    }

    // Put a new key-value pair in the cache, possibly removing the least recently used item
    void put(int key, int value) {
        if (mp.find(key) != mp.end()) {
            Node* curr = mp[key];
            deleteNode(curr);  // Remove the existing node from its current position
        }
        if (mp.size() == cap) {
            mp.erase(tail->prev->key);  // Remove the least recently used item
            deleteNode(tail->prev);
        }
        addNode(new Node(key, value));  // Add the new node to the front
        mp[key] = head->next;           // Update the map with the new position of the node
    }
};
/*
Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4
*/


//182. SYMMETRIC TREE                                                     {T.C =  O(N), S.C = O(H)}
//APPROACH
//check if null then true then check val is same then isMiror function by recursion checks symmetric or not
//similar to 69 quesition
class Solution {
public:
    bool isMirror(TreeNode* p, TreeNode* q){
        if(!p && !q){
            return true;
        }
        if(!p || !q){
            return false;
        }
        return p->val == q->val && isMirror(p->left, q->right) && isMirror(p->right, q->left);   
    }
    bool isSymmetric(TreeNode* root) {
        if(!root){
            return true;
        }
        return isMirror(root->left, root->right);
    }
};
/*
Example 1:
Input: root = [1,2,2,3,4,4,3]
Output: true

Example 2:
Input: root = [1,2,2,null,3,null,3]
Output: false*/


class Node {
public:
    int val;
    Node* left;
    Node* right;
    Node* next;

    Node() : val(0), left(NULL), right(NULL), next(NULL) {}

    Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

    Node(int _val, Node* _left, Node* _right, Node* _next)
        : val(_val), left(_left), right(_right), next(_next) {}
};
//183. POPULATING NEXT RIGHT POINTERS IN EACH NODE II                        {T.C = O(N), S.C = O(N)}
class Solution {
public:
    Node* connect(Node* root) {
        //base case
        if(!root){
            return root;                     //return root or null
        }
        queue<Node*>q;
        q.push(root);

        while(!q.empty()){
            int n = q.size();
            vector<Node*> v(n, 0);
            for(int i = 0 ; i < n ; i++){              //LOT(level order traversal)
                Node* temp = q.front();
                q.pop();
                if(temp->left){
                    q.push(temp->left);
                }
                if(temp->right){
                    q.push(temp->right);
                }
                v[i] = temp;
            }
            for(int i = 0 ; i < n-1; i++){                    //move till second last element cause last element points to null
                v[i]->next = v[i+1];
            }
            v[n-1]->next = NULL;
        }
        return root;
    }
};
/*
Example 1:
Input: root = [1,2,3,4,5,null,7]
Output: [1,#,2,3,#,4,5,7,#]
Explanation: Given the above binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.

Example 2:
Input: root = []
Output: []
*/


//184. FLATTEN BINARY TREE TO LINKED LIST                              {T.C = O(N), S.C = O(1)}
//APPROACH
//using moris traversal (S.C = o(1))
class Solution {
public:
    void flatten(TreeNode* root) {
        TreeNode* curr = root;
        while(curr){
            if(curr->left){
                TreeNode* predecessor = curr->left;        //first left then extream right 
                while(predecessor->right){
                    predecessor = predecessor->right;
                }
                predecessor->right = curr->right;       //set virtual pointer
                curr->right = curr->left;
                curr->left = NULL;
            }
            curr = curr->right;
        }
    }
};
/*
Input : 
          1
        /   \
       2     5
      / \     \
     3   4     6
Output :
1 2 3 4 5 6 
Explanation: 
After flattening, the tree looks 
like this
    1
     \
      2
       \
        3
         \
          4
           \
            5
             \
              6 
Here, left of each node points 
to NULL and right contains the 
next node in preorder.The inorder 
traversal of this flattened tree 
is 1 2 3 4 5 6.
*/


//185. PATH SUM                                                    {T.C = O(N), S.C = O(H)}
//APPROACH 
//simple recursion 
class Solution {
public:
    bool totalSum(TreeNode* root, int targetSum, int currSum){
        if(!root->left && !root->right){
            if(targetSum == (currSum + root->val)){
                return true;
            }
            return false;
        }
        if(root->left){
            if(totalSum(root->left, targetSum, currSum + targetSum)){
                return true;
            }
        }
        if(root->right){
            if(totalSum(root->right, targetSum, currSum + targetSum)){
                return true;
            }
        }
        return false;
    }
    bool hasPathSum(TreeNode* root, int targetSum) {
        int currSum = 0;
        if(!root){
            return false;
        }
        return totalSum(root, targetSum, currSum);
    }
};
/*
Example 1:
Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
Output: true
Explanation: The root-to-leaf path with the target sum is shown.

Example 2:
Input: root = [1,2,3], targetSum = 5
Output: false
Explanation: There two root-to-leaf paths in the tree:
(1 --> 2): The sum is 3.
(1 --> 3): The sum is 4.
There is no root-to-leaf path with sum = 5.
*/


//186. SUM ROOT TO LEAF NUMBERS                                                   {T.C = O(N), S.C = O(H)}
//APPROACH
//first take sum of leaf node then traverse left and right subtree after that n/10 for backtracking
class Solution {
public:
    void solve(TreeNode* root, int &sum , int &num){
        //base case
        if(!root){
            return;
        }

        //leaf node
        if(!root->left && !root->right){
            sum += root->val + num*10;            //multiplied by 10 to shift its digits to the left
            return;
        }
        num = num*10 + root->val;                  //continuous updating num
        solve(root->left, sum, num);
        solve(root->right, sum, num);
        num /= 10;                            //to backtrack and explore other branches.

    }
    int sumNumbers(TreeNode* root) {
        int sum = 0, num = 0;
        solve(root, sum, num);
        return sum;
    }
};
/*
Example 1:
Input: root = [1,2,3]
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.

Example 2:
Input: root = [4,9,0,5,1]
Output: 1026
Explanation:
The root-to-leaf path 4->9->5 represents the number 495.
The root-to-leaf path 4->9->1 represents the number 491.
The root-to-leaf path 4->0 represents the number 40.
Therefore, sum = 495 + 491 + 40 = 1026.
*/


//187. BINARY SEARCH TREE ITERATOR                                                  {T.C = O(1), S.C = O(H)}
class BSTIterator {
    stack<TreeNode*>st;
public:
    BSTIterator(TreeNode* root) {
        //push all left node in the stack
        while(root){
            st.push(root);
            root = root->left;
        }
    }
    
    int next() {
        TreeNode* currNode = st.top();
        st.pop();
        if(currNode->right){             // If the current node has a right subtree, explore its leftmost path
            TreeNode* temp = currNode->right;
            while(temp){
                st.push(temp);
                temp = temp->left;
            }
        }
        return currNode->val;
    }
    
    bool hasNext() {
        if(!st.empty()){
            return true;
        }
        return false;
    }
};
/*

Example 1:
Input
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
Output
[null, 3, 7, true, 9, true, 15, true, 20, false]
Explanation
BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
bSTIterator.next();    // return 3
bSTIterator.next();    // return 7
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 9
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 15
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 20
bSTIterator.hasNext(); // return False
*/


//188. COUNT COMPLETE TREE NODES                                                    {T.C = O(N), S.C = O(1)}
class Solution {
public:
    int countNodes(TreeNode* root) {
        if(!root){
            return 0;
        }
        //LRN
        int left = countNodes(root->left);
        int right = countNodes(root->right);

        return left + right + 1;                    //1 = node element
    }
};
/*
Example 1:
Input: root = [1,2,3,4,5,6]
Output: 6

Example 2:
Input: root = []
Output: 0
*/


//189. BINARY TREE ZIGZAG LEVEL ORDER TRAVERSAL                                      {T.C = O(N), S.C = O(N)}
//APPROACH
//using level order traversal just take an bool parameter when traverse 1 level reverse then reverse traversal row wise
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>>ans;
        //base case
        if(!root){
            return ans;
        }

        bool leftToRight = true;
        queue<TreeNode*>q;
        q.push(root);
        while(!q.empty()){
            int lvlSize = q.size();
            vector<int>lvlNodes;
            for(int i = 0 ; i < lvlSize ; i++){
                TreeNode* temp = q.front();
                q.pop();
                //normal insert and reverse insert
                int idx = leftToRight ? i : lvlSize - i - 1;
                lvlNodes.push_back(temp->val);
                if(temp->left){
                    q.push(temp->left);
                }
                if(temp->right){
                    q.push(temp->right);
                }
            }
            //level direction change
            if(!leftToRight){
                reverse(lvlNodes.begin(), lvlNodes.end());
            }
            ans.push_back(lvlNodes);
            leftToRight = !leftToRight;
        }
        return ans;
    }
};
/*
Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[20,9],[15,7]]

Example 2:
Input: root = [1]
Output: [[1]]
*/


//190. MINIMUM ABSOLUTE DIFFERENCE IN BST                                            {T.C = O(N), S.C = O(H)}
//APPROACH
//Store preVal first then find minDiff by subtracting each preVal to current value.
class Solution {
public:
    void solve(TreeNode* root, int &mini, int &preVal){
        //base case
        if(!root){
            return;
        }
        //LNR      //BST
        solve(root->left, mini, preVal);

        if(preVal != -1){
            mini = min(mini, root->val - preVal);
        }
        preVal = root->val;
        
        solve(root->right, mini, preVal);
    }
    int getMinimumDifference(TreeNode* root) {
        int mini = INT_MAX;
        int preVal = -1;
        solve(root, mini, preVal);
        return mini;
    }
};
/*
Example 1:
Input: root = [4,2,6,1,3]
Output: 1

Example 2:
Input: root = [1,0,48,null,null,12,49]
Output: 1
*/


//191. SURROUNDED REGIONS                                                   {T.C = O(N*M), S.C = O(1)}
//APPROACH
//reverse solution of number of islands in which we find the group rether then we solve the boudary cases and with attach
//boundary 'O' convert to '#' and another '0' with 'X'.
class Solution {
public:
    void dfs(vector<vector<char>>& board, int i, int j, int n, int m) {
        if (i < 0 || i >= n || j < 0 || j >= m || board[i][j] != 'O') {
            return;
        }
        board[i][j] = '#';

        // Traverse all 4 directions
        dfs(board, i - 1, j, n, m);
        dfs(board, i + 1, j, n, m);
        dfs(board, i, j - 1, n, m);
        dfs(board, i, j + 1, n, m);
    }

    void solve(vector<vector<char>>& board) {
        int n = board.size();
        int m = board[0].size();

        // Handle edge case of empty board
        if (n == 0){
            return;  
        }  

        // Moving over first and last columns
        for (int i = 0; i < n; i++) {
            if (board[i][0] == 'O') {
                dfs(board, i, 0, n, m);
            }
            if (board[i][m - 1] == 'O') {
                dfs(board, i, m - 1, n, m);
            }
        }

        // Moving over first and last rows
        for (int j = 0; j < m; j++) {
            if (board[0][j] == 'O') {
                dfs(board, 0, j, n, m);
            }
            if (board[n - 1][j] == 'O') {
                dfs(board, n - 1, j, n, m);
            }
        }

        // Convert remaining 'O's to 'X' and revert '#' back to 'O'
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
                if (board[i][j] == '#') {
                    board[i][j] = 'O';
                }
            }
        }
    }
};
/*
Example 1:
Input: board = [["X","X","X","X"],
                ["X","O","O","X"],
                ["X","X","O","X"],
                ["X","O","X","X"]]
Output:     [["X","X","X","X"],
             ["X","X","X","X"],
             ["X","X","X","X"],
             ["X","O","X","X"]]
Explanation: Notice that an 'O' should not be flipped if:
- It is on the border, or
- It is adjacent to an 'O' that should not be flipped.
The bottom 'O' is on the border, so it is not flipped.
The other three 'O' form a surrounded region, so they are flipped.

Example 2:
Input: board = [["X"]]
Output: [["X"]]
*/


//192. COURSE SCHEDULE II
/*
detect the cycle in a graph by using dfs if cycle present return empty vector else true or traverse graph and not getting any cycle
then store in vector and return it
*/
class Solution {
    bool dfs(vector<vector<int>>&adj, vector<int>&ans, vector<int>&vis, int node){
        vis[node] = 1;
        for(auto it : adj[node]){
            if(vis[it] == 1){
                return true;
            }else if(vis[it] == 0 && dfs(adj, ans, vis, it)){
                return true;
            }
        }
        vis[node] = 2;               //mark as completely visited
        ans.push_back(node);
        return false;
    }
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        int n = prerequisites.size();
        //create adjacency list
        vector<vector<int>>adj(numCourses);
        for(int i = 0 ; i < n; i++){
            int u = prerequisites[i][0];
            int v = prerequisites[i][1];

            // adj[v].push_back(u);               //reverse push according to question 2nd depends on 1s to arrow should be reverse
            adj[u].push_back(v);             
        }

        vector<int>ans;
        vector<int>vis(numCourses, 0);   //initialize with 0
        for(int i = 0 ; i < numCourses ; i++){
            if(vis[i] == 0 && dfs(adj, ans, vis, i)){
                return {};                   //return empty vector for present cycle (can not be schedule)
            }
        }
        // reverse(ans.begin(), ans.end());   //ensuring that the second course (prerequisite) points to the first course in the adjacency list.
        return ans;
    }
};
/*
Example 1:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].

Example 2:
Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0.
So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3].
*/


//193. SNAKES AND LADDERS
/*
finding minimum steps for this we have to perfrom bfs 
*/
class Solution {
public:
    //bfs
    int snakesAndLadders(vector<vector<int>>& board) {
        int n = board.size();
        vector<vector<int>>vis(n, vector<int>(n, 0));
        queue<int>q;
        q.push(1);
        vis[n-1][0] = 1;            //first step is already visited (1 place)
        int move = 0;

        while(!q.empty()){
            int sz = q.size();
            for(int i = 0 ; i < sz ; i++){
                int frontNode = q.front();
                q.pop();

                if(frontNode == n*n){     //base case or breaking or stoping point
                    return move;
                }
                for(int j = 1 ; j <= 6 ; j++){
                    int nextCell = frontNode + j;
                    if(nextCell > n*n){
                        break;
                    }

                    int nextRow = n-1 - (nextCell - 1)/n;    //finding row formula
                    int nextCol = (nextCell-1) % n;          //finding col formula

                    if(nextRow % 2 == n % 2){
                        nextCol = n-nextCol-1;
                    }

                    if(!vis[nextRow][nextCol]){
                        vis[nextRow][nextCol] = 1;
                        if(board[nextRow][nextCol] != -1){
                            q.push(board[nextRow][nextCol]);
                        }else{
                            q.push(nextCell);         //board[r][c] or snack or ladder's value
                        }
                    }
                }
            }
            move++;
        }
        return -1;
    }
};
/*

Example 1:
Input: board = [[-1,-1,-1,-1,-1,-1],
                [-1,-1,-1,-1,-1,-1],
                [-1,-1,-1,-1,-1,-1],
                [-1,35,-1,-1,13,-1],
                [-1,-1,-1,-1,-1,-1],
                [-1,15,-1,-1,-1,-1]]
Output: 4
Explanation: 
In the beginning, you start at square 1 (at row 5, column 0).
You decide to move to square 2 and must take the ladder to square 15.
You then decide to move to square 17 and must take the snake to square 13.
You then decide to move to square 14 and must take the ladder to square 35.
You then decide to move to square 36, ending the game.
This is the lowest possible number of moves to reach the last square, so return 4.

Example 2:
Input: board = [[-1,-1],[-1,3]]
Output: 1
*/


//194. MIIMUM GENETIC MUTATION
/*
create a set for storing unique strings and queue for bfs if with the each iteration the final node or string same as pervious return count of steps 
else return -1 
*/
class Solution {
public:
    int minMutation(string startGene, string endGene, vector<string>& bank) {
        unordered_set<string>vis;
        queue<string>q;
        q.push(startGene);
        vis.insert(startGene);
        int count = 0;
        
        while(!q.empty()){
            int n = q.size();           //update size of queue with each iteration
            for(int i = 0 ; i < n ; i++){
                string frontNode = q.front();
                q.pop();
                if(frontNode == endGene){
                    return count;
                }
                for(auto it : "ACGT"){
                    for(int j = 0 ; j < frontNode.size() ; j++){
                       string adjNode = frontNode;
                       adjNode[j] = it;
                       //not visited and also present in the bank
                       if(!vis.count(adjNode) && find(bank.begin(), bank.end(), adjNode) != bank.end()){
                           q.push(adjNode);
                           vis.insert(adjNode);
                       } 
                    }
                }
            }
            count++;
        }
        return -1;
    }
};
/*
Example 1:
Input: startGene = "AACCGGTT", endGene = "AACCGGTA", bank = ["AACCGGTA"]
Output: 1

Example 2:
Input: startGene = "AACCGGTT", endGene = "AAACGGTA", bank = ["AACCGGTA","AACCGCTA","AAACGGTA"]
Output: 2
*/


//195. WORD LADDER
/*
similar to above
*/
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        queue<pair<string,int>> q;
        q.push({beginWord,1});
        unordered_set<string> st(wordList.begin(),wordList.end());      //insert all elements in set
        st.erase(beginWord);
        while(!q.empty()){
            pair<string,int> frontNode=q.front();
            string word=frontNode.first;
            int steps=frontNode.second;
            q.pop();

            //check for destination 
            if(word==endWord)
                return steps;
            for(int i=0;i<word.length();i++){
                char orig=word[i];
                for(char ch='a';ch<='z';ch++){
                    word[i]=ch;
                    if(st.find(word)!=st.end()){
                        q.push({word,steps+1});
                        st.erase(word);
                    }
                }
                word[i]=orig;                                    // Revert back for the next iteration call
            }
        }
        return 0;                                                //Transformation not possible
    }
};
/*
Example 1:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.

Example 2:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: 0
Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.
*/


//196. DESIGN ADD AND SEARCH WORDS DATA STRUCTURE                                       {T.C = O(N + N), S.C = O(N*M {no. of word , lenghth of word})}
/*
same approach as above but slightly differnce because in this '.' dot char may be include so that we have to handle seperatly
*/
class WordDictionary {
public:
    struct trieNode{
        bool isEndOfWord;
        trieNode* child[26];
    };

    trieNode* getNode(){
        trieNode* newNode = new trieNode();

        newNode->isEndOfWord = false;
        for(int i = 0 ; i < 26 ; i++){
            newNode->child[i] = NULL;
        }
        return newNode;
    }  

    trieNode* root;

    WordDictionary() {
        root = getNode();
    }
    
    void addWord(string word) {
        trieNode* crawler = root;
        for(int i = 0 ; i < word.length(); i++){
            int idx = word[i] - 'a';
            if(crawler->child[idx] == NULL){
                crawler->child[idx] = getNode();
            }
            crawler = crawler->child[idx];
        }
        crawler->isEndOfWord = true;
    }
    
    bool searchHelper(string &word, int index, trieNode* crawler){
        //base case
        if(index == word.length()){
            if(crawler != NULL && crawler->isEndOfWord == true){
                return true;
            }
            return false;
        }

        if(word[index] == '.'){
            for(int i = 0 ; i  < 26 ; i++){
                if(crawler->child[i] != NULL && searchHelper(word, index+1, crawler->child[i])){
                    return true;
                }
            }
            return false;
        }else{
            int idx = word[index]-'a';
            if(crawler->child[idx] == NULL){
                return false;
            }
            return searchHelper(word, index+1, crawler->child[idx]);
        }
    }
    bool search(string word) {
        trieNode* crawler = root;
        return searchHelper(word, 0, crawler);            //0 = index
    }
};
/*
Example:
Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]

Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True
*/


//197. WORD SEARCH II                                                               {T.C = O(N*M*4^L), S.C = O(N*M)}
/*
we cant use simple dfs like above it will increase the time compelxity so that's why we use trie so that we can traverse only once
*/
class Solution {
public:
    //global ans vector
    vector<string> result;

    vector<pair<int, int>> directions{{-1, 0},{1, 0},{0, 1},{0, -1}};
    struct trieNode{
        bool isEndOfWord;
        string word;
        trieNode *child[26];
    };

    trieNode* getNode(){
        trieNode* newNode = new trieNode();
        newNode->isEndOfWord = false;
        newNode->word = "";
        for(int i = 0 ; i < 26 ; i++){
            newNode->child[i] = NULL;
        }
        return newNode;
    }

    void insert(trieNode* root, string word){
        trieNode* crawler = root;                  //iterator
        for(int i = 0 ; i < word.length() ; i++){
            int idx = word[i]-'a';
            if(crawler->child[idx] == NULL){
                crawler->child[idx] = getNode();
            }
            crawler = crawler->child[idx];         //move forward
        }
        crawler->isEndOfWord = true;
        crawler->word = word;
    }

void search(vector<vector<char>>& board, trieNode* root, int i, int j){

    //base case
    if(i < 0 || i >= board.size() || j < 0 || j >= board[0].size() || board[i][j] == '$' || root->child[board[i][j] - 'a'] == NULL){
        return;
    }
    int idx = board[i][j] - 'a';
    root = root->child[idx];       //move forward

    if(root->isEndOfWord == true){
        result.push_back(root->word);
        root->isEndOfWord = false; // Marking the word as found
    }
    char temp = board[i][j];       //store current charcter
    board[i][j] = '$';             //mark visited

    //explore all 4 directions
    for(auto p : directions){         //p = pair {{-1, 0}, ...}
        int new_i = i + p.first;
        int new_j = j + p.second;
        search(board, root, new_i, new_j);
    }
    board[i][j] = temp; //reset the board cell to its original character
}

    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        int row = board.size();
        int col = board[0].size();

        //create root
        trieNode* root = getNode();

        //insert word in trie
        for(auto it : words){
            insert(root, it);
        }      

        //traverse in grid (only once)
        for(int i = 0 ; i < row ; i++){
            for(int j = 0 ; j < col ; j++){
                int idx = board[i][j] - 'a';
                if(root->child[idx] != NULL){
                    search(board, root, i, j);       //after finding word push in vector (result)
                }
            }
        }
        return result;
    }
};
/*
Example 1:
Input: board = [["o","a","a","n"],
                ["e","t","a","e"],
                ["i","h","k","r"],
                ["i","f","l","v"]], 
                words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]

Example 2:
Input: board = [["a","b"],
                ["c","d"]], 
                words = ["abcb"]
Output: []
*/


//198. COMBINATIONS                                                                {T.C = O(nCk), S.C = O(K)}
/*
store temp vector then push in ans vector after we explore each each value of k with take and non take 
*/
class Solution {
public: 
    void solve(int n, int k , vector<vector<int>>&ans, vector<int>&temp, int i){
        //base case
        if(k == 0){       //all combination taken
            ans.push_back(temp);
            return;
        }
        if(i > n){
            return;
        }
        //take and not take particular value
        temp.push_back(i);
        solve(n, k-1, ans, temp, i+1);               //take and explore rest
        temp.pop_back();
        solve(n, k, ans, temp, i+1);                 //not take and explore rest
    }
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>>ans;
        vector<int>temp;                    //store temporary ans
        solve(n, k, ans, temp, 1);          //1 = range [1, n]
        return ans;
    }
};
/*
Example 1:
Input: n = 4, k = 2
Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
Explanation: There are 4 choose 2 = 6 total combinations.
Note that combinations are unordered, i.e., [1,2] and [2,1] are considered to be the same combination.

Example 2:
Input: n = 1, k = 1
Output: [[1]]
Explanation: There is 1 choose 1 = 1 total combination.
*/


//199. PERMUTATIONS                                                                 {T.C = O(N*N!), S.C = O(N)}
/*
we requried a for loop to traverse from 0 index and used vector to keep track the number is used or not
*/
class Solution {
public:
    void solve(vector<int>&nums, vector<vector<int>>&ans, vector<int>&temp, vector<bool>&used){
        int n = nums.size();
        //base case
        if(temp.size() == n){
            ans.push_back(temp);
            return;
        }

        for(int i = 0 ; i < n ; i++){
            if(!used[i]){
                temp.push_back(nums[i]);
                used[i] = true;

                solve(nums, ans, temp, used); 

                temp.pop_back();
                used[i] = false;
            }
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>>ans;
        vector<int>temp;
        vector<bool>used(n, false);
        solve(nums, ans, temp, used);                //0 = index not neccessary because we have to start 0 always
        return ans;
    }
};
//ANOTHER APPROACH
class Solution {
private:
    void solve(vector<int> nums  , int index , vector<vector<int>> &ans){
        //base case
        if(index >= nums.size()){
            ans.push_back(nums);
            return ;
        }

        for(int i = index ; i < nums.size() ; i++){
            swap(nums[index], nums[i]);
            solve(nums , index+1 , ans);
            
            //backtracking
            swap(nums[index], nums[i]);
        }
    }
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ans;
        int index = 0;
        solve(nums , index , ans);
        return ans;
    }
};
/*
Example 1:
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

Example 2:
Input: nums = [0,1]
Output: [[0,1],[1,0]]

Example 3:
Input: nums = [1]
Output: [[1]]
*/


//200. N-QUEENS                                                                      {T.C = O(N! OR EXP), S.C = O(N^2)}
/*
first we check each col internally check row if both not attacked then move to another col and move further 
conditions must be followed :
1. every row must have 1 queen
2. every col must hanve 1 queen
3. none have been attack by any other (diagonlly attack can also happen)
*/
class Solution {
public:
    //check only 3 directions because we are filling from left to right
    /*
        ^
        \      //left-up diagonal
        <- Q     //left row or same row
        /      //left-down diagonal
        V
    */
    bool isSafe(int col , int row, vector<string>&board, int n){
        //left-up diagonal
        int x = row;
        int y = col;
        while(x >= 0 && y >= 0){
            if(board[x][y] == 'Q'){
                return false;
            }
            y--;
            x--;
        }

        //left row or same row
        x = row, y = col;               //reinitialize
        while( y >= 0){
            if(board[x][y] == 'Q'){
                return false;
            }
            y--;
        }

        //left-down diagonal
        x = row, y = col;               //reinitialize
        while(x < n && y >= 0){
            if(board[x][y] == 'Q'){
                return false;
            }
            y--;
            x++;
        }
        
        return true;
    }
    void solve(int n, vector<vector<string>>&ans, vector<string>&board, int col){
        //base case
        if(col == n){
            ans.push_back(board);
            return;
        }

        for(int row = 0 ; row < n ; row++){
            if(isSafe(col, row, board, n)){
                board[row][col] = 'Q';
                solve(n, ans, board, col+1);
                board[row][col] = '.';                 //backtrack
            }
        }
    }
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>>ans;
        vector<string>board(n, string(n, '.'));
        solve(n, ans, board, 0);                        //0 = initial column
        return ans;
    }
};
/*

Example 1:
Input: n = 4
Output: [
        [".Q..",
          "...Q",
         "Q...",
         "..Q."],

         ["..Q.",
         "Q...",
         "...Q",
         ".Q.."]
        ]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above

Example 2:
Input: n = 1
Output: [["Q"]]
*/


//201. N-QUEENS II                                                             {T.C = O(N! OR EXP), S.C = O(N^2)}
/*
same as above just right size of ans vector
*/
class Solution {
public:
    bool isSafe(int row, int col, vector<string>&board, int n){
        //left-Up diagonal attack
        int x = row, y = col;
        while(x >= 0 && y >= 0){
            if(board[x][y] == 'Q'){
                return false;
            }
            x--, y--;
        }

        //left side row or same row
        x = row, y = col;
        while(y >= 0){
            if(board[x][y] == 'Q'){
                return false;
            }
            y--;
        }

        //left-Down diagonal attack
        x = row, y = col;
        while(x < n && y >= 0){
            if(board[x][y] == 'Q'){
                return false;
            }
            x++, y--;
        }

        return true;
    }
    void solve(int n, vector<vector<string>>&ans, vector<string>&board, int col){
        //base case
        if(col == n){
            ans.push_back(board);
            return;
        }

        for(int row = 0 ; row < n ; row++){
            if(isSafe(row, col, board, n)){
                board[row][col] = 'Q';
                solve(n, ans, board, col+1);
                board[row][col] = '.';              //backtrack
            }
        }
    }
    int totalNQueens(int n) {
        vector<vector<string>>ans;
        vector<string>board(n, string(n, '.'));
        solve(n, ans, board, 0);                     //0 = initial column
        return ans.size();
    }
};
/*
Example 1:
Input: n = 4
Output: 2
Explanation: There are two distinct solutions to the 4-queens puzzle as shown.

Example 2:
Input: n = 1
Output: 1
*/


//202. GENERATE PARANTEHSIS
/*
take a temp empty string and push the "(" if opening bracket is smaller then number and push ")" 
if closing bracket is smaller then opening bracket 
*/
class Solution {
public:
    void solve(int n, vector<string>&ans, string temp, int opBr, int clBr){  //temp should be pass by value not by refence
        //base case
        if(temp.length() == n*2){                        //n*2 = pairs (opening + closing)
            ans.push_back(temp);
            return;
        }

        if(opBr < n){
            solve(n, ans, temp +"(", opBr+1, clBr);
        }
        if(clBr < opBr){
            solve(n, ans, temp + ")", opBr, clBr+1);
        }
    }
    vector<string> generateParenthesis(int n) {
        vector<string>ans;
        string temp = "";
        solve(n, ans, temp, 0, 0);            //0 = opening bracket, 0 = closing bracket
        return ans;
    }
};
/*
Example 1:
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

Example 2:
Input: n = 1
Output: ["()"]
*/


//203. CONVERT SORTED ARRAY TO BINARY SEARCH TREE                                    {T.C = O(N), S.C = O(N)}               
/*
first finds the mid element and make it root node then create a left and right subtree (vector) then make recursive call
root's left = leftsubtree , root's right = rightsubtree and finally retiurn root
*/
class Solution {
public:
    TreeNode* solve(vector<int>&nums){
        //base case
        if(nums.empty()){
            return NULL;
        }

        int mid = nums.size()/2;
        
        TreeNode* root = new TreeNode(nums[mid]);

        vector<int>leftSubTree(nums.begin(), nums.begin() + mid);
        vector<int>rightSubTree(nums.begin()+mid+1, nums.end());

        root->left = solve(leftSubTree);
        root->right = solve(rightSubTree);
        return root;

    }
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return solve(nums);
    }
};
/*
Example 1:
Input: nums = [-10,-3,0,5,9]
Output: [0,-3,9,-10,null,5]
Explanation: [0,-10,5,null,-3,null,9] is also accepted:

Example 2:
Input: nums = [1,3]
Output: [3,1]
Explanation: [1,null,3] and [3,1] are both height-balanced BSTs.
*/


//204. SORT LIST
/*
//BRUTE FORCE APPROACH                                                      {T.C = O(N*LOGN), S.C = O(N)}
first convert list to vector then sort the vector and again convert vector to list
*/
class Solution {
public:
    ListNode* ArrToLl(vector<int>&ans){
        int n = ans.size();
        if(n == 0){
            return NULL;
        }

        ListNode* head = new ListNode(ans[0]);                  //initialize head with first element of array
        ListNode* curr = head;

        for(int i = 1 ; i < n ; i++){                          //start with 2nd element (1st is head)
            curr->next = new ListNode(ans[i]);
            curr = curr->next;
        }
        return head;
    }
    void llToArr(ListNode* head, vector<int>&ans){
        ListNode* temp = head;
        while(temp != NULL){
            ans.push_back(temp->val);
            temp = temp->next;
        }
    }
    ListNode* sortList(ListNode* head) {
        vector<int>ans;
        llToArr(head, ans);
        sort(ans.begin(), ans.end());
        return ArrToLl(ans);
    }
};

//ANOTHER APPROACH (MERGE SORT)                                              {T.C = O(N*LOGN), S.C = O(1)}
/*
1. Using 2pointer / fast-slow pointer find the middle node of the list.
2. Now call mergeSort for 2 halves.
3. Merge the Sort List (divide and conqueror Approach)
*/
class Solution {
public:
    ListNode* mergeList(ListNode* l1, ListNode* l2){
        ListNode* p1 = l1; 
        ListNode* p2 = l2;
        ListNode* dummyNode = new ListNode(-1);
        ListNode* p3 = dummyNode;
        //if both list is nonempty
        while(p1 && p2){
            if(p1->val < p2->val){
                p3->next = p1;
                p1 = p1->next;
            }else{ //p1->val >= p2->val
                p3->next = p2;
                p2 = p2->next;
            }
            p3 = p3->next;                                //move p3 for both above cases
        }
        while(p1){
            p3->next = p1;
            p1 = p1->next;
            p3 = p3->next;
        }
        while(p2){
            p3->next = p2;
            p2 = p2->next;
            p3 = p3->next;
        }
        return dummyNode->next;                          //original starts from next of dummy node
    }
    ListNode* sortList(ListNode* head) {
        //base case
        if(head == NULL || head->next == NULL){
            return head;                                    
        }
        //finding mid element
        ListNode* slow = head;
        ListNode* fast = head;
        ListNode* temp = NULL;

        while(fast && fast->next){
            temp = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        temp->next = NULL;                          //end of a first left half (list divided)

        ListNode* l1 = sortList(head);              //left half
        ListNode* l2 = sortList(slow);              //right half

        return mergeList(l1, l2);
    }
};
/*
Example 1:
Input: head = [4,2,1,3]
Output: [1,2,3,4]

Example 2:
Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]
*/


// Definition for a QuadTree node.
class Node {
public:
    bool val;
    bool isLeaf;
    Node* topLeft;
    Node* topRight;
    Node* bottomLeft;
    Node* bottomRight;
    
    Node() {
        val = false;
        isLeaf = false;
        topLeft = NULL;
        topRight = NULL;
        bottomLeft = NULL;
        bottomRight = NULL;
    }
    
    Node(bool _val, bool _isLeaf) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = NULL;
        topRight = NULL;
        bottomLeft = NULL;
        bottomRight = NULL;
    }
    
    Node(bool _val, bool _isLeaf, Node* _topLeft, Node* _topRight, Node* _bottomLeft, Node* _bottomRight) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }
};
//205. CONSTRUCT QUAD TREE                                                   {T.C = O(N^2*LOGN), S.C = O(LOGN)}
/*
first check if the node is leaf node then directly return node if not then we have to make 4 recursive call for each quadrant.
*/
class Solution {
public:
    bool isAllSame(vector<vector<int>>&grid, int x, int y, int n){
        int val = grid[x][y];

        for(int i = x ; i < x+n ; i++){            //i < x + n and j < y + n is used to iterate over a specific quadrant within the larger grid.
            for(int j = y ; j < y+n ; j++){
                if(grid[i][j] != val){
                    return false;
                }
            }
        }
        return true;
    }
    Node* solve(vector<vector<int>>&grid, int x , int y, int n){
        if(isAllSame(grid, x, y, n)){
            return new Node(grid[x][y], true);                 //element, true = leafFound(stop)
        }else{
            Node* root = new Node(1, false);                   //1 = anything you want , false (not leaf)
 
            root->topLeft = solve(grid, x, y, n/2);             //n/2 we making half grid every recursive call
            root->topRight = solve(grid, x, y+n/2, n/2);
            root->bottomLeft = solve(grid, x+n/2, y, n/2);
            root->bottomRight = solve(grid, x+n/2, y+n/2, n/2);
            
            return root;
        }
    }
    Node* construct(vector<vector<int>>& grid) {
        int n = grid.size();
        return solve(grid, 0, 0, n);                 //0 = starting row, 0 = starting col
    }
};
/*
Example 1:
Input: grid = [[0,1],
               [1,0]]
Output: [[0,1],[1,0],[1,1],[1,1],[1,0]]
Explanation: The explanation of this example is shown below:
Notice that 0 represents False and 1 represents True in the photo representing the Quad-Tree.

Example 2:
Input: grid = [[1,1,1,1,0,0,0,0],
               [1,1,1,1,0,0,0,0],
               [1,1,1,1,1,1,1,1],
               [1,1,1,1,1,1,1,1],
               [1,1,1,1,0,0,0,0],
               [1,1,1,1,0,0,0,0],
               [1,1,1,1,0,0,0,0],
               [1,1,1,1,0,0,0,0]]
Output: [[0,1],[1,1],[0,1],[1,1],[1,0],null,null,null,null,[1,0],[1,0],[1,1],[1,1]]
Explanation: All values in the grid are not the same. We divide the grid into four sub-grids.
The topLeft, bottomLeft and bottomRight each has the same value.
The topRight have different values so we divide it into 4 sub-grids where each has the same value.
*/


//206. MERGE K SORTED LINKED LIST                                               {T.C = O(N*LOGN), S.C = O(1)}
/*
put first element of each list in minheap it will always pop out minimum element and we have to store this into vector or ll
*/
class compare{
public:
    bool operator()(ListNode* a, ListNode* b){
       return a->val > b->val;
    }
};
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        ListNode* dummyNode = new ListNode(NULL);
        ListNode* temp = dummyNode;

        priority_queue<ListNode* , vector<ListNode*>, compare>minHeap;

        //put first element of each lists(k elements)
        for(int i = 0 ; i < lists.size() ; i++){
            if(lists[i]){
                minHeap.push(lists[i]);
            }
        }

        while(!minHeap.empty()){
            auto topNode = minHeap.top();
            minHeap.pop();
            temp->next = topNode;              
            temp = temp->next;
            if(topNode->next){
                minHeap.push(topNode->next);
            }
        }
        return dummyNode->next;
    }
};
/*
Example 1:
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6

Example 2:
Input: lists = []
Output: []
Example 3:
*/


//207. MAXIMUM SUM CIRCULAR SUBARRAY
/*
finding 2 types of sum 
1. maxi = maximum continuous subarray sum (kadane's algo simple)
2. totalSum-mini = maximum circular subarray sum (reverse kadane's algo for mini)
max of 1 or 2
*/
class Solution {
public:
    int maxSubarraySumCircular(vector<int>& nums) {
        int n = nums.size();
        int sum1 = 0, sum2 = 0;
        int maxi = INT_MIN, mini = INT_MAX;               
        int totalSum = 0;

        for(int i = 0 ; i < n ; i++){
            totalSum += nums[i];

            sum1 += nums[i];
            maxi = max(maxi, sum1);        //straight maxsubarray sum
            if(sum1 < 0){
                sum1 = 0;
            }

            sum2 += nums[i];
            mini = min(mini, sum2);
            if(sum2 > 0){                   //reverse of above (kadane's algo)
                sum2 = 0;
            }
        }

        if(totalSum == mini){          //if all negative
            return maxi;                    //represents the maximum single element among the negative numbers
        }
        return max(maxi, totalSum-mini);    //(continuous max subarray sum , circular maxisubarray sum )
    }
};
/*
Example 1:
Input: nums = [1,-2,3,-2]
Output: 3
Explanation: Subarray [3] has maximum sum 3.

Example 2:
Input: nums = [5,-3,5]
Output: 10
Explanation: Subarray [5,5] has maximum sum 5 + 5 = 10.

Example 3:
Input: nums = [-3,-2,-3]
Output: -2
Explanation: Subarray [-2] has maximum sum -2.
*/


//208. SEARCH INSERT POSITION                                          {T.C = O(LOGN), S.C = O(1)}
/*
simple binary search if element is in vector other wise handle boundry case and after that return low or start value
because at the breaking point start will show the exact position of insert.
*/
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int n = nums.size();
        int start = 0;
        int end = n-1;
        
        if(target > nums[end]){
            return end+1;                             //last + 1 index or new element
        }
        while(start <= end){  
            int mid = (start + end)/2;
            if(nums[mid] == target){
                return mid;
            }
            else if(nums[mid] < target){
                start = mid+1;
            }
            else{ //(nums[mid] > target)
                end = mid-1;
            }
        }
        return start;                                //return low or start because after breaking loop it will show correct position to insert
    }
};
/*
Example 1:
Input: nums = [1,3,5,6], target = 5
Output: 2

Example 2:
Input: nums = [1,3,5,6], target = 2
Output: 1

Example 3:
Input: nums = [1,3,5,6], target = 7
Output: 4
*/


//209. SEARCH A 2D MATRIX                                              {T.C = O(LOG(M*N)), S.C = O(1)}
/*
finding element in matrix by converting 2D to 1D representation by row index = mid/col and col index = mid%col then apply
simple binary search.
*/
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int row = matrix.size();
        int col = matrix[0].size();

        int start = 0;
        int end = row*col-1;

        while(start <= end){
            int mid = start + (end-start)/2;               //same as (start + end)/2

            int element = matrix[mid/col][mid%col];       //mid/col = compute row index   and  [mid%col] = compute col index

            if(element == target){
                return true;
            }else if(element < target){
                start = mid + 1;
            }
            else{  //(element > target)
                end = mid -1;
            }

        }
        return false;
    }
};
/*
Example 1:
Input: matrix = [[1 ,3 ,5 ,7],
                 [10,11,16,20],
                 [23,30,34,60]], 
                 target = 3
Output: true

Example 2:
Input: matrix = [[1 ,3 ,5, 7],
                 [10,11,16,20],
                 [23,30,34,60]], 
                 target = 13
Output: false
*/


//210. FIND FIRST AND LAST POSITION OF ELEMENT IN SORTED ARRAY                       {T.C = O(LOGN), S.C = O(1)}
/*
using 2 binary searches separately and find first and last occurance index
*/
class Solution {
public:
    int findFirst(vector<int>&nums, int target){
        int n = nums.size();
        int start = 0;
        int end = n-1;
        int first = -1;
        while(start <= end){
            int mid = start + (end-start)/2;

            if(nums[mid] == target){
                first = mid;
                end = mid-1;                                //left half
            }else if(nums[mid] < target){
                start = mid+1;
            }else{
                end = mid-1;
            }
        }
        return first;
    }
    int findLast(vector<int>&nums, int target){
        int n = nums.size();
        int start = 0;
        int end = n-1;
        int last = -1;

        while(start <= end){
            int mid = start + (end-start)/2;

            if(nums[mid] == target){
                last = mid;
                start = mid+1;                 //right half
            }else if(nums[mid] < target){
                start = mid+1;
            }else{
                end = mid-1;
            }
        }
        return last;
    }
    vector<int> searchRange(vector<int>& nums, int target) {
        int first = findFirst(nums, target);
        int last = findLast(nums, target);
        return {first, last};
    }
};
/*
Example 1:
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Example 2:
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]

Example 3:
Input: nums = [], target = 0
Output: [-1,-1]
*/


//211. MEDIAN OF TWO SORTED ARRAYS                                               {T.C = O(LOGN), S.C = O(1)}
/*
divide the two arrays in 6 parts cut1, cut2, left1, left2, right1, right2 accrodingly and use binary search
*/
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int n1 = nums1.size();
        int n2 = nums2.size();
        
        if(n2 < n1){
            return findMedianSortedArrays(nums2, nums1);
        }
        
        int low = 0 , high = n1;
        
        while(low <= high){
            int cut1 = (low+high)/2;
            int cut2 = ((n1+n2+1)/2) - cut1;
            int left1 = cut1 == 0 ? INT_MIN : nums1[cut1-1];
            int left2 = cut2 == 0 ? INT_MIN : nums2[cut2-1];
            int right1 = cut1 == n1 ? INT_MAX : nums1[cut1];
            int right2 = cut2 == n2 ? INT_MAX : nums2[cut2];
            
            if(left1 <= right2 && left2 <= right1){
                if((n1+n2) % 2 == 0){            //even
                    return (max(left1, left2) + min(right1, right2))/2.0;
                }
                else{                            //odd
                    return max(left1, left2);
                }
            }
            else if(left1 > left2){
                high = cut1-1;                   //move left
            }
            else{
                low = cut1+1;                    //move right
            }
        }
        return 0.0;        
    }
};
/*
Example 1:
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.

Example 2:
Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
*/


//212. IPO                                                                  {T.C = O(N*LOGN), S.C = O(N)}
/*
store profits and capitals in pair in vector then sort this vector according to there capital then create a maxHeap
and push element in heap if the current capital is less then w (total capital) after that add current element to w 
that is final answer after pop from heap.
*/
class Solution {
public:
    static bool cmp(pair<int, int>&a, pair<int,int> &b){  //static allows you to use it without an object of the class.
        return a.second < b.second;
    }
    int findMaximizedCapital(int k, int w, vector<int>& profits, vector<int>& capital) {
        int n = profits.size();
        vector<pair<int, int>>projects;
        for(int i = 0 ; i < n ; i++){
            projects.push_back({profits[i], capital[i]});
        }
        sort(projects.begin(), projects.end(), cmp); //acending order of capital
        priority_queue<int>pq;    //maxHeap
        int i = 0;
        while(k--){
            while(i < n && projects[i].second <= w){
                pq.push(projects[i].first);
                i++;
            }
            if(!pq.empty()){            //current element add to w it is the final answer
                w += pq.top();
                pq.pop();
            }
        }
        return w;
    }
};
/*
Example 1:
Input: k = 2, w = 0, profits = [1,2,3], capital = [0,1,1]
Output: 4
Explanation: Since your initial capital is 0, you can only start the project indexed 0.
After finishing it you will obtain profit 1 and your capital becomes 1.
With capital 1, you can either start the project indexed 1 or the project indexed 2.
Since you can choose at most 2 projects, you need to finish the project indexed 2 to get the maximum capital.
Therefore, output the final maximized capital, which is 0 + 1 + 3 = 4.

Example 2:
Input: k = 3, w = 0, profits = [1,2,3], capital = [0,1,2]
Output: 6
*/


//213. FIND K PAIRS WITH SMALLEST SUMS                                                 {T.C = O(N*M*LOGK), S.C = O(N*M)}
/*
brute force approach - make maxheap of {sum, {i, j}} first put first k pairs then match the sum with top element sum and update pq accordingly
after that break(sorted so right side is always greater) pull element of maxheap and put in ans vector
*/
class Solution {
public:
    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {

        vector<vector<int>>ans;
        int n1 = nums1.size();
        int n2 = nums2.size();

        //{sum, {i, j}}
        priority_queue<pair<int, pair<int, int>>>pq;    //maxHeap
        for(int i = 0; i < n1; i++){
            for(int j = 0; j < n2; j++){
                int sum = nums1[i] + nums2[j];
                if(pq.size() < k){
                    pq.push({sum, {i, j}});
                }else if(pq.top().first > sum){           //current sum less then till that sum
                    pq.pop();
                    pq.push({sum, {i, j}});
                }else{
                    break;
                }
            }
        }

        while(!pq.empty()){            //pop element and push into ans vector
            auto temp = pq.top();
            pq.pop();

            int i = temp.second.first;
            int j = temp.second.second;

            ans.push_back({nums1[i], nums2[j]});
        }
        return ans;
    }
};
/*
Example 1:
Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
Output: [[1,2],[1,4],[1,6]]
Explanation: The first 3 pairs are returned from the sequence: [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]

Example 2:
Input: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
Output: [[1,1],[1,1]]
Explanation: The first 2 pairs are returned from the sequence: [1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]
*/


//214. ADD BINARY                                                               {T.C = O(N), S.C = O(N)}
/*
we traverse from the end of both digit by converting string first and extract carry from that and add to next element
*/
class Solution {
public:
    string addBinary(string a, string b) {
        string ans;
        int i = a.length()-1;
        int j = b.length()-1;
        int carry = 0;

        while(i >= 0 || j >= 0 || carry){
            if(i >= 0){
                carry += a[i]-'0';
                i--;
            }
            if(j >= 0){
                carry += b[j]-'0';
                j--;
            }
            ans += carry%2 + '0';                     //add carry(string) to ans
            carry /= 2;                               //move carry to next position
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }
};
/*
Example 1:
Input: a = "11", b = "1"
Output: "100"

Example 2:
Input: a = "1010", b = "1011"
Output: "10101"
*/


//215. BITWISE AND OF NUMBERS RANGE                                              {T.C = O(N), S.C = O(1)}
/*
I step:we iterate this loop till both number becomes equal;
II step:when both number become equal put c number of zeroes through left shift to find & of numbers;
*/
class Solution {
public:
    int rangeBitwiseAnd(int l, int r) {
       int c=0;
       while(l!=r)
       {
           l>>=1;
           r>>=1;
           c++;
       }
       return l<<c;
    }
};
/*
Example 1:
Input: left = 5, right = 7
Output: 4

Example 2:
Input: left = 0, right = 0
Output: 0

Example 3:
Input: left = 1, right = 2147483647
Output: 0
*/


//216. PALINDROME NUMBER                                                    {T.C = O(N), S.C = O(N)}
/*
Brute Force = convert int to string then use 2 pointers (start , end) and match starting and ending element if it is same (true) else false
*/
class Solution {
public:
    bool isPalindrome(int x) {
        string temp = to_string(x);
        int n = temp.size();
        int i = 0;
        int j = n-1;
        while(i < j){
            if(temp[i] != temp[j]){
                return false;
            }
            i++, j--;
        }
        return true;
    }
};
/*
Example 1:
Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.

Example 2:
Input: x = -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
*/


//217. PLUS ONE                                                         {T.C = O(N), S.C = O(1)}
/*
Traverse from backward if digit != 9 then simply increase the digit and return it , other wise set digit[i] == 0 and finally insert 1 
at beggining when all digit is 99..
*/
class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        int n = digits.size();
        
        //traverse from backward
        for(int i = n-1 ; i >= 0 ; i--){
            if(digits[i] != 9){
                digits[i]++;
                return digits;
            }
            digits[i] = 0;                       //if we find digit = 9 then put 0 and move forward
        }
        digits.insert(digits.begin(), 1);        //when all are 9 and need to extra place to add 1
        return digits;
    }
};
/*
Example 1:
Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
Incrementing by one gives 123 + 1 = 124.
Thus, the result should be [1,2,4].

Example 2:
Input: digits = [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.
Incrementing by one gives 4321 + 1 = 4322.
Thus, the result should be [4,3,2,2].
*/


//218. FACTORIAL TRAILING ZEROES                                               {T.C = O(LOG(5)N), S.C = O(1)}
/*
zeroes are made only with (2*5) we have to keep count of both but actually 2 count is always greater
then 5 so we have to take only min count(5) by keep dividing and adding count to ans
eg = n = 150 
ans = (150/5 == 30) + (30/5 == 6) + (6/5 == 1) ==> 37 
*/
class Solution {
public:
    int trailingZeroes(int n) {
        int ans = 0;
        while(n/5 > 0){
            ans += n/5;
            n = n/5;
        }
        return ans;
    }
};
/*
Example 1:
Input: n = 3
Output: 0
Explanation: 3! = 6, no trailing zero.

Example 2:
Input: n = 5
Output: 1
Explanation: 5! = 120, one trailing zero.
*/


//219. SQRT(X)                                                                   {T.C = O(LOGN), S.C = O(1)}
/*
using binary search sqare = mid*mid if square == n return mid else use binary search (square < n) start = mid+1 else end = mid-1
*/
class Solution {
public:
    int mySqrt(int n){
        int s = 0;
        int e = n;
        long long int ans = -1;

        while(s <= e){
            long long int mid = s + (e - s)/2;
            long long int square = mid * mid;
            if(square == n){
                return mid;
            }
            else if (square < n){
                ans = mid;
                s = mid + 1;
            }
            else{//(square > n)
                e = mid - 1;
            }
        }
        return ans;
    }
};
/*
Example 1:
Input: x = 4
Output: 2
Explanation: The square root of 4 is 2, so we return 2.

Example 2:
Input: x = 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since we round it down to the nearest integer, 2 is returned.
*/


//220. POW(X, N)                                                          {T.C = O(LOGN), S.C = O(1)}
/*
basically we have different cases to handle 1. if n == 0 return 1, 2. if n is negative then make recursive call of reciprocal and pass
negative n, 3. if n is even divide in 2 parts and pass n/2 , 4. odd then x* even condition 
*/
class Solution {
public:
    double solve(double x, long n){
        //base case
        if(n == 0){                        //2^0 = 1
            return 1;
        }
        if(n < 0){                         //2-1 = 1/2  
            return solve(1/x, -n);
        }
        if(n % 2 == 0){                   //2^8 = 2*2^4
            return solve(x*x , n/2);
        }else{
            return x*solve(x*x, (n-1)/2); //2^9 = 2*(2*2^4)
        }
    }
    double myPow(double x, int n) {
        return solve(x, (long)n);
    }
};
/*
Example 1:
Input: x = 2.00000, n = 10
Output: 1024.00000

Example 2:
Input: x = 2.10000, n = 3
Output: 9.26100

Example 3:
Input: x = 2.00000, n = -2
Output: 0.25000
Explanation: 2-2 = 1/22 = 1/4 = 0.25
*/


//221. MAX POINTS ON A LINE                             
/*
{T.C = O(N^3), S.C = O(1)}
Brute force = using simple math first find slope of first two point and then check with another point if slope is equal then increase count
and return it.
*/
class Solution {
public:
    int maxPoints(vector<vector<int>>& points) {
        int maxi = 0;
        int n = points.size();
        //base case
        if(n == 1){
            return 1;
        }

        for(int i = 0 ; i < n ; i++){           //1st point
            for(int j = i+1 ; j < n ; j++){     //2nd point
                int count = 2;
                int dx = points[j][0] - points[i][0];  //(x2-x1)
                int dy = points[j][1] - points[i][1];  //(y2-y1)
                //we can use dy/dx and compare with dy_/dx_ but (/) is costly so we use dy*dx_ == dy_*dx
                for(int k = 0 ; k < n ; k++){   //3rd point
                    if(k != i && k != j){       //3rd point should not be same to 1st or 2nd
                        int dx_ = points[k][0] - points[i][0];  //or point[j][0]   //(x3-x1) or (x3-x2)
                        int dy_ = points[k][1] - points[i][1];  //(y3-y1)
                        if(dy*dx_ == dy_*dx){
                            count++;
                        }
                    }
                }
                maxi = max(maxi, count);
            }
        }
        return maxi;
    }
};
/*
{T.C = O(N^2), S.C = O(N)}
Optimized approach = we find theta for each point and store theta, count in map and return max count
*/
class Solution {
public:
    int maxPoints(vector<vector<int>>& points) {
        int maxi = 0;
        int n = points.size();
        //base case
        if(n == 1){
            return 1;
        }

        for(int i = 0 ; i < n ; i++){           //1st point
            unordered_map<double, int>mp;       //(theta, count)
            for(int j = 0 ; j < n ; j++){      //2nd point
                if(j == i){                    //if same point then continue 
                    continue;
                }
                int dx = points[j][0] - points[i][0];  //(x2-x1)
                int dy = points[j][1] - points[i][1];  //(y2-y1)

                double theta = atan2(dy, dx);       //theta = tan-1(dy/dx)
                mp[theta]++;
            }
            for(auto it : mp){
                maxi = max(maxi, it.second+1);         //+1 for current point
            }
        }
        return maxi;
    }
};
/*
Example 1:
Input: points = [[1,1],[2,2],[3,3]]
Output: 3

Example 2:
Input: points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
Output: 4
*/


//222. COIN CHANGE II                                                      {T.C = O(N), S.C = O(N*N)}
/*
use recursion + memoization , use 2D dp changing(index, amount) 
*/
class Solution {
public:
    int solveMem(vector<int>&coins, int amount, vector<vector<int>>&dp, int index){
        //base case
        if(amount == 0){            //if amount 0 means nothing take (1 possible nothing take way)
            return 1;                 
        }
        if(amount < 0 || index >= coins.size()){
            return 0;
        }

        //step3 if ans already present return it
        if(dp[index][amount] != -1){
            return dp[index][amount];
        }

        //step2 recursive call
        int incl = solveMem(coins, amount-coins[index], dp, index);        //keeping index same , we can use the coin again.
        int excl = solveMem(coins, amount , dp, index+1);

        dp[index][amount] = incl + excl;                      //total ways (incl + excl)
        return dp[index][amount];
    }
    int change(int amount, vector<int>& coins) {
        int n = coins.size();
        //step1 create a dp vector
        vector<vector<int>>dp(n+1, vector<int>(amount+1, -1));
        return solveMem(coins, amount, dp, 0);                 //0 = initial index
    }
};
/*
Example 1:
Input: amount = 5, coins = [1,2,5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1

Example 2:
Input: amount = 3, coins = [2]
Output: 0
Explanation: the amount of 3 cannot be made up just with coins of 2.
*/


//223. TRIANGLE                                                         {T.C = O(N*M), S.C = O(N*M)}
/*
create 2D dp , intialize row , col and col = vector[row].size() in recursive call we have only 2 option (i or i+1 col) store ans in dp
with current element + min(i, i+1 col) ans.
*/
class Solution {
public:
    int solveMem(vector<vector<int>>&triangle, vector<vector<int>>&dp, int row, int col){
        int n = triangle.size();
        //base case
        if(row >= n){
            return 0;
        }
        int m = triangle[row].size();
        if(col >= m){
            return 0;
        }

        //step3 if ans already present return it
        if(dp[row][col] != -1){
            return dp[row][col];
        }

        //step2 recursive call
        int take_i0 = solveMem(triangle, dp, row+1, col);      //taking col = i
        int take_i1 = solveMem(triangle, dp, row+1, col+1);    //taking col = i+1

        dp[row][col] = triangle[row][col] + min(take_i0, take_i1);   //add current element(triangle[row][col])
        return dp[row][col];
    }
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = triangle.size();
        //step1 create a dp vector
        vector<vector<int>>dp(n+1, vector<int>(n+1, -1)); 
        return solveMem(triangle, dp, 0, 0);                     //0 = row, 0 = col
    }
};
/*
Example 1:
Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
Output: 11
Explanation: The triangle looks like:
   2
  3 4
 6 5 7
4 1 8 3
The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).

Example 2:
Input: triangle = [[-10]]
Output: -10
*/


//224. MINIMUM PATH SUM                                                        {T.C = O(N*M), S.C = O(N*M)}
/*
create 2D dp , intialize row , col and col = vector[0].size() in recursive call we have only 2 option (down(row+1) or right(col+1)) store ans in dp
with current element + min(down, right ) ans.
handle extra base case if reach to bottom right return grid[row][col]
*/
class Solution {
public:
    int solveMem(vector<vector<int>>&grid, vector<vector<int>>&dp, int row, int col){
        int n = grid.size();
        int m = grid[0].size();
        //base case
        if(row >= n || col >= m){
            return 1e9;
        }
        if(row == n-1 && col == m-1){              //reach to bottom right
            return grid[row][col];
        }

        //step3 if ans already present return it
        if(dp[row][col] != -1){
            return dp[row][col];
        }

        //step2 recursive call
        int down = solveMem(grid, dp, row+1, col);
        int right = solveMem(grid, dp, row, col+1);

        dp[row][col] = grid[row][col] + min(down, right);                  //add current element (grid[row][col])
        return dp[row][col];
    }
    int minPathSum(vector<vector<int>>& grid) {
        int n = grid.size();
        int m = grid[0].size();
        //step1 create dp vector
        vector<vector<int>>dp(n+1, vector<int>(m+1, -1));
        return solveMem(grid, dp, 0, 0);                   //0 = row, 0 = col
    }
};
/*
Example 1:
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.

Example 2:
Input: grid = [[1,2,3],[4,5,6]]
Output: 12
*/


//225. UNIQUE PATHS II                                                     {T.C = O(N*M), S.C = O(N*M)}
/*
create 2D dp , intialize row , col and col = vector[0].size() in recursive call we have only 2 option (down(row+1) or right(col+1)) store ans and return
it (shows total ways)
handle extra base case if reach to bottom right return grid[row][col] && vector[row][col] == 1(obstacle) return 0
*/
class Solution {
public:
    int solveMem(vector<vector<int>>&obstacleGrid, vector<vector<int>>&dp, int row, int col){
        int n = obstacleGrid.size();
        int m = obstacleGrid[0].size();
        //base case
        if(row >= n || row < 0 || col >= m || col < 0 || obstacleGrid[row][col] == 1){  //boundary cases
            return 0;
        }
        if(row == n-1 && col == m-1){              //reached required position
            return 1;
        }

        //step3 if ans already present return it
        if(dp[row][col] != -1){
            return dp[row][col];
        }

        //step2 recursive call
        int down = solveMem(obstacleGrid, dp, row+1, col);
        int right = solveMem(obstacleGrid, dp, row, col+1);

        dp[row][col] = down + right;                                         //total ways
        return dp[row][col];
    }
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int n = obstacleGrid.size();
        int m = obstacleGrid[0].size();
        //step1 create a dp vector
        vector<vector<int>>dp(n+1, vector<int>(m+1, -1));
        return solveMem(obstacleGrid, dp, 0, 0);                //0 = row, 0 = col, 0 = count
    }
};
/*
Example 1:
Input: obstacleGrid = [[0,0,0],
                       [0,1,0],
                       [0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right

Example 2:
Input: obstacleGrid = [[0,1],
                       [0,0]]
Output: 1
*/


//226. INTERLEAVING STRING                                                {T.C = O(m*n*N), S.C = O(m*n*N)}
/*
Using 3D dp match each char of s1 and s2 with s3 , if char match move the particular index of matching string and s3 string , base case
if all index exhaust simultaneously then true, m+n != N gives false.
*/
class Solution {
public:
    bool solveMem(string &s1, string&s2, string&s3, vector<vector<vector<int>>>&dp, int i, int j, int k){
        int m = s1.length();
        int n = s2.length();
        int N = s3.length();
        //base case
        if(m+n != N){                      //s3 = s1 + s2
            return false;
        }
        if(i == m && j == n && k == N){     //all strings exhaust simultaneously
            return true;
        }
        if(k >= N){                          // If 's3' is fully processed but 's1' and 's2' are not fully interleaved
            return false;
        }

        //step3 if ans already present return it
        if(dp[i][j][k] != -1){
            return dp[i][j][k];
        }

        //step2 recursive call
        bool ans1 = false; 
        bool ans2 = false;
        if(s1[i] == s3[k]){                   //if s1 char match with s3 char, move i and k both
            ans1 = solveMem(s1, s2, s3, dp, i+1, j, k+1);
        }
        if(s2[j] == s3[k]){                   //if s2 char match with s3 char, move j and k both
            ans2 = solveMem(s1, s2, s3, dp, i, j+1, k+1);
        }

        return dp[i][j][k] = ans1 || ans2;
    }
    bool isInterleave(string s1, string s2, string s3) {
        int m = s1.length();
        int n = s2.length();
        int N = s3.length();
        vector<vector<vector<int>>>dp(m+1, vector<vector<int>>(n+1,vector<int>(N+1, -1) ));
        return solveMem(s1, s2, s3, dp, 0, 0, 0);          //0 = initial index of each string
    }
};
/*
Slightly Optimized Code                                                {T.C = O(M*N), S.C = O(M*N)}
we dont need to keep track of k we can use k => i+j all other is same and use 2D dp instead of 3D dp
*/
class Solution {
public:
    bool solveMem(string &s1, string&s2, string&s3, vector<vector<int>>&dp, int i, int j){
        int m = s1.length();
        int n = s2.length();
        int N = s3.length();
        //base case
        if(m+n != N){                      //s3 = s1 + s2
            return false;
        }
        if(i == m && j == n && i+j == N){     //all strings exhaust simultaneously
            return true;
        }
        if(i+j >= N){                          // If 's3' is fully processed but 's1' and 's2' are not fully interleaved
            return false;
        }

        //step3 if ans already present return it
        if(dp[i][j] != -1){
            return dp[i][j];
        }

        //step2 recursive call
        bool ans1 = false; 
        bool ans2 = false;
        if(s1[i] == s3[i+j]){                   //if s1 char match with s3 char, move i 
            ans1 = solveMem(s1, s2, s3, dp, i+1, j);
        }
        if(s2[j] == s3[i+j]){                   //if s2 char match with s3 char, move j 
            ans2 = solveMem(s1, s2, s3, dp, i, j+1);
        }

        return dp[i][j] = ans1 || ans2;
    }
    bool isInterleave(string s1, string s2, string s3) {
        int m = s1.length();
        int n = s2.length();
        vector<vector<int>>dp(m+1, vector<int>(n+1, -1));
        return solveMem(s1, s2, s3, dp, 0, 0);          //0 = initial index of string s1 and s2
    }
};
/*
Example 1:
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true
Explanation: One way to obtain s3 is:
Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
Since s3 can be obtained by interleaving s1 and s2, we return true.

Example 2:
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
Output: false
Explanation: Notice how it is impossible to interleave s2 with any other string to obtain s3.
*/


//227. MAXIMAL SQUARE                                                {T.C = O(N*M), S.C = O(N*M)}
/*
traverse the matrix and move right, down, diag find min ways for finding square then choose square which include max 1's
*/
class Solution {
public:
    int solveMem(vector<vector<char>>&matrix, vector<vector<int>>&dp,int &maxi, int row, int col){
        int n = matrix.size();
        int m = matrix[0].size();
        //base case
        if(row >= n || col >= m){
            return 0;
        }

        //step3 if ans already present return it
        if(dp[row][col] != -1){
            return dp[row][col];
        }

        //step2 recursive call
        int right = solveMem(matrix, dp, maxi, row, col+1);
        int down = solveMem(matrix, dp, maxi, row+1, col);
        int diag = solveMem(matrix, dp, maxi, row+1, col+1);

        if(matrix[row][col] == '1'){
            dp[row][col] = 1 + min({right, down, diag});    //These positions represent the three directions that can extend the square containing '1's. Adding 1 represents the current cell itself, making it a square.
            maxi = max(maxi, dp[row][col]);
            return dp[row][col];
        }

        return 0;
    }
    int maximalSquare(vector<vector<char>>& matrix) {
        int n = matrix.size();
        int m = matrix[0].size();
        int maxi = 0;
        //step1 create a dp vector
        vector<vector<int>>dp(n+1, vector<int>(m+1, -1));
        solveMem(matrix, dp, maxi, 0, 0);                    //0 = row, 0 = col

        return maxi*maxi;                                    //area = maxi * maxi

    }
};
/*
Example 1:
Input: matrix = [["1","0","1","0","0"],
                 ["1","0","1","1","1"],
                 ["1","1","1","1","1"],
                 ["1","0","0","1","0"]]
Output: 4

Example 2:
Input: matrix = [["0","1"],
                 ["1","0"]]
Output: 1
*/

/*----------------------------------------------------------  THE END  ---------------------------------------------------------------------*/


//228. PRIME NUMBER OF SET BITS IN A BINARY REPRESENTATION                   {T.C = O(N^2), S.C = O(1)}
class Solution {
public:
    bool isPrime(int n){
        if(n <= 1) return false;
        if(n <= 3) return true;          //2,3 prime
        if(n % 2 == 0 || n % 3 == 0) return false;  //above line handled
        for(int i = 5; i * i <= n ; i += 6){  //check from 5 to root(n) (all forms are of 6k ± 1)
            if(n % i == 0 || n % (i + 2) == 0) return false;
        }
        return true;
    }
    int countPrimeSetBits(int left, int right) {
        int count = 0;
        for(int i = left ; i <= right; i++){
            int bitCounts = __builtin_popcount(i);
            if(isPrime(bitCounts)) count++;
        }
        return count;
    }
};
/*
Example 1:
Input: left = 6, right = 10
Output: 4
Explanation:
6  -> 110 (2 set bits, 2 is prime)
7  -> 111 (3 set bits, 3 is prime)
8  -> 1000 (1 set bit, 1 is not prime)
9  -> 1001 (2 set bits, 2 is prime)
10 -> 1010 (2 set bits, 2 is prime)
4 numbers have a prime number of set bits.

Example 2:
Input: left = 10, right = 15
Output: 5
Explanation:
10 -> 1010 (2 set bits, 2 is prime)
11 -> 1011 (3 set bits, 3 is prime)
12 -> 1100 (2 set bits, 2 is prime)
13 -> 1101 (3 set bits, 3 is prime)
14 -> 1110 (3 set bits, 3 is prime)
15 -> 1111 (4 set bits, 4 is not prime)
5 numbers have a prime number of set bits.
*/


//229. DEGREE OF AN ARRAY                                              {T.C = O(N), S.C = O(N)}
class Solution {
public:
    int findShortestSubArray(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> freq, mp;              //mp {ele, idx}
        int minLen = 0, maxFreq = 0;

        for (int i = 0; i < n; i++) {
            freq[nums[i]]++;                           //fill freq
            if (!mp.count(nums[i])) mp[nums[i]] = i;   //fill mp

            if (freq[nums[i]] > maxFreq) {
                maxFreq = freq[nums[i]];
                minLen = i - mp[nums[i]] + 1;
            } else if (freq[nums[i]] == maxFreq) {
                minLen = min(minLen, i - mp[nums[i]] + 1);
            }
        }

        return minLen;
    }
};
/*
Example 1:
Input: nums = [1,2,2,3,1]
Output: 2
Explanation: 
The input array has a degree of 2 because both elements 1 and 2 appear twice.
Of the subarrays that have the same degree:
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
The shortest length is 2. So return 2.

Example 2:
Input: nums = [1,2,2,3,1,4,2]
Output: 6
Explanation: 
The degree is 3 because the element 2 is repeated 3 times.
So [2,2,3,1,4,2] is the shortest subarray, therefore returning 6.
*/


//230. DESIGN HASH SET                                     {T.C = O(1), S.C = O(1)}
class MyHashSet {
public:
    vector<bool>vec;
    MyHashSet() {
        vec.resize(1e6+1, false);
    }
    
    void add(int key) {
        vec[key] = true;
    }
    
    void remove(int key) {
        vec[key] = false;
    }
    
    bool contains(int key) {
        return vec[key];
    }
};
/**
 * Your MyHashSet object will be instantiated and called as such:
 * MyHashSet* obj = new MyHashSet();
 * obj->add(key);
 * obj->remove(key);
 * bool param_3 = obj->contains(key);
 */


//231. DESIGN HASH MAP                                     {T.C = O(1), S.C = O(1)}
class MyHashMap {
public:
    vector<int>vec;
    MyHashMap() {
        vec.resize(1e6+1, -1);
    }
    
    void put(int key, int value) {
        vec[key] = value;
    }
    
    int get(int key) {
        return vec[key];
    }
    
    void remove(int key) {
        vec[key] = -1;
    }
};

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap* obj = new MyHashMap();
 * obj->put(key,value);
 * int param_2 = obj->get(key);
 * obj->remove(key);
 */


//232. VALID PALINDROME                                       {T.C = O(N), S.C = O(N)}
class Solution {
public:
    bool isValidPalindrome(string &s){
        int n = s.length();
        int left = 0, right = n-1;
        while(left < right){
            if(s[left] != s[right]) return false;
            left++, right--;
        }
        return true;
    }
    bool isPalindrome(string s) {
        string temp = "";
        for(auto it : s){
            if(isalnum(it)){
                temp.push_back(tolower(it));
            }
        }
        return isValidPalindrome(temp);
    }
};
/*
Example 1:
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Example 2:
Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Example 3:
Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.
Since an empty string reads the same forward and backward, it is a palindrome.
*/


//233. VALID PALINDROME II                                       {T.C = O(N), S.C = O(N)}
class Solution {
public:
    bool isPalindrome(string &s, int i, int j){
        while(i < j){
            if(s[i] != s[j]) return false;
            i++, j--;
        }
        return true;
    }
    bool validPalindrome(string s) {
        int n = s.length();
        int left = 0, right = n-1;
        while(left < right){
            if(s[left] == s[right]){
                left++, right--;
            }else{
                return isPalindrome(s, left+1, right) || isPalindrome(s, left, right-1);   //skip 1 place
            }
        } 
        return true;
    }
};
/*
Example 1:
Input: s = "aba"
Output: true

Example 2:
Input: s = "abca"
Output: true
Explanation: You could delete the character 'c'.

Example 3:
Input: s = "abc"
Output: false
*/