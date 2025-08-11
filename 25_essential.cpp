//01. 3 SUM                                                         {T.C = O(N^2), S.C = O(N)}
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


//02. CONTAINER WITH MOST WATER                                               {T.C = O(N), S.C = O(1)}
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


//03. LONGEST CONSECUTIVE SEQUENCE                               
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


//STRINGS********************************************************
//04. LONGEST SUBSTRING WITHOUT REPEATING CHARACTERS             {T.C = O(N), S.C = O(N)}
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


//05. MINIMUM WINDOW SUBSTRING                                   {T.C = O(N), S.C = O(N)}
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


/*DYNAMIC PROGRAMMING********************************************************************/
//06. CLIMBING STAIRS                                              {T.C = O(N), S.C = O(N)}
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



//07. HOUSE ROBBER I                                               {T.C = O(N), S.C = O(N)}
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


//08. LONGEST PALINDROMIC SUBSTRING
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


//09. COIN CHANGE                                                      {T.C = O(N*TARGET), S.C = O(N*TARGET)}            
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
    

//10. PARTITION EQUAL SUBSET SUM                                          {T.C = O(N*SUM), S.C = O(N*SUM)}
class Solution {
public:
    int dp[201][20001];                         //sum (max limit = 200x100 = 20000)
    bool solveMem(vector<int>&nums, int sum , int i){
        int n = nums.size();
        //base case
        if(sum == 0) return true;
        if(i >= n || sum < 0) return false;

        if(dp[i][sum] != -1) return dp[i][sum];

        int incl = solveMem(nums, sum-nums[i], i+1);
        int excl = solveMem(nums, sum, i+1);

        return dp[i][sum] = incl || excl;
    }
    bool canPartition(vector<int>& nums) {
        memset(dp, -1, sizeof(dp));
        int sum = 0;
        for(auto it : nums) sum += it;

        if(sum % 2 != 0) return false;                   //for odd, partition not possible
        int target = sum/2;                              //2 partition 

        return solveMem(nums, target, 0);             //0 = initial index
    }
};
/*
Example 1:
Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].

Example 2:
Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.
*/


/*TREES**************************************************************************** */
//11. BINARY TREE LEVEL ORDER TRAVERSAL                         {T.C = O(N), S.C = O(N)} 
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


//12. DIAMETER OF BINARY TREE                                            {T.C = O(N), S.C = O(H)}
class Solution {
public:
    int findDiameter(TreeNode* root, int &maxi){
        //base case
        if(!root) return 0;

        int left = findDiameter(root->left, maxi);
        int right = findDiameter(root->right, maxi);

        maxi = max(maxi, left+right);

        return max(left, right) + 1;                     //include curr node

    }
    int diameterOfBinaryTree(TreeNode* root) {
        int maxi = INT_MIN;
        int ans = findDiameter(root, maxi);            //we have to pass by refernce(not call in main func)

        return  maxi;
    }
};
/*
Example 1:
Input: root = [1,2,3,4,5]
Output: 3
Explanation: 3 is the length of the path [4,2,1,3] or [5,2,1,3].

Example 2:
Input: root = [1,2]
Output: 1
*/

//13. LOWEST COMMON ANCESTOR OF BST                              
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


/*BACKTRACKING******************************************************************** */
//14. SUBSETS                                                     {T.C = O(2^N), S.C = O(N)}
class Solution {
public:
    void solve(vector<int>&nums, vector<int>&temp, vector<vector<int>>&ans, int i){
        int n = nums.size();
        //base case
        if(i >= n){
            ans.push_back(temp);
            return;
        }

        temp.push_back(nums[i]);                  //incl
        solve(nums, temp, ans, i+1);

        temp.pop_back();                          //excl
        solve(nums, temp, ans, i+1);
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>>ans;
        vector<int>temp;
        solve(nums, temp, ans, 0);                  //0 = initial index
        return ans;
    }
};
/*
Example 1:
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

Example 2:
Input: nums = [0]
Output: [[],[0]]
*/


//15. COMBINATION SUM                                             {T.C = O(EXP / N^TARGET), S.C = O(N)}
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


//16. WORD SEARCH I                                                    {T.C = O(4^N*N*M), S.C = O(N*M)}
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


/*LINKED LIST***********************************************************************/
//17. REVERSE LINKED LIST                                        {T.C = O(N), S.C = O(1)}
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


//18. LINKED LIST CYCLE                                           {T.C = O(N), S.C = O(1)}
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


//19. MERGE 2 SORTED LIST                                         {T.C = O(N), S.C = O(1)}
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


/*STACKS******************************************************************************* */
//20. VALID PARENTHESES                                           {T.C = O(N), S.C = O(N)}
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


//21. DAILY TEMPRATURES                                          {T.C = O(N*LOGN), S.C = O(N)}
/*
Take an stack, traverse the vector(tempratures) , while(!st.empty() && element > element[stk.top{index}]) , prevIdx = st.top()
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


/*GRAPHS****************************************************************************/
//22. NUMBER OF ISLANDS (IN A MATRIX)                             {T.C = O(N*M), S.C = O(N*M)}
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


//23. CLONE A GRAPH                                               {T.C = O(N), S.C = O(N)}
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

//24. COURSE SCHEDULE                                             {T.C = O(V+E), S.C = O(V+E)}
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


//25. REDUNDANT CONNECTIONS                                             {T.C = O(N), S.C = O(N)}
class Solution {
public:
    //UNION AND FIND                              {T.C = O(N), S.C = O(N)}
    vector<int>parent, rank;
    int find(int i){
        if(parent[i] == i) return i;
        return parent[i] = find(parent[i]);
    }
    void Union(int x, int y){
        int xParent = find(x);
        int yParent = find(y);

        if(xParent == yParent) return;

        if(rank[xParent] > rank[yParent]) parent[yParent] = xParent;
        else if(rank[xParent] < rank[yParent]) parent[xParent] = yParent;
        else{
            parent[xParent] = yParent;
            rank[yParent]++;
        }
    }
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        rank.resize(n+1, 0);                  //1 based indexing
        parent.resize(n+1); 
        for(int i = 0 ; i < n ; i++) parent[i] = i;

        for(auto &it : edges){
            int uParent = find(it[0]);
            int vParent = find(it[1]);
            
            if(uParent == vParent) return {it[0], it[1]};
            else Union(uParent, vParent);
        }
        return {};
    }
};
/*
Example 1:
Input: edges = [[1,2],[1,3],[2,3]]
Output: [2,3]

Example 2:
Input: edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]
Output: [1,4]
*/