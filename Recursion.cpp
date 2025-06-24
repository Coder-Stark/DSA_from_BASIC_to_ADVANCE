// //RECURSION


#include<bits/stdc++.h>
using namespace std;
/*
1. RECURSION = MAKE INPUT SMALLER  (DECISION -> AUTOMATIC SMALLER INPUT)
2. RECURSION => CHOICES + DECISION
3. MAKE RECURSION TREE = INPUT - OUTPUT METHOD
                        OP, IP                                            " ", "ab"
            not take                take                       not take                take
        OP1, IP(SMALLER)         OP2, IP(SMALLER)             " ", "b"                "a", "b"

    FINAL ANSWER WHEN IP(SMALLER )=> " " (EMPTY)

4. CODE FLOW - 

***APPROACH RECURSION PROBLEM
1. IBH  (BASE CASE -> HYPOTHESIS -> INDUCTION)
2. RECURSION TREE
3. CHOICE DIAGRAM
*/



//*************************IBH PATTERN (EASY PROBLEMS)

//01. PRINT 1 TO N WITHOUT USING LOOP (USING RECURSION)                               {T.C = O(N), S.C = O(1)}
class Solution {
    public:
    void printTillN(int n) {
        //base case
        if(n == 0) return;                         //B
        printTillN(n-1);                           //H
        cout<<n<<" ";                              //I
    }
};
/*
Input: n = 5
Output: 1 2 3 4 5
Explanation: We have to print numbers from 1 to 5.

Input: n = 10
Output: 1 2 3 4 5 6 7 8 9 10
Explanation: We have to print numbers from 1 to 10.
*/


//02. PRINT N TO 1 WITHOUT USING LOOP (USING RECURSION)                               {T.C = O(N), S.C = O(1)}
class Solution {
  public:
    void printNos(int n) {
        //base case
        if(n == 0) return;
        cout<<n<<" ";
        printNos(n-1);
    }
};
/*
Input:
N = 10
Output: 10 9 8 7 6 5 4 3 2 1
*/


//03. FACTORIAL                                                                      {T.C = O(N), S.C = O(1)}
class Solution {
  public:
    int factorial(int n) {
        //base case
        if(n == 0 || n == 1) return 1;
        
        return n * factorial(n-1);
    }
};
/*
Input: n = 5
Output: 120
Explanation: 1 x 2 x 3 x 4 x 5 = 120

Input: n = 4
Output: 24
Explanation: 1 x 2 x 3 x 4 = 24
*/


//04. MAXIMUM DEPHT OF BINARY TREE OR HEIGHT OF BINARY TREE                         {T.C = O(N), S.C= O(H)}
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
Input: root = [3,9,20,null,null,15,7]
Output: 3

Example 2:
Input: root = [1,null,2]
Output: 2
*/


//05. SORT AN ARRAY                                                                {T.C = O(N^2), S.C = O(N)}
class Solution {
public:
    void insert(vector<int>&nums, int temp){
        int n = nums.size();
        //base case
        if(n == 0 || nums[n-1] <= temp){          //temp > last ele (simple push)
            nums.push_back(temp);      
            return;
        }
        int temp2 = nums[n-1];
        nums.pop_back();

        insert(nums, temp);             //not temp2

        nums.push_back(temp2);         //put saved element back
    }
    void sortFunc(vector<int>&nums){
        int n = nums.size();
        //base case
        if(n == 1) return;                    //1 element already sorted
        int temp = nums[n-1];                 //extract last ele
        nums.pop_back();

        sortFunc(nums);
        insert(nums, temp);                   //insert in correct position
    }
    vector<int> sortArray(vector<int>& nums) {
        sortFunc(nums);
        return nums;
    }
};
/*
Example 1:
Input: nums = [5,2,3,1]
Output: [1,2,3,5]
Explanation: After sorting the array, the positions of some numbers are not changed (for example, 2 and 3), while the positions of other numbers are changed (for example, 1 and 5).

Example 2:
Input: nums = [5,1,1,2,0,0]
Output: [0,0,1,1,2,5]
Explanation: Note that the values of nums are not necessairly unique.
*/


//06. SORT A STACK                                                {T.C = O(N^2), S.C = O(N)}
void insert(stack<int>&stk, int temp){
    int n = stk.size();
    //base case
    if(n == 0 || stk.top() <= temp){                //by default gives reverse sort (for incr sort >=)
        stk.push(temp);
        return;
    }
    
    int temp2 = stk.top();
    stk.pop();
    
    insert(stk, temp);
    stk.push(temp2);                  //put saved element back
}
void sortFunc(stack<int>&stk){
    int n = stk.size();           //s = stack
    //base case
    if(n == 1) return;
    
    int temp = stk.top();
    stk.pop();
    
    sortFunc(stk);
    insert(stk, temp);
}
void SortedStack ::sort() {
    sortFunc(s);
}
/*
Example 1:
Input:
Stack: 3 2 1
Output: 3 2 1

Example 2:
Input:
Stack: 11 2 32 3 41
Output: 41 32 11 3 2
*/


//07. DELETE MID OF A STACK                                        {T.C = O(N), S.C= O(N)}
class Solution {
  public:
    // Function to delete middle element of a stack.
    void solve(stack<int>&stk, int k){
        //base case
        if(k == 1){              //K = 1 shows find midIdx (which we have to remove)
            stk.pop();           //mid ele
            return;
        }
        int temp = stk.top();
        stk.pop();
        
        solve(stk, k-1);
        stk.push(temp);           //put saved element back
    }
    void deleteMid(stack<int>& s) {
        int n = s.size();
        int mid = (n/2)+1;
        solve(s, mid);
    }
};
/*
Input: s = [10, 20, 30, 40, 50]
Output: [50, 40, 20, 10]
Explanation: The bottom-most element will be 10 and the top-most element will be 50. Middle element will be element at index 3 from bottom, which is 30. Deleting 30, stack will look like {10 20 40 50}.

Input: s = [10, 20, 30, 40]
Output: [40, 30, 10]
Explanation: The bottom-most element will be 10 and the top-most element will be 40. Middle element will be element at index 2 from bottom, which is 20. Deleting 20, stack will look like {10 30 40}.

Input: s = [5, 8, 6, 7, 6, 6, 5, 10, 12, 9]
Output: [9, 12, 10, 5, 6, 7, 6, 8, 5]
*/


//08. REVERSE A STACK                                               {T.C = O(N^2), S.C = O(N)}
class Solution {
  public:
    void insertBottom(stack<int>&stk, int temp){
        //base case
        if(stk.empty()){
            stk.push(temp);
            return;
        }
        
        int temp2 = stk.top();
        stk.pop();
        
        insertBottom(stk, temp);
        stk.push(temp2);                       //put saved ele back
    }
    void Reverse(stack<int> &St) {
        int n = St.size();
        //base case
        if(St.empty()) return;
        int temp = St.top();
        St.pop();
        
        Reverse(St);
        insertBottom(St, temp);
    }
};
/*
Example 1:
Input:
St = {3,2,1,7,6}
Output:
{6,7,1,2,3}
Explanation:
Input stack after reversing will look like the stack in the output.

Example 2:
Input:
St = {4,3,9,6}
Output:
{6,9,3,4}
Explanation:
Input stack after reversing will look like the stack in the output.
*/


//09. KTH SYMBOL IN GRAMMER                                             {T.C = O(N), S.C = O(1)}
class Solution {
public:
    int kthGrammar(int n, int k) {
        //base case
        if(n == 1 && k == 1) return 0;

        int len = pow(2, n-1);
        int mid = len/2;

        if(k <= mid) return kthGrammar(n-1, k);
        else return !kthGrammar(n-1, k-mid);
    }
};
/*
Example 1:
Input: n = 1, k = 1
Output: 0
Explanation: row 1: 0

Example 2:
Input: n = 2, k = 1
Output: 0
Explanation: 
row 1: 0
row 2: 01

Example 3:
Input: n = 2, k = 2
Output: 1
Explanation: 
row 1: 0
row 2: 01
*/


//10. TOWER OF HANOI                                           {T.C = O(2^N), S.C = O(N)}
class Solution {
  public:
    int towerOfHanoi(int n, int from, int to, int aux) {
        //base case
        if(n == 0) return 0;
        int way1 = towerOfHanoi(n-1, from, aux, to);
        int way2 = towerOfHanoi(n-1, aux, to, from);
        
        return way1 + way2 + 1;         //1 for curr step
    }
};
/*
Input: n = 2
Output: 3
Explanation: For n =2 , steps will be as follows in the example and total 3 steps will be taken.
move disk 1 from rod 1 to rod 2
move disk 2 from rod 1 to rod 3
move disk 1 from rod 2 to rod 3
*/


//11. JOSEPHUS PROBLEM                                         {T.C = O(N^2), S.C = O(N)}
class Solution {
  public:
    void solve(vector<int>person , int k , int index , int &ans){ 
        if(person.size() == 1) { 
            ans = person[0]; 
            return; 
        } 
        index = (index + k)%person.size(); 
        person.erase(person.begin() + index); 
        solve(person , k , index , ans); 
    }
        
    int josephus(int n, int k){ 
        k = k-1; 
        vector<int>person; 
        for(int i = 1 ; i <= n ; i++) { 
            person.push_back(i); 
        } 
        int index = 0 , ans = -1; 
        solve(person , k , index , ans); 
        return ans;
    }
};
/*
Input: n = 3, k = 2
Output: 3
Explanation: There are 3 persons so skipping 1 person i.e 1st person 2nd person will be killed. Thus the safe position is 3.

Input: n = 5, k = 3
Output: 4
Explanation: There are 5 persons so skipping 2 person i.e 3rd person will be killed. Thus the safe position is 4.
*/


// //************************IP-OP PATTERN
/*
SUBSTRING = CONTINOUS
SUBSEQUENCE = NON-CONTINOUS WITH ORDER
SUBSET = NON-CONTINOUS WITHOUT ORDER

POWERSET == SUBSET == SUBSEQUENCE(SOME CASE)
*/
//11. SUBSETS                                        {T.C = O(2^N * N [for storing op1, op2]), S.C = O(N)}
//ON STRING
void solve(string ip, string op){
    int n = ip.size(), m = op.size();
    if(n == 0){
        cout<<op<<endl;                                 //its not storing directly printing so SC = o(n)[stack space]
        return;
    }

    string op1 = op;                   //not take case
    string op2 = op;                   //take case

    op2.push_back(ip[0]);
    ip.erase(ip.begin() + 0);

    solve(ip, op1);
    solve(ip, op2);

    return;
}
int main(){
    string ip;
    cin>>ip;
    string op = "";
    solve(ip, op);
    cout<<op<<endl;
}
/*
input - abc
output -   c b bc a ac ab abc
*/


//ON VECTOR
class Solution {
  public:
    void solve(vector<int>ip, vector<int>op, vector<vector<int>>&ans){
        //base case
        if(ip.size() == 0){
            ans.push_back(op);
            return;
        }
        
        vector<int>op1 = op;
        vector<int>op2 = op;
        
        op2.push_back(ip[0]);
        ip.erase(ip.begin() + 0);              //new input
        
        solve(ip, op1, ans);
        solve(ip, op2, ans);
    }
    vector<vector<int>> subsets(vector<int>& arr) {
        vector<int>op;
        vector<vector<int>>ans;
        solve(arr, op, ans);
        sort(ans.begin(), ans.end());
        return ans;
    }
};
/*
Input: arr = [1, 2, 3]
Output: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
Explanation: 
The subsets of [1, 2, 3] in lexicographical order are:
[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]

Input: arr = [1, 2]
Output: [[], [1], [1, 2], [2]]
Explanation:
The subsets of [1, 2] in lexicographical order are:
[], [1], [1, 2], [2]

Input: arr = [10]
Output: [[], [10]]
Explanation: For the array with a single element [10], the subsets are [ ] and [10].
*/


//12. UNIQUE SUBSETS                                               {T.C = O(2^N * N), S.C = O(2^N * N)}
class Solution {
  public:
    // Function to find all possible unique subsets.
    void solve(vector<int>ip, vector<int>op, set<vector<int>>&ans){
        //base case
        if(ip.size() == 0){
            ans.insert(op);
            // ans.push_back(op);
            return;
        }
        
        vector<int>op1 = op;
        vector<int>op2 = op;
        
        op2.push_back(ip[0]);
        ip.erase(ip.begin() + 0);          //new input
        
        solve(ip, op1, ans);
        solve(ip, op2, ans);
    }
    vector<vector<int>> AllSubsets(vector<int> arr, int n) {
        sort(arr.begin(), arr.end());
        vector<int>op;
        set<vector<int>>ans;
        solve(arr, op, ans);

        vector<vector<int>>res(ans.begin(), ans.end());
        return res;
    }
};
/*
Example 1:
Input: N = 3, arr[] = {2,1,2}
Output:(),(1),(1 2),(1 2 2),(2),(2 2)
Explanation: 
All possible subsets = (),(2),(1),(1,2),(2),(2,2),(2,1),(2,1,2)
After Sorting each subset = (),(2),(1),(1,2),(2),(2,2),(1,2),(1,2,2) 
Unique Susbsets in Lexicographical order = (),(1),(1,2),(1,2,2),(2),(2,2)

Example 2:
Input: N = 4, arr[] = {1,2,3,3}
Output: (),(1),(1 2),(1 2 3)
(1 2 3 3),(1 3),(1 3 3),(2),(2 3)
(2 3 3),(3),(3 3)
*/


//13. PERMUTATIONS WITH SPACES                                           {T.C = O(2^N * N), S.C = O(2^N * N)}
class Solution {
public:
    void solve(string ip, string op, vector<string>& ans) {
        // Base case
        if (ip.size() == 0) {
            ans.push_back(op);
            return;
        }

        string op1 = op + ip[0];         // Without space
        string op2 = op + " " + ip[0];   // With space

        // string remaining = ip.substr(1); // Create a new string for next level
        // solve(remaining, op1, ans);
        // solve(remaining, op2, ans);
        
        ip.erase(ip.begin() + 0);
        solve(ip, op1, ans);
        solve(ip, op2, ans);
    }

    vector<string> permutation(string s) {
        vector<string> ans;
        if (s.empty()) return ans;
        
        string op(1, s[0]);              // Start with the first character
        string ip = s.substr(1);         // Remaining string

        solve(ip, op, ans);
        sort(ans.begin(), ans.end());
        return ans;
    }
};
/*
Example 1:
Input:
s = "ABC"
Output: (A B C)(A BC)(AB C)(ABC)
Explanation:
ABC
AB C
A BC
A B C
These are the possible combination of "ABC".

Example 2:
Input:
s = "BBR"
Output: (B B R)(B BR)(BB R)(BBR)
*/


//14. PERMUTATIONS WITH CASE CHANGE                              {T.C = O(2^N * N), S.C = O(2^N * N)}
void solve(string ip, string op, vector<string>&ans){
    //base case
    if(ip.size() == 0){
        ans.push_back(op);                                 //its take space of o(2^n * n) for storing 
        return;
    }

    string op1 = op;
    string op2 = op;

    op1.push_back(tolower(ip[0]));
    op2.push_back(toupper(ip[0]));
    // ip.erase(it.begin() + 0)
    ip = ip.substr(1);                                  //remove first char of input (ip.erase(it.begin() + 0))

    solve(ip, op1, ans);
    solve(ip, op2, ans);
}
int main(){
    string ip;
    cin>>ip;
    string op = "";
    vector<string>ans;
    solve(ip, op, ans);

    for(auto it : ans) cout<<it<<" ";
    return 0;
}
/*
input - ab
output - ab aB Ab AB 
*/


//15. LOWER CASE PERMUTATION                            {T.C = O(2^N * N), S.C = O(2^N * N)}
class Solution {
public:
    void solve(string ip, string op, vector<string>&ans){
        //base case
        if(ip.size() == 0){
            ans.push_back(op);
            return;
        }

        if(isalpha(ip[0])){            //make 2 calls
            string op1 = op;
            string op2 = op;

            op1.push_back(tolower(ip[0]));
            op2.push_back(toupper(ip[0]));
            ip = ip.substr(1);

            solve(ip, op1, ans);
            solve(ip, op2, ans);
        }else{                           //digit direct push
            string op1 = op;
            op1.push_back(ip[0]);
            ip = ip.substr(1);

            solve(ip, op1, ans);
        }
    }
    vector<string> letterCasePermutation(string s) {
        string op = "";
        vector<string>ans;
        solve(s, op, ans);
        return ans;
    }
};
/*
Example 1:
Input: s = "a1b2"
Output: ["a1b2","a1B2","A1b2","A1B2"]

Example 2:
Input: s = "3z4"
Output: ["3z4","3Z4"]
*/


//16. GENERATE PARENTHESIS                                  {T.C = O(2^2N), S.C = O(2N)}
class Solution {
public:
    void solve(int ipOp, int ipCl, string op, vector<string>&ans){
        //base case
        if(ipOp == 0 && ipCl == 0){
            ans.push_back(op);
            return;
        }
        if(ipOp != 0){
            string op1 = op;
            op1.push_back('(');
            solve(ipOp-1, ipCl, op1, ans);           //ipOpen bracket count reduce
        }
        if(ipCl > ipOp){
            string op2 = op;
            op2.push_back(')');
            solve(ipOp, ipCl-1, op2, ans);
        }
    }
    vector<string> generateParenthesis(int n) {
        int open = n;
        int close = n;
        string op = "";
        vector<string>ans;
        solve(open, close, op, ans);
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