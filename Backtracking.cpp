//BACKTRACKING

#include<bits/stdc++.h>
using namespace std;

//BACTRACKING = RECURSION + CONTROL + PASS BY REFERENCE
/*
Identification

1. Choice + Decision
2. All Combination
3. Many Number of Choices
4. Constraints = 1 < n < 10       (single digits)

Suedo code

void solve(&variable, ...){              //by refernce for variable(main)
    //base case
    if(isSolved()){
        print / store;
    }

    //handle all choices
    for(all choices){
        if(isValid()){
            change in variable
            solve(..., ...);
            revert changes in variable
        }
    }
}   
*/


//01. PERMUTATION OF A STRING                                     {T.C = O(N! * N^2), S.C = O(N! * N^2)}
//using recursion
class Solution {
    public:
    void solveRec(string ip, string op, vector<string>&ans){
        //base case
        if(ip.size() == 0){
            ans.push_back(op);
            return;
        }
        unordered_set<char>st;
        for(int i = 0 ; i < ip.size(); i++){
            if(!st.count(ip[i])){
                string newIp = ip.substr(0, i) + ip.substr(i+1);     //ip[i] is removed and add further chr
                string newOp = op + ip[i];
                
                solveRec(newIp, newOp, ans);
            }
            st.insert(ip[i]);
        }
        
    }
    vector<string> findPermutation(string &s) {
        string op = "";
        vector<string>ans;
        solveRec(s, op, ans);
        return ans;
    }
};

//PERMUTATION OF A STRING                                     {T.C = O(N! * N), S.C = O(N! * N)}
//using backtracking
class Solution {
  public:
    void solveBack(string &s, int startIdx, vector<string>&ans){
        int n = s.length();
        //base case
        if(startIdx == n-1){
            ans.push_back(s);
            return;
        }
        
        unordered_set<char>st;
        for(int i = startIdx; i < n ; i++){
            if(!st.count(s[i])){
                swap(s[i], s[startIdx]);
                solveBack(s, startIdx + 1, ans);
                swap(s[i], s[startIdx]);               //backtrack
            }
            st.insert(s[i]);
        }
    }
    vector<string> findPermutation(string &s) {
        vector<string>ans;
        solveBack(s, 0, ans);                  //0 = start Idx
        return ans;
    }
};
/*
Input: s = "ABC"
Output: ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]
Explanation: Given string ABC has 6 unique permutations.

Input: s = "ABSG"
Output: ["ABGS", "ABSG", "AGBS", "AGSB", "ASBG", "ASGB", "BAGS", "BASG", "BGAS", "BGSA", "BSAG", "BSGA", "GABS", "GASB", "GBAS", "GBSA", "GSAB", "GSBA", "SABG", "SAGB", "SBAG", "SBGA", "SGAB", "SGBA"]
Explanation: Given string ABSG has 24 unique permutations.

Input: s = "AAA"
Output: ["AAA"]
Explanation: No other unique permutations can be formed as all the characters are same.
*/


//02. LARGEST NUMBER IN K SWAPS                           {T.C = O(N!/(N-K)! * N^2), S.C = O(N)}
class Solution {
  public:
    // Function to find the largest number after k swaps.
    void solve(string &s, int k, int startIdx, string &ans){
        int n = s.size();
        //base case
        if(startIdx == n-1 || k == 0){
            return;                       //not return ans cause (at most k not exact k)
        }
        char maxEle = *max_element(s.begin()+startIdx+1, s.end());
        
        for(int i = startIdx + 1; i < n; i++){
            if(s[startIdx] < s[i] && s[i] == maxEle){
                swap(s[startIdx], s[i]);
                if(s.compare(ans) > 0) ans = s;
                solve(s, k-1, startIdx+1, ans);
                
                swap(s[startIdx], s[i]);
            }
        }
        solve(s, k , startIdx+1, ans);            //horizontal driftign of node
    }
    string findMaximumNum(string& s, int k) {
        string ans = s;
        solve(s, k, 0, ans);               //0 = startIdx
        return ans;
    }
};
/*
Input: s = "1234567", k = 4
Output: 7654321
Explanation: Three swaps can make the input 1234567 to 7654321, swapping 1 with 7, 2 with 6 and finally 3 with 5.

Input: s = "3435335", k = 3
Output: 5543333
Explanation: Three swaps can make the input 3435335 to 5543333, swapping 3 with 5, 4 with 5 and finally 3 with 4.

Input: s = "1034", k = 2
Output: 4301
Explanation: Two swaps can make the input 1034 to 4301, swapping 1 with 4 and finally 0 with 3. 
*/


//03. N DIGIT NUMBER WITH DIGIT IN INCREASING ORDER                                {T.C = O(9^N), S.C = O(N)}
class Solution {
  public:
    void solve(int n, vector<int>&v, vector<int>&ans){
        //base case
        if(n == 0){
            int num = 0;
            for(int i = 0 ; i < v.size(); i++){
                num = num * 10 + v[i];
            }
            ans.push_back(num);
            return;
        }
        
        for(int i = 1; i <= 9; i++){
            if(v.empty() || i > v.back()){
                v.push_back(i);
                solve(n-1, v, ans);
                
                v.pop_back();
            }
        }
        
    }
    vector<int> increasingNumbers(int n) {
        vector<int>ans; 
        if(n == 1){
            for(int i = 0 ; i <= 9; i++) ans.push_back(i);
            return ans;
        }
        vector<int>v;
        solve(n, v, ans);
        return ans;
        
    }
};
/*
Example 1:
Input:
n = 1
Output:
0 1 2 3 4 5 6 7 8 9
Explanation:
Single digit numbers are considered to be 
strictly increasing order.

Example 2:
Input:
n = 2
Output:
12 13 14 15 16 17 18 19 23....79 89
Explanation:
For n = 2, the correct sequence is
12 13 14 15 16 17 18 19 23 and so on 
up to 89.
*/


//04. RAT IN A MAZE                                                   {T.C = O(4^(N*M)), S.C = O(N*M)}
class Solution {
  public:
    bool isValid(int i, int j, int n, int m) {
        return (i >= 0 && i < n && j >= 0 && j < m);
    }

    void solve(vector<vector<int>>& matrix, string path, vector<string>& ansPaths, vector<vector<bool>>& vis, int i, int j) {
        int n = matrix.size(), m = matrix[0].size();

        if (!isValid(i, j, n, m) || matrix[i][j] == 0 || vis[i][j]) return;

        if (i == n - 1 && j == m - 1) {
            ansPaths.push_back(path);
            return;
        }

        vis[i][j] = true;

        vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        string dirChar = "UDLR";

        for (int d = 0; d < 4; d++) {
            int newR = i + directions[d].first;
            int newC = j + directions[d].second;
            solve(matrix, path + dirChar[d], ansPaths, vis, newR, newC);
        }

        vis[i][j] = false; // backtrack
    }

    vector<string> ratInMaze(vector<vector<int>>& maze) {
        int n = maze.size(), m = maze[0].size();
        vector<string> ansPaths;
        vector<vector<bool>> vis(n, vector<bool>(m, false));

        if (maze[0][0] == 0) return {};

        solve(maze, "", ansPaths, vis, 0, 0);
        sort(ansPaths.begin(), ansPaths.end());
        return ansPaths;
    }
};
/*
Input: mat[][] = [[1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 0, 0], [0, 1, 1, 1]]
Output: ["DDRDRR", "DRDDRR"]
Explanation: The rat can reach the destination at (3, 3) from (0, 0) by two paths - DRDDRR and DDRDRR, when printed in sorted order we get DDRDRR DRDDRR.

Input: mat[][] = [[1, 0], [1, 0]]
Output: []
Explanation: No path exists as the destination cell is blocked.

Input: mat = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
Output: ["DDRR", "RRDD"]
Explanation: The rat has two possible paths to reach the destination: 1. "DDRR" 2. "RRDD", These are returned in lexicographically sorted order.
*/


//05. FIND ALL POSSIBLE PALINDROMIC PARTITIONS OF A STRING               {T.C = O(2^N * N), S.C = O(N^2)}
class Solution {
  public:
    bool isPalindrome(string &s){
        int i = 0, j = s.length()-1;
        while(i < j){
            if(s[i] != s[j]) return false;
            i++, j--;
        }
        return true;
    }
    void solveBack(string &s, vector<string>&temp, vector<vector<string>>&ans, int startIdx){
        int n = s.length();
        //base case
        if(startIdx == n){
            ans.push_back(temp);
            return;
        }
        string tempStr = "";
        for(int i = startIdx; i < n; i++){
            tempStr += s[i];
            if(isPalindrome(tempStr)){
                temp.push_back(tempStr);
                solveBack(s, temp, ans, i+1);           //startIdx = i+1
                
                temp.pop_back();                       //backtrack
            }
        }
    }
    vector<vector<string>> allPalindromicPerms(string S) {
        vector<string>temp;
        vector<vector<string>>ans;
        solveBack(S, temp, ans, 0);           //0 = initial idx
        return ans;
    }
};
/*
Example 1:
Input:
S = "geeks"
Output:
g e e k s
g ee k s
Explanation:
All possible palindromic partitions
are printed.

Example 2:
Input:
S = "madam"
Output:
m a d a m
m ada m
madam
*/


//06. WORD BREAK-1                                                      {T.C = O(2^N), S.C = O(N)}
class Solution {
  public:
    void solveBack(string &s, unordered_set<string>&st, bool &ans, int startIdx){
        int n = s.length();
        //base case
        if(startIdx == n){
            ans = true;
            return;
        }
        string tempStr = "";
        for(int i = startIdx; i < n; i++){
            tempStr += s[i];
            if(st.count(tempStr)){
                solveBack(s, st, ans, i+1);
                if(ans) return;
                
            }
        }
    }
    bool wordBreak(string &s, vector<string> &dictionary) {
        unordered_set<string>st(dictionary.begin(), dictionary.end());
        bool ans = false;
        solveBack(s, st, ans, 0);                   //ans(bool) = true = initial idx
        return ans;
    }
};
/*
Input: s = "ilike", dictionary[] = ["i", "like", "gfg"]
Output: true
Explanation: s can be breakdown as "i like".

Input: s = "ilikegfg", dictionary[] = ["i", "like", "man", "india", "gfg"]
Output: true
Explanation: s can be breakdown as "i like gfg".

Input: s = "ilikemangoes", dictionary[] = ["i", "like", "man", "india", "gfg"]
Output: false
Explanation: s cannot be formed using dictionary[] words.
*/


//07. WORD BREAK - 2                                         {T.C = O(2^N), S.C = O(N)}
class Solution {
  public:
    void solveBack(string &s, unordered_set<string>&st, vector<string>&ans, string temp, int startIdx){
        int n = s.length();
        //base case
        if(startIdx == n){
            if(!temp.empty() && temp.back() == ' ') temp.pop_back();      //remove trailing zeroes
            ans.push_back(temp);
            return;
        }
        string tempStr = "";
        for(int i = startIdx ; i < n; i++){
            tempStr += s[i];
            if(st.count(tempStr)){
                solveBack(s, st, ans, temp + tempStr + " ", i+1);
            }
        }
    }
    vector<string> wordBreak(vector<string>& dict, string& s) {
        unordered_set<string>st(dict.begin(), dict.end());
        vector<string>ans;
        string temp = "";
        solveBack(s, st, ans, temp , 0);             //0 = startIdx
        return ans;
    }
};
/*
Input: s = "likegfg", dict[] = ["lik", "like", "egfg", "gfg"]
Output: 
"lik egfg"
"like gfg"
Explanation: All the words in the given sentences are present in the dictionary.

Input: s = "geeksforgeeks", dict[] = ["for", "geeks"]
Output: "geeks for geeks"
Explanation: The string "geeksforgeeks" can be broken into valid words from the dictionary in one way.
*/


//08. LETTER COMBINATION OF A PHONE NUMBER                                   {T.C = O(4^N * N), S.C = O(4^N  * N)}
class Solution {
public:
    void solveBack(string &s, unordered_map<char,string>&mp, vector<string>&ans, string &temp, int startIdx){
        int n = s.length();
        //base case
        if(startIdx == n){
            ans.push_back(temp);
            return;
        }

        string letters = mp[s[startIdx]];
        for(auto it : letters){
            temp.push_back(it);
            solveBack(s, mp, ans, temp, startIdx+1);
            temp.pop_back();                        //backtrack
        }

    }
    vector<string> letterCombinations(string digits) {
        //base case
        if(digits.empty()) return {};
        
        unordered_map<char,string>mp = {
            {'2', "abc"},
            {'3', "def"},
            {'4', "ghi"},
            {'5', "jkl"},
            {'6', "mno"},
            {'7', "pqrs"},
            {'8', "tuv"},
            {'9', "wxyz"}
        };
        vector<string>ans;
        string temp = "";
        solveBack(digits, mp, ans, temp, 0);          //0 = startIdx
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

//09. N-QUEEN


//10. SUDUKO SOLVER

