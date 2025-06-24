//01. COMBINATION SUM                                            {T.C = O(2^N), S.C = O(N)} 
class Solution {
public:
    void solve(vector<int>&cand, vector<vector<int>>&ans, vector<int>&temp, int target, int i){
        int n = cand.size();
        //base case
        if(target == 0){
            ans.push_back(temp);
            return;
        }
        if(i >= n || target < 0) return;                     //invalid case

        temp.push_back(cand[i]);
        solve(cand, ans, temp, target - cand[i], i);        //incl (i not i+1 (unlimited time takes currCandidate))
        temp.pop_back();
        solve(cand, ans, temp, target, i+1);                //excl
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>>ans;
        vector<int>temp;
        solve(candidates, ans, temp, target, 0);                  //0 =  initial idx
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


//02. WORD LADDER                                            {T.C = O(M*26*N), S.C = O(N)}
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string>st(wordList.begin(), wordList.end());

        queue<pair<string, int>>q;              //word, steps
        q.push({beginWord, 1});    
        st.erase(beginWord);                    //after using word remove from wordList (1 word use only once)

        while(!q.empty()){
            auto frontNode = q.front();
            q.pop();
            string currWord = frontNode.first;
            int steps = frontNode.second;

            if(currWord == endWord) return steps;             //word transformed return its steps

            for(int i = 0 ; i < currWord.length(); i++){
                char origWordChar = currWord[i];

                for(char ch = 'a' ; ch <= 'z' ; ch++){
                    currWord[i] = ch;                       //update new char in currword (make new word)
                    if(st.count(currWord)){
                        q.push({currWord, steps+1});
                        st.erase(currWord);                 //if exact word found remove word from dictionary
                    }
                }
                currWord[i] = origWordChar;                 //backtrack
            }
        }      
        return 0;                                       //no sequence found
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


//03. PERMUTATION                                              {T.C = O(N!*N), S.C = O(N)}
class Solution {
public:
    void solve(vector<int>&nums, vector<vector<int>>&ans, int idx){
        int n = nums.size();
        //base case
        if(idx >= n){
            ans.push_back(nums);
            return;
        }

        for(int i = idx ; i < n; i++){
            swap(nums[idx], nums[i]);
            solve(nums, ans, idx+1);                   //idx+1 (not i+1)

            swap(nums[idx], nums[i]);                  //backtrack
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>>ans;
        solve(nums, ans, 0);                //0 = initial idx
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


//04. SUDOKU SOLVER                                            {T.C = O(9^N), S.C = O(N)}
class Solution {
public:
    bool isValid(vector<vector<char>>&board, int row , int col, int ch){
        for(int i = 0 ; i < 9 ; i++){
            if(board[row][i] == ch) return false;        //check all cols have ch or not
                
            if(board[i][col] == ch) return false;        //check all rows have ch or not
                
            if(board[3*(row/3) + i/3][3*(col/3) + i % 3] == ch) return false;   //check all 3x3 sub boxes have ch or not
        }
        return true;
    }
    bool solve(vector<vector<char>>&board){
        int n = board.size(), m = board[0].size();

        for(int i = 0 ; i < n; i++){
            for(int j = 0 ; j < m; j++){
                if(board[i][j] == '.'){
                    for(char ch = '1' ; ch <= '9' ; ch++){
                        if(isValid(board, i, j, ch)){
                            board[i][j] = ch;
                            if(solve(board) == true) return true;

                            board[i][j] = '.';               //reset (backtrack)
                        }
                    }
                    return false;
                }
            }
        }
        return true;                                    //already filled all
    }
    void solveSudoku(vector<vector<char>>& board) {
        solve(board);
    }
};
/*
Input: board = [["5","3",".",".","7",".",".",".","."],
                ["6",".",".","1","9","5",".",".","."],
                [".","9","8",".",".",".",".","6","."],
                ["8",".",".",".","6",".",".",".","3"],
                ["4",".",".","8",".","3",".",".","1"],
                ["7",".",".",".","2",".",".",".","6"],
                [".","6",".",".",".",".","2","8","."],
                [".",".",".","4","1","9",".",".","5"],
                [".",".",".",".","8",".",".","7","9"]]
Output: [["5","3","4","6","7","8","9","1","2"],
         ["6","7","2","1","9","5","3","4","8"],
         ["1","9","8","3","4","2","5","6","7"],
         ["8","5","9","7","6","1","4","2","3"],
         ["4","2","6","8","5","3","7","9","1"],
         ["7","1","3","9","2","4","8","5","6"],
         ["9","6","1","5","3","7","2","8","4"],
         ["2","8","7","4","1","9","6","3","5"],
         ["3","4","5","2","8","6","1","7","9"]]
*/