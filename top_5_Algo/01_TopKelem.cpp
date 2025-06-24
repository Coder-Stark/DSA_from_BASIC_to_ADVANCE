

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

struct Node {
    int data;
    Node* next;
    Node(int val) : data(val), next(nullptr) {}
};

// Utility: Insert at head
Node* insertHead(Node* head, int val) {
    Node* newNode = new Node(val);
    newNode->next = head;
    return newNode;
}

// Convert string to reversed linked list
Node* stringToList(const string& num) {
    Node* head = nullptr;
    for (int i = num.size() - 1; i >= 0; --i)
        head = insertHead(head, num[i] - '0');
    return head;
}

// Multiply two numbers represented by linked lists
Node* multiply(Node* l1, Node* l2) {
    if (!l1 || !l2) return new Node(0);

    // Convert linked lists to vectors
    vector<int> num1, num2;
    for (Node* p = l1; p; p = p->next) num1.push_back(p->data);
    for (Node* p = l2; p; p = p->next) num2.push_back(p->data);

    int n = num1.size(), m = num2.size();
    vector<int> result(n + m, 0);

    // Multiply
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result[i + j] += num1[i] * num2[j];
            if (result[i + j] >= 10) {
                result[i + j + 1] += result[i + j] / 10;
                result[i + j] %= 10;
            }
        }
    }

    // Remove leading zeros
    while (result.size() > 1 && result.back() == 0)
        result.pop_back();

    // Convert vector to linked list (reversed)
    Node* resHead = nullptr;
    for (int i = result.size() - 1; i >= 0; --i)
        resHead = insertHead(resHead, result[i]);

    return resHead;
}

// Print linked list
void printList(Node* head) {
    while (head) {
        cout << head->data;
        head = head->next;
    }
    cout << "\n";
}

// Driver code
int main() {
    string a, b;
    cout << "Enter first large number: ";
    cin >> a;
    cout << "Enter second large number: ";
    cin >> b;

    Node* num1 = stringToList(a);
    Node* num2 = stringToList(b);

    Node* product = multiply(num1, num2);

    cout << "Product: ";
    printList(product);
    return 0;
}


// 01. K LARGEST ELEMENTS

// BRUTE FORCE    (USING MAXHEAP [ALL ELE PUSH AT ONCE])       {T.C = O(N*LOGN), S.C = O(N)}
class Solution{
    public:
    vector<int> kLargest(vector<int> &arr, int k){
        int n = arr.size();
        priority_queue<int> pq;
        for (int i = 0; i < n; i++){
            pq.push(arr[i]);
        }
        vector<int> ans;
        while (k--){
            ans.push_back(pq.top());
            pq.pop();
        }
        return ans;
    }
};


//OPTIMIZED APPROACH (USING MINHEAP [PUSH ONLY K ELEM FIRST])    {T.C = O(N*LOGK), S.C = O(K)}
class Solution{
    public:
    vector<int> kLargest(vector<int> &arr, int k){
        int n = arr.size();
        priority_queue<int, vector<int>, greater<int>> pq; // minHeap
        for (int i = 0; i < k; i++) pq.push(arr[i]);       // push first k ele in minHeap
        
        for (int i = k; i < n; i++){                     // handle remaining ele
            if (arr[i] > pq.top()){                      // curr > topele (update heap)
                pq.pop();
                pq.push(arr[i]);
            }
        }
        
        vector<int> ans;
        while (!pq.empty()){
            ans.push_back(pq.top());
            pq.pop();
        }
        
        reverse(ans.begin(), ans.end()); // ans in reverse order
        return ans;
    }
};


//GENERAL APPROACH OR REFINEMENT OF ABOVE CODE
class Solution{
public:
    vector<int> kLargest(vector<int> &arr, int k){
        int n = arr.size();
        priority_queue<int, vector<int>, greater<int>> pq; // minHeap
        for (auto it : arr){
            pq.push(it);
            if (pq.size() > k) pq.pop();
        }

        vector<int> ans;
        while (!pq.empty()){
            ans.push_back(pq.top());
            pq.pop();
        }

        reverse(ans.begin(), ans.end()); // ans is in reverse order
        return ans;
    }
};
/*
Input: arr[] = [12, 5, 787, 1, 23], k = 2
Output: [787, 23]
Explanation: 1st largest element in the array is 787 and second largest is 23.

Input: arr[] = [1, 23, 12, 9, 30, 2, 50], k = 3 
Output: [50, 30, 23]
Explanation: Three Largest elements in the array are 50, 30 and 23.

Input: arr[] = [12, 23], k = 1
Output: [23]
Explanation: 1st Largest element in the array is 23.
*/


//02. KTH LARGEST ELEMENT IN AN ARRAY                             {T.C = O(N*LOGK), S.C = O(K)}
class Solution {
    public:
    //for finding largest build minheap and vice versa  
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int,vector<int>, greater<int>>minHeap;
        for(auto it : nums){
            minHeap.push(it);
            if(minHeap.size() > k) minHeap.pop();
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


//03. TOP K FREQUENT ELEMENTS
// BRUTE FORCE    (USING MAXHEAP [ALL ELE PUSH AT ONCE])       {T.C = O(N*LOGN), S.C = O(N)}
class Solution {
    public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int,int>mp;
        for(auto it : nums) mp[it]++;
        
        priority_queue<pair<int,int>>maxHeap;                 //freq, ele
        for(auto it : mp) maxHeap.push({it.second, it.first});
        
        vector<int>ans;
        while(k--){
            ans.push_back(maxHeap.top().second);              //second is element
            maxHeap.pop();
        }
        return ans;
    }
};


//OPTIMIZED APPROACH (USING MINHEAP [PUSH ONLY K ELEM FIRST])    {T.C = O(N*LOGK), S.C = O(K)}
class Solution {
public:
    typedef pair<int,int>P;
    vector<int> topKFrequent(vector<int>& nums, int k) {
        int n = nums.size();
        unordered_map<int,int>mp;
        for(auto it : nums) mp[it]++;                                  //ele, freq

        priority_queue<P, vector<P>, greater<P>>pq;          //minHeap (freq, ele)
        for(auto it : mp){
            pq.push({it.second, it.first});
            if(pq.size() > k) pq.pop();
        }
        
        vector<int>ans;
        while(!pq.empty()){
            ans.push_back(pq.top().second);
            pq.pop();
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
