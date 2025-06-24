#include<bits/stdc++.h>
using namespace std;


void solve(string ip, string op, vector<string>&ans){
    //base case
    if(ip.size() == 0){
        ans.push_back(op);
        return;
    }

    string op1 = op;
    string op2 = op;

    op1.push_back(tolower(ip[0]));
    op2.push_back(toupper(ip[0]));
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