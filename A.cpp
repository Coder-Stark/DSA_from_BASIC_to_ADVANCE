#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <set>
#include <queue>
using namespace std;

struct Node {
    string name;
    Node* parent;
    vector<Node*> children;
    int lockedBy;
    int lockedDescendantCount;
    Node(string n) : name(n), parent(nullptr), lockedBy(-1), lockedDescendantCount(0) {}
};

unordered_map<string, Node*> nodes;

// Checks if any ancestors are locked
bool hasLockedAncestor(Node* node) {
    while (node) {
        if (node->lockedBy != -1) return true;
        node = node->parent;
    }
    return false;
}

// Checks if any descendants are locked
bool hasLockedDescendant(Node* node) {
    return node->lockedDescendantCount > 0;
}

// Lock operation
/*
Locking Rules:
1. Node must not be already locked
2. No ancestor can be locked
3. No descendant can be locked
*/
bool lock(Node* node, int uid) {                                 //{T.C = O(H), S.C = O(1)}
    if (node->lockedBy != -1 || hasLockedAncestor(node) || hasLockedDescendant(node))
        return false;
    
    node->lockedBy = uid;                  //lock the node
    //update ancestor count
    Node* temp = node->parent;
    while (temp) {
        temp->lockedDescendantCount++;
        temp = temp->parent;
    }
    return true;
}

// Unlock operation
/*
Unlocking Rules:
1. Only the user who locked it can unlock
2. Must update ancestor counts
*/
bool unlock(Node* node, int uid) {                                 //{T.C = O(H), S.C = O(1)}
    if (node->lockedBy != uid)
        return false;
    
    node->lockedBy = -1;                     //unlock the node
    //update ancestor counts
    Node* temp = node->parent;
    while (temp) {
        temp->lockedDescendantCount--;
        temp = temp->parent;
    }
    return true;
}

// Upgrade lock
/*
Upgrade Rules:
1. Target node must be unlocked
2. No ancestors can be locked
3. Must have at least one locked descendant
4. All locked descendants must belong to the same user
*/
bool upgrade(Node* node, int uid) {                                     //{T.c = o(subtree size), S.c = (subtree size)}
    if (node->lockedBy != -1 || hasLockedAncestor(node) == true || node->lockedDescendantCount == 0)
        return false;
    
    queue<Node*> q;
    q.push(node);
    vector<Node*> lockedNodes;
    
    while (!q.empty()) {
        Node* curr = q.front();
        q.pop();
        if (curr->lockedBy != -1) {
            if (curr->lockedBy != uid) return false;
            lockedNodes.push_back(curr);
        }
        for (auto child : curr->children)
            q.push(child);
    }
    
    for (auto lockedNode : lockedNodes)
        unlock(lockedNode, uid);
    
    return lock(node, uid);
}

int main() {
    int N, m, Q;                           //n = nodes, m = branching factor, q = queries
    cin >> N >> m >> Q;
    vector<string> names(N);
    
    for (int i = 0; i < N; ++i) {          //read node names
        cin >> names[i];
        nodes[names[i]] = new Node(names[i]);
    }
    
    for (int i = 1; i < N; ++i) {          //build tree structure
        nodes[names[(i - 1) / m]]->children.push_back(nodes[names[i]]);
        nodes[names[i]]->parent = nodes[names[(i - 1) / m]];
    }
    
    while (Q--) {
        int op, uid;
        string name;
        cin >> op >> name >> uid;
        Node* node = nodes[name];
        bool result = false;
        if (op == 1)
            result = lock(node, uid);
        else if (op == 2)
            result = unlock(node, uid);
        else if (op == 3)
            result = upgrade(node, uid);
        
        cout << (result ? "true" : "false") << endl;
    }
    return 0;
}