/**
 * https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/
 */

/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* prev;
    Node* next;
    Node* child;
};
*/


class Solution {
public:
    Node* flatten(Node* root) {
        if (!root) return root;

        Node *a = root->next;
        Node *b = root->child;
        if (a) flatten(a);
        if (b) flatten(b);
        
        a = root->next;
        b = root->child;
        if (b) {
            //Move child to next
            root->next = b;
            b->prev = root;
            root->child = nullptr;
            //Append original next
            Node *tmp = b;
            while ( tmp->next ) tmp = tmp -> next;
            tmp->next = a;
            if (a) tmp->next->prev = tmp;
        }
        return root;
    }
};
