/**
 * https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
 */

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void flatten(TreeNode* root) {
        if (!root) return;

        TreeNode* a = root->left;
        TreeNode *b = root->right;
        if (a) flatten(a);
        if (b) flatten(b);
        
        a = root->left;
        b = root->right;
        if (a) {
            //Move left to right
            root->right = root->left;
            root->left = nullptr;
            //Append right
            TreeNode *tmp = a;
            while ( tmp->right ) tmp = tmp -> right;
            tmp->right = b;
        }
    }
};
