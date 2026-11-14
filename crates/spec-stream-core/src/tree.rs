use crate::error::CoreError;
use crate::logits::{Logits, TokenId};

/// Node in a speculative candidate tree (used in tree-based speculative decoding like Medusa / Eagle).
#[derive(Debug, Clone, PartialEq)]
pub struct TreeNode {
    pub token_id: TokenId,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub depth: usize,
    pub cumulative_prob: f32,
}

/// The result of verifying a speculative tree against the target model.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeVerificationResult {
    /// The token IDs along the best accepted path (excluding the root).
    pub accepted_tokens: Vec<TokenId>,
    /// The node indices in the tree corresponding to the accepted path.
    pub accepted_node_indices: Vec<usize>,
    /// The token emitted by the target model from the terminal accepted node.
    pub emitted_target_token: TokenId,
    /// Total number of tokens produced in this cycle.
    pub total_tokens_produced: usize,
}

/// Tree structure representing multiple branch proposals for speculative decoding.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeTree {
    pub nodes: Vec<TreeNode>,
}

impl SpeculativeTree {
    pub fn new(root_token: TokenId) -> Self {
        Self {
            nodes: vec![TreeNode {
                token_id: root_token,
                parent: None,
                children: Vec::new(),
                depth: 0,
                cumulative_prob: 1.0,
            }],
        }
    }

    /// Adds a child branch to an existing node.
    pub fn add_child(
        &mut self,
        parent_idx: usize,
        token_id: TokenId,
        prob: f32,
    ) -> Result<usize, CoreError> {
        if parent_idx >= self.nodes.len() {
            return Err(CoreError::TreeError(format!(
                "Parent node index {parent_idx} out of range"
            )));
        }

        let parent_depth = self.nodes[parent_idx].depth;
        let parent_prob = self.nodes[parent_idx].cumulative_prob;
        let new_idx = self.nodes.len();

        self.nodes.push(TreeNode {
            token_id,
            parent: Some(parent_idx),
            children: Vec::new(),
            depth: parent_depth + 1,
            cumulative_prob: parent_prob * prob,
        });

        self.nodes[parent_idx].children.push(new_idx);
        Ok(new_idx)
    }

    /// Generates the 2D Boolean causal/tree attention mask for parallel target model execution.
    /// Returns an NxN boolean matrix where `mask[i][j] == true` if node `i` can attend to node `j`.
    pub fn compute_tree_attention_mask(&self) -> Vec<Vec<bool>> {
        let n = self.nodes.len();
        let mut mask = vec![vec![false; n]; n];

        for (i, row) in mask.iter_mut().enumerate().take(n) {
            // A node can always attend to itself
            row[i] = true;

            // And to all its ancestors in the tree
            let mut curr = self.nodes[i].parent;
            while let Some(parent_idx) = curr {
                row[parent_idx] = true;
                curr = self.nodes[parent_idx].parent;
            }
        }

        mask
    }

    /// Returns the flattened list of token IDs in tree order.
    pub fn token_ids(&self) -> Vec<TokenId> {
        self.nodes.iter().map(|n| n.token_id).collect()
    }

    /// Verifies the entire candidate tree against the target model's logits.
    /// Finds the longest valid path where the target model's prediction at each parent matches the child token.
    pub fn verify_greedy(
        &self,
        target_logits: &[Logits],
    ) -> Result<TreeVerificationResult, CoreError> {
        let n = self.nodes.len();
        if target_logits.len() < n {
            return Err(CoreError::DimensionMismatch {
                expected: n,
                actual: target_logits.len(),
            });
        }

        // Compute argmax predictions for each node in the tree
        let mut target_preds = Vec::with_capacity(n);
        for logits in target_logits.iter().take(n) {
            target_preds.push(logits.argmax()?);
        }

        // Traverse using DFS / BFS to find the longest accepted branch
        let mut best_path = Vec::new();
        let mut stack = vec![(0usize, Vec::new())]; // (current_node_idx, current_path)

        while let Some((node_idx, path)) = stack.pop() {
            let mut extended = false;
            let expected_next = target_preds[node_idx];

            for &child_idx in &self.nodes[node_idx].children {
                // If child matches the target model's prediction from node_idx
                if self.nodes[child_idx].token_id == expected_next {
                    let mut child_path = path.clone();
                    child_path.push(child_idx);
                    stack.push((child_idx, child_path));
                    extended = true;
                }
            }

            if !extended && path.len() > best_path.len() {
                best_path = path;
            }
        }

        // The terminal node of the best accepted path (or root 0 if no children accepted)
        let terminal_node = best_path.last().copied().unwrap_or(0);
        let emitted_token = target_preds[terminal_node];

        let accepted_tokens: Vec<TokenId> = best_path
            .iter()
            .map(|&idx| self.nodes[idx].token_id)
            .collect();

        let total_tokens_produced = accepted_tokens.len() + 1;

        Ok(TreeVerificationResult {
            accepted_tokens,
            accepted_node_indices: best_path,
            emitted_target_token: emitted_token,
            total_tokens_produced,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logits(vocab_size: usize, target_token: usize) -> Logits {
        let mut vals = vec![0.0f32; vocab_size];
        vals[target_token] = 10.0;
        Logits::new(vals)
    }

    #[test]
    fn test_tree_attention_mask() {
        let mut tree = SpeculativeTree::new(100);
        let a = tree.add_child(0, 101, 0.8).unwrap();
        let _b = tree.add_child(a, 102, 0.7).unwrap();
        let _c = tree.add_child(0, 103, 0.2).unwrap();

        let mask = tree.compute_tree_attention_mask();
        assert_eq!(mask.len(), 4);

        assert!(mask[2][2]);
        assert!(mask[2][1]);
        assert!(mask[2][0]);
        assert!(!mask[2][3]);

        assert!(mask[3][3]);
        assert!(mask[3][0]);
        assert!(!mask[3][1]);
        assert!(!mask[3][2]);
    }

    #[test]
    fn test_tree_speculative_verification() {
        // Tree:
        // Root (0, id=10)
        //   ├── Branch 1: Child A (1, id=20) -> Child B (2, id=30)
        //   └── Branch 2: Child C (3, id=40) -> Child D (4, id=50)
        let mut tree = SpeculativeTree::new(10);
        let a = tree.add_child(0, 20, 0.5).unwrap();
        let _b = tree.add_child(a, 30, 0.5).unwrap();
        let c = tree.add_child(0, 40, 0.5).unwrap();
        let _d = tree.add_child(c, 50, 0.5).unwrap();

        // Suppose Root predicts 40 (Branch 2 chosen), Child C predicts 50, and Child D predicts 99 (Bonus token)
        let target_logits = vec![
            make_logits(100, 40), // From Root -> matches Child C (3)
            make_logits(100, 99), // Child A (ignored)
            make_logits(100, 99), // Child B (ignored)
            make_logits(100, 50), // From Child C -> matches Child D (4)
            make_logits(100, 77), // From Child D -> Bonus token = 77
        ];

        let result = tree.verify_greedy(&target_logits).unwrap();
        assert_eq!(result.accepted_tokens, vec![40, 50]);
        assert_eq!(result.accepted_node_indices, vec![3, 4]);
        assert_eq!(result.emitted_target_token, 77);
        assert_eq!(result.total_tokens_produced, 3);
    }
}
