use crate::{
    collections::HashMap,
    graph::{ComputingProperty, NodeID, NodeSteps},
    tensor::AutodiffTensor,
};
use alloc::{boxed::Box, sync::Arc, vec::Vec};
use burn_tensor::backend::Backend;
use core::any::Any;

use super::{
    base::{Checkpointer, NodeTree},
    retro_forward::{RetroForward, RetroForwards},
    state::{BackwardStates, State},
};

#[derive(Debug)]
/// Determines if a node should checkpoint its computed output or its retro_forward for recomputation
/// The action is normally created by the child of the node, once the node is determined to be needed
enum CheckpointingAction {
    /// The node's already computed output should be saved
    Computed(Box<dyn Any + Send>),
    /// The node should recompute itself when asked
    Recompute(Arc<dyn RetroForward>),
}

// TODO: Remove that when proper client server.
unsafe impl Send for CheckpointingAction {}

#[derive(new, Debug, Default)]
/// Accumulates checkpoints as checkpointing actions during the forward pass,
/// and builds a checkpointer right before the backward pass
pub struct CheckpointerBuilder {
    explicit_actions: Vec<(NodeID, CheckpointingAction)>,
    backup_actions: Vec<(NodeID, CheckpointingAction)>,
}

/// Determines if a checkpoint should impact the n_required values (Main)
/// or if it should just keep the state in case it's required (Backup)
///
pub(crate) enum ActionType {
    /// Explicit actions have been explicitly requested by some operation to retrieve their state
    Explicit,
    /// Backup actions are not always needed. They exist to save the output of an operation
    /// whose child is memory bound, in case the state is indirectly needed when computing
    /// the child's retro_forward. If no explicit action ever asks for the child's output, then
    /// the backup output will go out of scope when the checkpointer is built.
    Backup,
}

impl CheckpointerBuilder {
    pub(crate) fn checkpoint<B: Backend>(
        &mut self,
        tensor: &AutodiffTensor<B>,
        action_type: ActionType,
    ) {
        let action_list = match action_type {
            ActionType::Explicit => &mut self.explicit_actions,
            ActionType::Backup => &mut self.backup_actions,
        };
        match &tensor.node.properties {
            ComputingProperty::ComputeBound | ComputingProperty::Ambiguous => action_list.push((
                tensor.node.id,
                CheckpointingAction::Computed(Box::new(tensor.primitive.clone())),
            )),
            ComputingProperty::MemoryBound { retro_forward } => action_list.push((
                tensor.node.id,
                CheckpointingAction::Recompute(retro_forward.clone()),
            )),
        }
    }

    pub(crate) fn extend(&mut self, other: CheckpointerBuilder) {
        for other_action in other.explicit_actions {
            self.explicit_actions.push(other_action)
        }
        for other_unsure in other.backup_actions {
            self.backup_actions.push(other_unsure)
        }
    }

    pub(crate) fn build(self, graph: &NodeSteps) -> Checkpointer {
        let node_tree = self.make_tree(graph);
        let mut backward_states_map = HashMap::new();
        let mut retro_forwards_map = HashMap::new();

        // Find recursion stopping points
        let stop_nodes: Vec<NodeID> = self.find_stop_nodes();

        // We start by identifying how many times each node will be required.
        let n_required_map = self.build_n_required_map(&node_tree, stop_nodes);

        // Then we checkpoint the nodes with the corresponding n_required value
        self.insert_checkpoints(
            &mut backward_states_map,
            &mut retro_forwards_map,
            n_required_map,
        );

        Checkpointer::new(
            BackwardStates::new(backward_states_map),
            RetroForwards::new(retro_forwards_map),
            node_tree,
        )
    }

    fn find_stop_nodes(&self) -> Vec<NodeID> {
        let stop_nodes = self
            .explicit_actions
            .iter()
            .chain(self.backup_actions.iter())
            .filter_map(|(id, action)| match action {
                CheckpointingAction::Computed(_) => Some(*id),
                _ => None,
            })
            .collect();
        stop_nodes
    }

    fn build_n_required_map(
        &self,
        node_tree: &NodeTree,
        stop_nodes: Vec<NodeID>,
    ) -> HashMap<NodeID, usize> {
        let mut n_required_map = HashMap::<NodeID, usize>::default();
        for (node_id, action) in self.explicit_actions.iter() {
            match action {
                CheckpointingAction::Computed(_) => match n_required_map.get_mut(node_id) {
                    Some(n) => *n += 1,
                    None => {
                        n_required_map.insert(*node_id, 1);
                    }
                },
                CheckpointingAction::Recompute(_) => Self::update_n_required_of_parents(
                    node_id,
                    &mut n_required_map,
                    node_tree,
                    &stop_nodes,
                ),
            }
        }
        n_required_map
    }

    fn insert_checkpoints(
        self,
        backward_states_map: &mut HashMap<NodeID, State>,
        retro_forward_map: &mut HashMap<NodeID, Arc<dyn RetroForward>>,
        mut n_required_map: HashMap<NodeID, usize>,
    ) {
        // loop over chained explicit and backup actions, inserting a checkpoint
        // for the first occurrence of each id
        for (node_id, action) in self
            .explicit_actions
            .into_iter()
            .chain(self.backup_actions.into_iter())
        {
            // skip checkpointing of repeat node ids
            let Some(n_required) = n_required_map.remove(&node_id) else {
                continue;
            };
            match action {
                CheckpointingAction::Computed(state_content) => {
                    // insert computed checkpoint
                    backward_states_map.insert(
                        node_id,
                        State::Computed {
                            state_content,
                            n_required,
                        },
                    );
                }
                CheckpointingAction::Recompute(retro_forward) => {
                    // insert lazy checkpoint
                    retro_forward_map.insert(node_id, retro_forward);
                    backward_states_map.insert(node_id, State::Recompute { n_required });
                }
            }
        }
    }

    fn make_tree(&self, graph: &NodeSteps) -> NodeTree {
        let mut tree = HashMap::default();
        for (id, step) in graph {
            tree.insert(*id, step.parents());
        }
        NodeTree::new(tree)
    }

    fn update_n_required_of_parents(
        id: &NodeID,
        n_required_map: &mut HashMap<NodeID, usize>,
        node_tree: &NodeTree,
        stop_nodes: &Vec<NodeID>,
    ) {
        match n_required_map.get_mut(id) {
            Some(n) => *n += 1,
            None => {
                n_required_map.insert(*id, 1);
                if !stop_nodes.contains(id) {
                    if let Some(parents) = node_tree.parents(id) {
                        for p in parents {
                            Self::update_n_required_of_parents(
                                &p,
                                n_required_map,
                                node_tree,
                                stop_nodes,
                            );
                        }
                    }
                }
            }
        }
    }
}
