use crate::Shape;
use crate::backend::Backend;
use crate::ops::FloatTensor;
use core::ops::Range;
use serde::{Deserialize, Serialize};

/// Long Short-Term Memory operations
pub mod lstm;

/// Struct describing the size of an RNN problem
#[derive(new, Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub struct RnnSize {
    /// Sequence size
    pub seq_d: usize,
    /// Batch size
    pub bat_d: usize,
    /// Input size
    pub inp_d: usize,
    /// Hidden size
    pub hid_d: usize,
}

impl RnnSize {
    /// Shape of an output trajectory of RNN states
    pub fn traj_shape(&self) -> Shape {
        Shape {
            dims: vec![self.seq_d, self.bat_d, self.hid_d],
        }
    }
    /// Shape of a single RNN state
    pub fn state_shape(&self) -> Shape {
        Shape {
            dims: vec![1, self.bat_d, self.hid_d],
        }
    }
    /// Element range within an output trajectory of RNN states
    pub const fn traj_elem_range(&self, i: usize) -> [Range<usize>; 3] {
        [i..i + 1, 0..self.bat_d, 0..self.hid_d]
    }
    /// Gate range within a trajectory of flattened transitions
    pub const fn gate_range(&self, i: usize, j: usize) -> [Range<usize>; 3] {
        [
            i..i + 1,
            0..self.bat_d,
            self.hid_d * j..self.hid_d * (j + 1),
        ]
    }
}

#[derive(Clone, Debug)]
/// Enum describing the RNN cell
pub enum RnnCell<B: Backend> {
    /// LSTM Cell, stateful [1, bat_d, hid_d]
    Lstm(FloatTensor<B>),
    /// GRU Cell, stateless
    Gru,
}

impl<B: Backend> RnnCell<B> {
    /// Get count of gates used in the RNN Cell variant
    ///
    /// ## NOTE
    ///
    /// This is actually the number of independant activations (LSTM eg. input, forget, output, cell)
    /// derived from the input. This number may not match literature definitions. (LSTM eg. input, forget, output)
    pub const fn gate_count(&self) -> usize {
        match self {
            RnnCell::Lstm { .. } => 4,
            RnnCell::Gru => 3,
        }
    }
}

/// Complete trajectory output of RNN forward
#[derive(new, Debug, Clone)]
pub struct RnnTrajectory<B: Backend> {
    /// Trajectory of stacked recurrent states `[d_sequence + 1, d_batch, d_hidden * n_gates]`
    pub traj: FloatTensor<B>,
    /// Final hidden state `[1, d_batch, d_hidden]`
    pub hidden_state: FloatTensor<B>,
    /// Optional final cell state `[1, d_batch, d_hidden]`
    pub cell: RnnCell<B>,
    /// Optional cached tensors for accelerated backprop
    pub cache: Option<Vec<FloatTensor<B>>>,
}

/// Rnn operations trait
pub trait RnnOps<B: Backend> {
    /// Run an RNN forward over the input sequence using the given weights, biases and state.
    ///
    /// # Arguments:
    ///
    /// - `input` input tensor (x) ``[d_sequence, d_batch, d_input]``
    /// - `hidden_state` hidden state (h) ``[1, d_batch, d_hidden]``
    /// - `cell_state` optional cell state (c) ``[1, d_batch, d_hidden]``
    /// - `input_weights` combined-gate input weight matrix (W) ``[1, d_input, d_hidden * n_gates]``
    /// - `recurrent_weights` combined-gate recurrent weight matrix (R) ``[1, d_hidden, d_hidden * n_gates]``
    /// - `biases` optional combined gate bias matrix (b) ``[1, 1, d_hidden * n_gates]``
    /// - `mode` [RnnMode] struct specifying the RNN cell variant
    /// - `size` [RnnSize] struct describing the dimensions of the problem
    ///
    /// # Returns:
    ///
    /// An [RnnOut](super::rnn::RnnOut) struct
    ///
    /// # Details
    ///
    /// - The weights of individual gates must be flattened as single input and recurrent weight tensors
    /// - The arguments are expected in 'unsqueezed' form to simplify internal operations.
    fn rnn(
        input: FloatTensor<B>,
        hidden_state: FloatTensor<B>,
        input_weights: FloatTensor<B>,
        recurrent_weights: FloatTensor<B>,
        biases: Option<FloatTensor<B>>,
        cell: RnnCell<B>,
        size: &RnnSize,
    ) -> RnnTrajectory<B> {
        let device = B::float_device(&input);
        // calculate trajectory input transformation with optional bias
        let wx = B::linear(input, input_weights, biases);
        // perform forward transition specific to the cell variant
        match cell {
            RnnCell::Lstm(cell_state) => lstm::lstm_forward(
                wx,
                recurrent_weights,
                hidden_state,
                cell_state,
                size,
                &device,
            ),
            _ => unimplemented!(),
        }
    }

    /// Backwards pass for the [rnn](RnnOps::rnn) operation,
    /// returning the gate activation gradients
    fn rnn_gates_backward(
        recurrent_weights: FloatTensor<B>,
        traj_grad: FloatTensor<B>,
        cache: Vec<FloatTensor<B>>,
        cell: RnnCell<B>,
        size: &RnnSize,
    ) -> FloatTensor<B> {
        let device = B::float_device(&recurrent_weights);
        match cell {
            RnnCell::Lstm(out_cell_state) => lstm::lstm_gates_backward::<B>(
                recurrent_weights,
                out_cell_state,
                traj_grad,
                cache,
                size,
                &device,
            ),
            RnnCell::Gru => unimplemented!(),
        }
    }

    /// Partial backwards pass for the [rnn](RnnOps::rnn) operation,
    /// returning the gradient for the input
    fn rnn_input_backward(
        input_weights: FloatTensor<B>,
        gates_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        let iw_t = B::float_transpose(input_weights);
        B::float_matmul(gates_grad, iw_t)
    }

    /// Partial backwards pass for the [rnn](RnnOps::rnn) operation,
    /// returning the gradient for the input weights
    fn rnn_input_weights_backward(
        input: FloatTensor<B>,
        gates_grad: FloatTensor<B>,
        size: &RnnSize,
    ) -> FloatTensor<B> {
        let x_t = B::float_transpose(B::float_reshape(
            input,
            [1, size.seq_d * size.bat_d, size.inp_d].into(),
        ));
        B::float_matmul(x_t, gates_grad)
    }

    /// Partial backwards pass for the [rnn](RnnOps::rnn) operation,
    /// returning the gradient for the recurrent weights
    fn rnn_recurrent_weights_backward(
        traj: FloatTensor<B>,
        gates_grad: FloatTensor<B>,
        size: &RnnSize,
    ) -> FloatTensor<B> {
        let h_t = B::float_transpose(B::float_reshape(
            traj,
            [1, size.seq_d * size.bat_d, size.hid_d].into(),
        ));
        B::float_matmul(h_t, gates_grad)
    }

    /// Partial backwards pass for the [rnn](RnnOps::rnn) operation,
    /// returning the gradient for the biases
    fn rnn_biases_backward(gates_grad: FloatTensor<B>) -> FloatTensor<B> {
        B::float_sum_dim(gates_grad, 1)
    }
}
