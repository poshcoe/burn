use crate::backend::Backend;
use crate::tensor::FloatTensor;
use burn_std::{Shape, Slice, s};
use serde::{Deserialize, Serialize};

/// Long Short-Term Memory operations
pub mod lstm;
use lstm::{LstmElemwise, LstmElemwiseBackward, LstmOps};

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
    /// Shape of a single RNN state
    pub fn state_shape(&self) -> Shape {
        burn_std::shape!(1, self.bat_d, self.hid_d)
    }

    /// Gate range within a trajectory of flattened transitions
    pub fn gate_range(&self, j: usize) -> [Slice; 3] {
        s![0, .., self.hid_d * j..self.hid_d * (j + 1)]
    }
}

/// Enum of RNN cell operating modes
#[derive(new, Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub enum RnnMode {
    /// LSTM cell (stateful)
    Lstm,
    /// GRU cell
    Gru,
}

impl RnnMode {
    /// Get the gate count of the RNN mode
    pub const fn gate_count(&self) -> usize {
        match self {
            &RnnMode::Lstm => 4,
            &RnnMode::Gru => 3,
        }
    }
}

/// Options describing an RNN operation
#[derive(new, Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub struct RnnOptions {
    /// RNN cell mode
    pub mode: RnnMode,
    /// RNN problem size
    pub size: RnnSize,
    /// Whether to include gate results in RNN outputs (required for accelerated backprop)
    pub enable_gate_output: bool,
}

impl RnnOptions {
    /// Get new copy of the options with backprop cache enabled
    pub fn with_gate_output(&self) -> Self {
        RnnOptions {
            mode: self.mode.clone(),
            size: self.size.clone(),
            enable_gate_output: true,
        }
    }
}

/// Result of [rnn](RnnOps::rnn)
#[derive(new)]
pub struct Rnn<B: Backend> {
    /// output tensor of stacked hidden states `[d_sequence, d_batch, d_hidden]`
    pub out: FloatTensor<B>,
    /// final hidden state `[1, d_batch, d_hidden]`
    pub hidden_state: FloatTensor<B>,
    /// final cell state `[1, d_batch, d_hidden]` (required for some cell types, eg. LSTM)
    pub cell_state: Option<FloatTensor<B>>,
}

/// Result of [rnn_elemwise](RnnOps::rnn_elemwise)
#[derive(new)]
pub struct RnnElemwise<B: Backend> {
    /// output hidden state
    pub h_out: FloatTensor<B>,
    /// output cell state (required for some cell types)
    pub c_out: Option<FloatTensor<B>>,
    /// output of cell gates (required for backprop)
    pub g_out: Option<FloatTensor<B>>,
}

/// Result of [rnn_elemwise_backward](RnnOps::rnn_elemwise_backward)
#[derive(new)]
pub struct RnnElemwiseBackward<B: Backend> {
    /// gradient of gate inputs
    pub g_grad: FloatTensor<B>,
    /// cell state gradient
    pub c_grad: Option<FloatTensor<B>>,
}

/// Rnn operations trait
pub trait RnnOps<B: Backend>: LstmOps<B> {
    /// Run an RNN forward over the input sequence using the given weights, biases and state.
    ///
    /// # Arguments:
    ///
    /// - `input` input tensor (x) ``[d_sequence, d_batch, d_input]``
    /// - `hidden_state` hidden state (h) ``[1, d_batch, d_hidden]``
    /// - `cell_state` cell state required for RNN cells with separate internal state (eg. LSTM) (c) ``[1, d_batch, d_hidden]``
    /// - `input_weights` combined-gate input weight matrix (W) ``[1, d_input, d_hidden * n_gates]``
    /// - `recurrent_weights` combined-gate recurrent weight matrix (R) ``[1, d_hidden, d_hidden * n_gates]``
    /// - `biases` optional combined gate bias matrix (b) ``[1, 1, d_hidden * n_gates]``
    /// - `cache` cache of tensors required only for accelerated backprop/autodiff
    /// - `options` [RnnOptions] struct describing the RNN problem
    ///
    /// # Returns:
    ///
    /// An [Rnn] struct
    ///
    /// # Details
    ///
    /// - The weights of individual gates must be flattened as single input and recurrent weight tensors
    /// - The arguments are expected in 'unsqueezed' form to simplify internal operations.
    fn rnn(
        input: FloatTensor<B>,
        mut hidden_state: FloatTensor<B>,
        mut cell_state: Option<FloatTensor<B>>,
        input_weights: FloatTensor<B>,
        recurrent_weights: FloatTensor<B>,
        biases: Option<FloatTensor<B>>,
        options: &RnnOptions,
    ) -> Rnn<B> {
        let size = &options.size;
        let mut out_vec = Vec::with_capacity(size.seq_d);
        // calculate trajectory input transformation with optional bias
        let mut wx = B::float_matmul(input, input_weights);
        if let Some(b) = biases {
            wx = B::float_add(wx, b);
        }
        // loop over recurrent operations
        for i in 0..size.seq_d {
            // remove seq_d from wx_i (squeeze)
            let wx_i = B::float_slice(wx.clone(), &s![i, .., ..]);
            // calculate recurrent transformation
            let rh = B::float_matmul(hidden_state, recurrent_weights.clone());
            let g = B::float_add(wx_i, rh);
            // run elemwise cell forward
            RnnElemwise {
                h_out: hidden_state,
                c_out: cell_state,
                ..
            } = B::rnn_elemwise(g, cell_state, options);
            out_vec.push(hidden_state.clone());
        }
        // build output from stacked hidden states
        let out = B::float_cat(out_vec, 0);
        Rnn::new(out, hidden_state, cell_state)
    }

    /// Run the elementwise components of an RNN cell forward once.
    ///
    /// Not intened for external use. Use [rnn](Self::rnn)
    ///
    /// # Arguments
    ///
    /// - `g` cell gate inputs (G = W.X + R.H + b) `[1, d_batch, d_hidden * n_gates]`
    /// - `c` optional cell state `[1, d_batch, d_hidden]`
    /// - `options` [RnnOptions] struct describing the RNN problem
    ///
    /// # Returns
    ///
    /// An [RnnElemwise] struct
    fn rnn_elemwise(
        g: FloatTensor<B>,
        c: Option<FloatTensor<B>>,
        options: &RnnOptions,
    ) -> RnnElemwise<B> {
        let tracked = options.enable_gate_output;
        match &options.mode {
            RnnMode::Lstm => {
                let c = c.expect("Initial LSTM cell state must be provided.");
                let LstmElemwise {
                    h_out: h,
                    c_out: c,
                    g_out: gates,
                } = B::lstm_elemwise(g, c, &options.size, tracked);
                RnnElemwise::new(h, Some(c), gates)
            }
            RnnMode::Gru => unimplemented!(),
        }
    }

    /// Elementwise components of the backward pass for the [rnn_elemwise](RnnOps::rnn_elemwise) operation.
    ///
    /// Called during auto differentiation.
    ///
    /// # Arguments
    ///
    /// - `h_out_grad` gradient of output hidden state `[1, d_batch, d_hidden]`
    /// - `c` optional forward cell input state `[1, d_batch, d_hidden]`
    /// - `c_out` optional forward cell output state `[1, d_batch, d_hidden]`
    /// - `c_out_grad` optional gradient of cell output state `[1, d_batch, d_hidden]`
    /// - `g_out` forward cell gate outputs `[1, d_batch, d_hidden * n_gates]`
    /// - `options` [RnnOptions] struct describing the RNN problem
    ///
    /// # Returns
    ///
    /// An [RnnElemwiseBackward] struct
    ///
    /// # Details
    ///
    /// - Although separate [Option] parameters, `c` `c_out` and `c_out_grad` cannot be provided exclusively.
    fn rnn_elemwise_backward(
        h_out_grad: FloatTensor<B>,
        c: Option<FloatTensor<B>>,
        c_out: Option<FloatTensor<B>>,
        c_out_grad: Option<FloatTensor<B>>,
        g_out: FloatTensor<B>,
        options: &RnnOptions,
    ) -> RnnElemwiseBackward<B> {
        match options.mode {
            RnnMode::Lstm => {
                let LstmElemwiseBackward { g_grad, c_grad } = B::lstm_elemwise_backward(
                    h_out_grad,
                    c.unwrap(),
                    c_out.unwrap(),
                    c_out_grad.unwrap(),
                    g_out,
                    &options.size,
                );
                RnnElemwiseBackward::new(g_grad, Some(c_grad))
            }
            RnnMode::Gru => unimplemented!(),
        }
    }
}
