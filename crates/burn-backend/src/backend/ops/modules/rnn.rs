use crate::tensor::FloatTensor;
use crate::{TensorMetadata, backend::Backend};
use burn_std::{FloatDType, Shape, Slice};
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
    /// Shape of an output trajectory of RNN states
    pub fn traj_shape(&self) -> Shape {
        burn_std::shape!(self.seq_d, self.bat_d, self.hid_d)
    }

    /// Shape of a single RNN state
    pub fn state_shape(&self) -> Shape {
        burn_std::shape!(1, self.bat_d, self.hid_d)
    }

    /// Gate range within a trajectory of flattened transitions
    pub const fn gate_range(&self, i: usize, j: usize) -> [Slice; 3] {
        [
            Slice::index(i as isize),
            Slice::full(),
            Slice::new(
                (self.hid_d * j) as isize,
                Some((self.hid_d * (j + 1)) as isize),
                1,
            ),
        ]
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
    /// Whether to cache tensors required for accelerated backprop
    pub enable_backprop_cache: bool,
}

impl RnnOptions {
    /// Get new copy of the options with backprop cache enabled
    pub fn with_backprop_cache(&self) -> Self {
        RnnOptions {
            mode: self.mode.clone(),
            size: self.size.clone(),
            enable_backprop_cache: true,
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
    /// tensor cache, required only for accelerated backprop
    pub cache: Option<Vec<FloatTensor<B>>>,
}

/// Result of [rnn_elemwise](RnnOps::rnn_elemwise)
#[derive(new)]
pub struct RnnElemwise<B: Backend> {
    /// output hidden state
    pub h: FloatTensor<B>,
    /// output cell state (required for some cell types)
    pub c: Option<FloatTensor<B>>,
    /// output of cell gates (required for backprop)
    pub gates: Option<FloatTensor<B>>,
}

/// Result of [rnn_elemwise_backward](RnnOps::rnn_elemwise_backward)
#[derive(new)]
pub struct RnnElemwiseBackward<B: Backend> {
    /// gradient of gate outputs
    pub gates_grad: FloatTensor<B>,
    /// intermediate cell state gradient (required for some cell types)
    pub c_int_grad: Option<FloatTensor<B>>,
}

/// Rnn operations trait
pub trait RnnOps<B: Backend>: LstmOps<B> {
    /// Run an RNN forward over the input sequence using the given weights, biases and state.
    ///
    /// # Arguments:
    ///
    /// - `input` input tensor (x) ``[d_sequence, d_batch, d_input]``
    /// - `hidden_state` hidden state (h) ``[1, d_batch, d_hidden]``
    /// - `cell_state` cell state required for RNN cells with seperate internal state (eg. LSTM) (c) ``[1, d_batch, d_hidden]``
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
        let mut cache = options
            .enable_backprop_cache
            .then_some(Vec::with_capacity(size.seq_d));
        // calculate trajectory input transformation with optional bias
        let mut wx = B::float_matmul(input, input_weights);
        if let Some(b) = biases {
            wx = B::float_add(wx, b);
        }
        // loop over recurrent operations
        for i in 0..size.seq_d {
            let wx_i = B::float_slice(
                wx.clone(),
                &[Slice::index(i as isize), Slice::full(), Slice::full()],
            );
            // calculate recurrent transformation
            let rh = B::float_matmul(hidden_state, recurrent_weights.clone());
            let wx_rh = B::float_add(wx_i, rh);
            // run elemwise cell forward
            let gates;
            RnnElemwise {
                h: hidden_state,
                c: cell_state,
                gates,
            } = B::rnn_elemwise(wx_rh, cell_state, options);
            out_vec.push(hidden_state.clone());
            // push cache elements
            if let Some(cache) = &mut cache {
                cache.push(gates.unwrap());
                cell_state.as_ref().map(|c| cache.push(c.clone()));
            }
        }
        // build output from stacked hidden states
        let out = B::float_cat(out_vec, 0);
        Rnn::new(out, hidden_state, cell_state, cache)
    }

    /// Run the elementwise components of an RNN cell forward once.
    /// Do not call directly. Use [rnn](Self::rnn)
    fn rnn_elemwise(
        wx_rh: FloatTensor<B>,
        c: Option<FloatTensor<B>>,
        options: &RnnOptions,
    ) -> RnnElemwise<B> {
        // run elemwise cell forward
        let tracked = options.enable_backprop_cache;
        match &options.mode {
            RnnMode::Lstm => {
                let c = c.expect("Initial LSTM cell state must be provided.");
                let LstmElemwise { h, c, gates } =
                    B::lstm_elemwise(wx_rh, c, &options.size, tracked);
                RnnElemwise::new(h, Some(c), gates)
            }
            RnnMode::Gru => unimplemented!(),
        }
    }

    /// Backwards pass for the [rnn](RnnOps::rnn) operation,
    /// returning the gate activation gradients
    fn rnn_backward(
        recurrent_weights: FloatTensor<B>,
        mut c_out: Option<FloatTensor<B>>,
        out_grad: FloatTensor<B>,
        mut cache: Vec<FloatTensor<B>>,
        options: &RnnOptions,
    ) -> FloatTensor<B> {
        let device = B::float_device(&recurrent_weights);
        let size = &options.size;
        let mut gates_grads_vec = Vec::with_capacity(size.seq_d);
        // prepare transpose of recurrent weights
        let r_t = B::float_transpose(recurrent_weights);
        // init intermediate grads (learnable states not supported)
        let dtype: FloatDType = out_grad.dtype().into();
        let mut h_int_grad = B::float_zeros(size.state_shape(), &device, dtype);
        let mut c_int_grad =
            c_out
                .is_some()
                .then_some(B::float_zeros(size.state_shape(), &device, dtype));
        // perform in-sequence operations backward
        for i in (0..size.seq_d).rev() {
            let h_out_grad = B::float_slice(
                out_grad.clone(),
                &[Slice::index(i as isize), Slice::full(), Slice::full()],
            );
            // pop cache elements
            let c = c_out.is_some().then_some(cache.pop().unwrap());
            let gates = cache.pop().unwrap();
            // run the elementwise components of the backward pass
            let gates_grad;
            RnnElemwiseBackward {
                gates_grad,
                c_int_grad,
            } = B::rnn_elemwise_backward(
                h_out_grad,
                h_int_grad,
                c.clone(),
                c_out,
                c_int_grad,
                gates,
                &options,
            );
            c_out = c;
            // calculate intermediate hidden state gradient
            h_int_grad = B::float_matmul(gates_grad.clone(), r_t.clone());
            gates_grads_vec.push(gates_grad);
        }
        // backprop cache should have been emptied during grad calculation
        assert!(cache.is_empty(), "backprop cache not emptied!");
        // build and reshape gates grad for downstream grad calcs
        let gates_grads = B::float_cat(gates_grads_vec, 0);
        B::float_reshape(
            gates_grads,
            [
                1,
                size.seq_d * size.bat_d,
                size.hid_d * options.mode.gate_count(),
            ]
            .into(),
        )
    }

    /// Elementwise recurrent components of the backward pass for the [rnn](RnnOps::rnn) operation.
    /// Do not call directly, use [rnn_backward](Self::rnn_backward)
    fn rnn_elemwise_backward(
        h_out_grad: FloatTensor<B>,
        h_int_grad: FloatTensor<B>,
        c: Option<FloatTensor<B>>,
        c_out: Option<FloatTensor<B>>,
        c_int_grad: Option<FloatTensor<B>>,
        gates: FloatTensor<B>,
        options: &RnnOptions,
    ) -> RnnElemwiseBackward<B> {
        match options.mode {
            RnnMode::Lstm => {
                let LstmElemwiseBackward {
                    gates_grad,
                    c_int_grad,
                } = B::lstm_elemwise_backward(
                    h_out_grad,
                    h_int_grad,
                    c.unwrap(),
                    c_out.unwrap(),
                    c_int_grad.unwrap(),
                    gates,
                    &options.size,
                );
                RnnElemwiseBackward::new(gates_grad, Some(c_int_grad))
            }
            RnnMode::Gru => unimplemented!(),
        }
    }

    /// Partial backwards pass for the [rnn](RnnOps::rnn) operation,
    /// returning the gradient for the input (X)
    fn rnn_input_backward(
        input_weights: FloatTensor<B>,
        gates_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        let iw_t = B::float_transpose(input_weights);
        B::float_matmul(gates_grad, iw_t)
    }

    /// Partial backwards pass for the [rnn](RnnOps::rnn) operation,
    /// returning the gradient for the input weights (W)
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
    /// returning the gradient for the recurrent weights (R)
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
