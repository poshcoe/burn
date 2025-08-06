use crate::backend::Backend;
use crate::ops::FloatTensor;
use crate::{ElementConversion, TensorMetadata};

/// tanh'(x) = 1 - tanh^2(x)
///
/// TODO: add B::tanh_backward elsewhere
fn d_tanh<B: Backend>(tanh: FloatTensor<B>) -> FloatTensor<B> {
    B::float_add_scalar(
        B::float_neg(B::float_powi_scalar(tanh, 2.elem())),
        1.0.elem(),
    )
}

/// LSTM output
#[derive(new, Debug, Clone)]
pub struct LstmOut<B: Backend> {
    /// stacked hidden states `[d_sequence + 1, d_batch, d_hidden]`
    pub hidden_states: FloatTensor<B>,
    /// stacked cell states `[d_sequence + 1, d_batch, d_hidden]`
    pub cell_states: FloatTensor<B>,
    /// cache of activations for accelerated backprop `[d_sequence, d_batch, d_hidden * 4]`
    pub cache: Option<FloatTensor<B>>,
}

pub(crate) fn lstm<B: Backend>(
    input: FloatTensor<B>,
    mut hidden_state: FloatTensor<B>,
    mut cell_state: FloatTensor<B>,
    input_weights: FloatTensor<B>,
    recurrent_weights: FloatTensor<B>,
    biases: Option<FloatTensor<B>>,
    tracked: bool,
) -> LstmOut<B> {
    let device = B::float_device(&input);
    let [seq_d, bat_d, _inp_d] = input.shape().dims();
    let [_, _inp_d, hid_d_4] = input_weights.shape().dims();
    let hid_d = hid_d_4 / 4;
    let mut hidden_states = B::float_empty([seq_d + 1, bat_d, hid_d].into(), &device);
    let mut cell_states = B::float_empty([seq_d + 1, bat_d, hid_d].into(), &device);
    let mut cache = tracked.then_some(B::float_empty([seq_d, bat_d, hid_d_4].into(), &device));
    // insert initial states
    let range = &[0..1, 0..bat_d, 0..hid_d];
    hidden_states = B::float_slice_assign(hidden_states, range, hidden_state.clone());
    cell_states = B::float_slice_assign(cell_states, range, cell_state.clone());
    // calculate input transformation and add biases
    let wx = B::float_matmul(input, input_weights);
    let wx_b = match biases {
        Some(b) => B::float_add(wx, b),
        None => wx,
    };
    // perform in-sequence operations
    for i in 0..seq_d {
        // calculate recurrent transformation
        let rh = B::float_matmul(hidden_state, recurrent_weights.clone());
        // sum input and recurrent transformations
        let wx_b_rh = B::float_add(
            B::float_slice(wx_b.clone(), &[i..i + 1, 0..bat_d, 0..hid_d_4]),
            rh,
        );
        // prepare split of combined transformations
        let [wx_b_rh_i, wx_b_rh_f, wx_b_rh_c, wx_b_rh_o] = [0, 1, 2, 3].map(|j| {
            B::float_slice(
                wx_b_rh.clone(),
                &[0..1, 0..bat_d, hid_d * j..hid_d * (j + 1)],
            )
        });
        // calculate gate activations
        let ig = B::sigmoid(wx_b_rh_i);
        let fg = B::sigmoid(wx_b_rh_f);
        let cg = B::float_tanh(wx_b_rh_c);
        let og = B::sigmoid(wx_b_rh_o);
        // cache activations (if tracked) for accelerated backprop
        cache = cache.map(|mut v| {
            for (j, xg) in [&ig, &fg, &cg, &og].into_iter().cloned().enumerate() {
                v = B::float_slice_assign(v, &[i..i + 1, 0..bat_d, hid_d * j..hid_d * (j + 1)], xg);
            }
            v
        });
        // transition states
        cell_state = B::float_add(B::float_mul(fg, cell_state), B::float_mul(ig, cg));
        hidden_state = B::float_mul(og, B::float_tanh(cell_state.clone()));
        let range = &[i + 1..i + 2, 0..bat_d, 0..hid_d];
        hidden_states = B::float_slice_assign(hidden_states, range, hidden_state.clone());
        cell_states = B::float_slice_assign(cell_states, range, cell_state.clone());
    }
    LstmOut {
        hidden_states,
        cell_states,
        cache,
    }
}

/// LSTM state gradients
#[derive(new)]
pub struct LstmStateGrads<B: Backend> {
    /// Initial hidden state gradient `[1, d_batch, d_hidden]`
    pub hidden_state_grad: FloatTensor<B>,
    /// Initial cell state gradient `[1, d_batch, d_hidden]`
    pub cell_state_grad: FloatTensor<B>,
    /// temporary "cache" gradient `[seq_d * bat_d, d_hidden * 4]`
    pub cache_grad: FloatTensor<B>,
}

pub(crate) fn lstm_states_backward<B: Backend>(
    recurrent_weights: FloatTensor<B>,
    cell_states: FloatTensor<B>,
    mut cache: FloatTensor<B>,
    hidden_states_grad: FloatTensor<B>,
) -> LstmStateGrads<B> {
    let device = B::float_device(&cache);
    let [seq_d, bat_d, hid_d_4] = cache.shape().dims();
    let hid_d = hid_d_4 / 4;
    // init storage for hidden and cell gradients
    let mut dh = B::float_zeros([1, bat_d, hid_d].into(), &device);
    let mut dc = B::float_zeros([1, bat_d, hid_d].into(), &device);
    // prepare inputs
    let r_t = B::float_transpose(recurrent_weights);
    // calculate collected elem-wise components
    let state_ranges = &[1..seq_d + 1, 0..bat_d, 0..hid_d];
    let hidden_states_grad = B::float_slice(hidden_states_grad, state_ranges);
    let c_tanh = B::float_tanh(B::float_slice(cell_states.clone(), state_ranges));
    let c_dtanh = d_tanh::<B>(c_tanh.clone());
    // perform in-sequence operations
    for i in (0..seq_d).rev() {
        let ranges = &[i..i + 1, 0..bat_d, 0..hid_d];
        let c = B::float_slice(cell_states.clone(), ranges);
        let dh_total = B::float_add(dh, B::float_slice(hidden_states_grad.clone(), ranges));
        let mut dc_total = dc;
        // fetch gate activations from cache
        let [ig, fg, cg, og] = [0, 1, 2, 3].map(|j| {
            B::float_slice(
                cache.clone(),
                &[i..i + 1, 0..bat_d, hid_d * j..hid_d * (j + 1)],
            )
        });
        // calculate cell state derivative
        let dc_tanh = B::float_mul(og.clone(), dh_total.clone());
        dc_total = B::float_add(
            dc_total,
            B::float_mul(dc_tanh, B::float_slice(c_dtanh.clone(), ranges)),
        );
        dc = B::float_mul(fg.clone(), dc_total.clone());
        // calculate gate derivatives
        let dig = B::sigmoid_backward(ig.clone(), B::float_mul(cg.clone(), dc_total.clone()));
        let dfg = B::sigmoid_backward(fg, B::float_mul(c, dc_total.clone()));
        let dcg = B::float_mul(d_tanh::<B>(cg), B::float_mul(ig, dc_total));
        let dog = B::sigmoid_backward(
            og,
            B::float_mul(dh_total, B::float_slice(c_tanh.clone(), ranges)),
        );
        // update cache
        for (j, dxg) in [dig, dfg, dcg, dog].into_iter().enumerate() {
            cache = B::float_slice_assign(
                cache,
                &[i..i + 1, 0..bat_d, hid_d * j..hid_d * (j + 1)],
                dxg,
            );
        }
        // calculate hidden state derivative
        dh = B::float_matmul(
            B::float_slice(cache.clone(), &[i..i + 1, 0..bat_d, 0..hid_d_4]),
            r_t.clone(),
        );
    }
    let hidden_state_grad = dh;
    let cell_state_grad = dc;
    let cache_grad = B::float_reshape(cache, [1, seq_d * bat_d, hid_d_4].into());
    LstmStateGrads {
        hidden_state_grad,
        cell_state_grad,
        cache_grad,
    }
}

pub(crate) fn lstm_input_backward<B: Backend>(
    input_weights: FloatTensor<B>,
    cache_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let w_t = B::float_transpose(input_weights);
    B::float_matmul(cache_grad, w_t)
}

pub(crate) fn lstm_input_weights_backward<B: Backend>(
    input: FloatTensor<B>,
    cache_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let [seq_d, bat_d, inp_d] = input.shape().dims();
    let x_t = B::float_transpose(B::float_reshape(input, [1, seq_d * bat_d, inp_d].into()));
    B::float_matmul(x_t, cache_grad)
}

pub(crate) fn lstm_recurrent_weights_backward<B: Backend>(
    hidden_states: FloatTensor<B>,
    cache_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let [seq_d_p1, bat_d, hid_d] = hidden_states.shape().dims();
    let seq_d = seq_d_p1 - 1;
    let ranges = &[0..seq_d, 0..bat_d, 0..hid_d];
    let h_t = B::float_transpose(B::float_reshape(
        B::float_slice(hidden_states, ranges),
        [1, seq_d * bat_d, hid_d].into(),
    ));
    B::float_matmul(h_t, cache_grad)
}

pub(crate) fn lstm_biases_backward<B: Backend>(cache_grad: FloatTensor<B>) -> FloatTensor<B> {
    B::float_sum_dim(cache_grad, 1)
}
