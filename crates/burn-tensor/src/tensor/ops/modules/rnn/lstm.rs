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
    /// stacked output hidden states `[d_sequence + 1, d_batch, d_hidden]`
    pub hidden_states: FloatTensor<B>,
    /// stacked output cell states `[d_sequence + 1, d_batch, d_hidden]`
    pub cell_states: FloatTensor<B>,
    /// cache of activations for accelerated backprop `[d_sequence, d_batch, d_hidden * 4]`
    pub cache: Option<FloatTensor<B>>,
}

pub(crate) fn lstm<B: Backend>(
    x: FloatTensor<B>,
    mut h: FloatTensor<B>,
    mut c: FloatTensor<B>,
    iw: FloatTensor<B>,
    rw: FloatTensor<B>,
    b: Option<FloatTensor<B>>,
    tracked: bool,
) -> LstmOut<B> {
    let device = B::float_device(&x);
    let [seq_d, bat_d, _inp_d] = x.shape().dims();
    let [_, _inp_d, hid_d_4] = iw.shape().dims();
    let hid_d = hid_d_4 / 4;
    let mut h_out = B::float_empty([seq_d + 1, bat_d, hid_d].into(), &device);
    let mut c_out = B::float_empty([seq_d + 1, bat_d, hid_d].into(), &device);
    // insert initial states
    let range = &[0..1, 0..bat_d, 0..hid_d];
    h_out = B::float_slice_assign(h_out, range, h.clone());
    c_out = B::float_slice_assign(c_out, range, c.clone());
    // calculate input transformation and add biases
    let iwx = B::float_matmul(x, iw);
    let mut iwx_b = match b {
        Some(b) => B::float_add(iwx, b),
        None => iwx,
    };
    // perform in-sequence operations
    for i in 0..seq_d {
        // calculate recurrent transformation
        let rwh = B::float_matmul(h, rw.clone());
        // sum input and recurrent transformations
        let linear = B::float_add(
            B::float_slice(iwx_b.clone(), &[i..i + 1, 0..bat_d, 0..hid_d_4]),
            rwh,
        );
        // prepare split of combined transformations
        let [linear_i, linear_f, linear_c, linear_o] = [0, 1, 2, 3].map(|j| {
            B::float_slice(
                linear.clone(),
                &[0..1, 0..bat_d, hid_d * j..hid_d * (j + 1)],
            )
        });
        // calculate gate activations
        let ig = B::sigmoid(linear_i);
        let fg = B::sigmoid(linear_f);
        let cg = B::float_tanh(linear_c);
        let og = B::sigmoid(linear_o);
        // cache activations (if tracked) for accelerated backprop
        if tracked {
            // reuse iwx_b as cache
            for (j, xg) in [&ig, &fg, &cg, &og].into_iter().cloned().enumerate() {
                iwx_b = B::float_slice_assign(
                    iwx_b,
                    &[i..i + 1, 0..bat_d, hid_d * j..hid_d * (j + 1)],
                    xg,
                );
            }
        }
        // transition states
        c = B::float_add(B::float_mul(fg, c), B::float_mul(ig, cg));
        h = B::float_mul(og, B::float_tanh(c.clone()));
        let range = &[i + 1..i + 2, 0..bat_d, 0..hid_d];
        h_out = B::float_slice_assign(h_out, range, h.clone());
        c_out = B::float_slice_assign(c_out, range, c.clone());
    }
    LstmOut {
        hidden_states: h_out,
        cell_states: c_out,
        cache: tracked.then_some(iwx_b),
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
    rw: FloatTensor<B>,
    c_out: FloatTensor<B>,
    mut cache: FloatTensor<B>,
    h_out_grad: FloatTensor<B>,
) -> LstmStateGrads<B> {
    let device = B::float_device(&cache);
    let [seq_d, bat_d, hid_d_4] = cache.shape().dims();
    let hid_d = hid_d_4 / 4;
    // init storage for hidden and cell gradients
    let mut h_grad = B::float_zeros([1, bat_d, hid_d].into(), &device);
    let mut c_grad = B::float_zeros([1, bat_d, hid_d].into(), &device);
    // prepare transpose of recurrent weights
    let rw_t = B::float_transpose(rw);
    // calculate elem-wise components
    let offset_ranges = &[1..seq_d + 1, 0..bat_d, 0..hid_d];
    let hidden_states_grad = B::float_slice(h_out_grad, offset_ranges);
    let c_out_tanh = B::float_tanh(B::float_slice(c_out.clone(), offset_ranges));
    let c_out_dtanh = d_tanh::<B>(c_out_tanh.clone());
    // perform in-sequence operations backward
    for i in (0..seq_d).rev() {
        let ranges = &[i..i + 1, 0..bat_d, 0..hid_d];
        let c_out = B::float_slice(c_out.clone(), ranges);
        let h_grad_total = B::float_add(h_grad, B::float_slice(hidden_states_grad.clone(), ranges));
        // fetch gate activations from cache
        let [ig, fg, cg, og] = [0, 1, 2, 3].map(|j| {
            B::float_slice(
                cache.clone(),
                &[i..i + 1, 0..bat_d, hid_d * j..hid_d * (j + 1)],
            )
        });
        // calculate cell state derivative
        let c_tanh_grad = B::float_mul(og.clone(), h_grad_total.clone());
        let c_grad_total = B::float_add(
            c_grad,
            B::float_mul(c_tanh_grad, B::float_slice(c_out_dtanh.clone(), ranges)),
        );
        c_grad = B::float_mul(fg.clone(), c_grad_total.clone());
        // calculate gate derivatives
        let dig = B::sigmoid_backward(ig.clone(), B::float_mul(cg.clone(), c_grad_total.clone()));
        let dfg = B::sigmoid_backward(fg, B::float_mul(c_out, c_grad_total.clone()));
        let dcg = B::float_mul(d_tanh::<B>(cg), B::float_mul(ig, c_grad_total));
        let dog = B::sigmoid_backward(
            og,
            B::float_mul(h_grad_total, B::float_slice(c_out_tanh.clone(), ranges)),
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
        h_grad = B::float_matmul(
            B::float_slice(cache.clone(), &[i..i + 1, 0..bat_d, 0..hid_d_4]),
            rw_t.clone(),
        );
    }
    // reshape cache for downstream grad calcs
    let cache_grad = B::float_reshape(cache, [1, seq_d * bat_d, hid_d_4].into());
    LstmStateGrads {
        hidden_state_grad: h_grad,
        cell_state_grad: c_grad,
        cache_grad,
    }
}

pub(crate) fn lstm_input_backward<B: Backend>(
    iw: FloatTensor<B>,
    cache_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let iw_t = B::float_transpose(iw);
    B::float_matmul(cache_grad, iw_t)
}

pub(crate) fn lstm_input_weights_backward<B: Backend>(
    x: FloatTensor<B>,
    cache_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let [seq_d, bat_d, inp_d] = x.shape().dims();
    let x_t = B::float_transpose(B::float_reshape(x, [1, seq_d * bat_d, inp_d].into()));
    B::float_matmul(x_t, cache_grad)
}

pub(crate) fn lstm_recurrent_weights_backward<B: Backend>(
    h_out: FloatTensor<B>,
    cache_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let [seq_d_p1, bat_d, hid_d] = h_out.shape().dims();
    let seq_d = seq_d_p1 - 1;
    let ranges = &[0..seq_d, 0..bat_d, 0..hid_d];
    let h_out_t = B::float_transpose(B::float_reshape(
        B::float_slice(h_out, ranges),
        [1, seq_d * bat_d, hid_d].into(),
    ));
    B::float_matmul(h_out_t, cache_grad)
}

pub(crate) fn lstm_biases_backward<B: Backend>(cache_grad: FloatTensor<B>) -> FloatTensor<B> {
    B::float_sum_dim(cache_grad, 1)
}
