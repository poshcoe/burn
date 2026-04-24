use crate::Device;
use crate::backend::Backend;
use crate::ops::FloatTensor;
use crate::ops::rnn::{RnnCell, RnnSize, RnnTrajectory};

/// Default portable LSTM cell implementation
///
/// # Details
///
/// - The combined-gate weights must be flattened in `(input, forget, cell, output)` order.
pub(crate) fn lstm_forward<B: Backend>(
    wx: FloatTensor<B>,
    r: FloatTensor<B>,
    mut h: FloatTensor<B>,
    mut c: FloatTensor<B>,
    size: &RnnSize,
    device: &Device<B>,
) -> RnnTrajectory<B> {
    // init tensor for output trajectory of stacked hidden states
    let mut traj = B::float_empty(size.traj_shape(), device);
    // create cache if tracking gradients
    let mut cache = B::ad_enabled().then_some({
        let cell_inps = B::float_empty(size.traj_shape(), device);
        let gate_outs = B::float_empty([size.seq_d, size.bat_d, size.hid_d * 4].into(), device);
        vec![cell_inps, gate_outs]
    });
    // loop over recurrent operations
    for i in 0..size.seq_d {
        // get input transformation element
        let wx_i = B::float_slice(wx.clone(), &[i..i + 1, 0..size.bat_d, 0..size.hid_d * 4]);
        // calculate hidden transformation
        let rh_i = B::float_matmul(h, r.clone());
        [h, c] = lstm_cell_forward::<B>(wx_i, rh_i, c, &mut cache, i, size);
        // stack hidden states to create output trajectory
        traj = B::float_slice_assign(traj, &size.traj_elem_range(i), h.clone());
    }
    RnnTrajectory::new(traj, h, RnnCell::Lstm(c), cache)
}

fn lstm_cell_forward<B: Backend>(
    wx_i: FloatTensor<B>,
    rh_i: FloatTensor<B>,
    mut c: FloatTensor<B>,
    cache: &mut Option<Vec<FloatTensor<B>>>,
    seq_i: usize,
    size: &RnnSize,
) -> [FloatTensor<B>; 2] {
    // sum input and recurrent transformations
    let sum = B::float_add(wx_i, rh_i);
    // split into gate inputs
    let [ix, fx, cx, ox] =
        std::array::from_fn(|j| B::float_slice(sum.clone(), &size.gate_range(0, j)));
    // calculate gate outputs
    // TODO: perf opportunity with grouped sigmoid activation
    let ig = B::sigmoid(ix);
    let fg = B::sigmoid(fx);
    let cg = B::tanh(cx);
    let og = B::sigmoid(ox);
    // cache gate outputs and cell states for backprop (if tracking gradients)
    if let Some(v) = cache.take() {
        let [mut cell_inps, mut gate_outs] = v.try_into().unwrap();
        // cache input cell states
        cell_inps = B::float_slice_assign(cell_inps, &size.traj_elem_range(seq_i), c.clone());
        // cache gate outputs
        for (j, xg) in [&ig, &fg, &cg, &og].into_iter().cloned().enumerate() {
            gate_outs = B::float_slice_assign(gate_outs, &size.gate_range(seq_i, j), xg);
        }
        *cache = Some(vec![cell_inps, gate_outs])
    }
    // transition states
    c = B::float_add(B::float_mul(fg, c), B::float_mul(ig, cg));
    let h = B::float_mul(og, B::float_tanh(c.clone()));
    [h, c]
}

pub(crate) fn lstm_gates_backward<B: Backend>(
    r: FloatTensor<B>,
    mut c_out: FloatTensor<B>,
    out_grad: FloatTensor<B>,
    cache: Vec<FloatTensor<B>>,
    size: &RnnSize,
    device: &Device<B>,
) -> FloatTensor<B> {
    // init zero state grads (learnable states not supported)
    let mut h_grad = B::float_zeros(size.state_shape(), device);
    let mut c_grad = B::float_zeros(size.state_shape(), device);
    // get cached cell states and activations
    let [cell_inps, mut gate_outs] = cache.try_into().expect("Incompatible LSTM cache");
    // prepare transpose of recurrent weights
    let r_t = B::float_transpose(r);
    // perform in-sequence operations backward
    for i in (0..size.seq_d).rev() {
        // split cached gate activations
        let [ig, fg, cg, og] =
            std::array::from_fn(|j| B::float_slice(gate_outs.clone(), &size.gate_range(i, j)));
        let c_inp = B::float_slice(cell_inps.clone(), &size.traj_elem_range(i));
        let h_out_grad = B::float_slice(out_grad.clone(), &size.traj_elem_range(i));
        // calculate intermediate state gradients
        let h_grad_total = B::float_add(h_grad, h_out_grad);
        let c_out_tanh = B::float_tanh(c_out);
        let c_tanh_grad = B::float_matmul(og.clone(), h_grad_total.clone());
        let c_grad_total = B::float_add(c_grad, B::tanh_backward(c_out_tanh.clone(), c_tanh_grad));
        c_grad = B::float_mul(fg.clone(), c_grad_total.clone());
        // calculate gate grads
        let dog = B::sigmoid_backward(og, B::float_mul(h_grad_total, c_out_tanh));
        let dfg = B::sigmoid_backward(fg, B::float_mul(c_inp.clone(), c_grad_total.clone()));
        let dig = B::sigmoid_backward(ig.clone(), B::float_mul(cg.clone(), c_grad_total.clone()));
        let dcg = B::tanh_backward(cg, B::float_mul(ig, c_grad_total));
        // concat gates grads, reusing grad cache tensor
        for (j, dg) in [dig, dfg, dcg, dog].into_iter().enumerate() {
            gate_outs = B::float_slice_assign(gate_outs, &size.gate_range(i, j), dg);
        }
        // calculate intermediate hidden state gradient
        h_grad = B::float_matmul(
            B::float_slice(gate_outs.clone(), &size.traj_elem_range(i)),
            r_t.clone(),
        );
        c_out = c_inp;
    }
    // reshape gates grad for downstream grad calcs
    B::float_reshape(
        gate_outs,
        [1, size.seq_d * size.bat_d, size.hid_d * 4].into(),
    )
}
