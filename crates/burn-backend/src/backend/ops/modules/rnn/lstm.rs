use crate::backend::Backend;
use crate::ops::rnn::RnnSize;
use crate::tensor::FloatTensor;

/// Result of [lstm_elemwise](LstmOps::lstm_elemwise)
#[derive(new)]
pub struct LstmElemwise<B: Backend> {
    /// hidden state
    pub h: FloatTensor<B>,
    /// cell state
    pub c: FloatTensor<B>,
    /// output of cell gates (required for backprop)
    pub gates: Option<FloatTensor<B>>,
}

/// Result of [lstm_elemwise_backward](LstmOps::lstm_elemwise_backward)
#[derive(new)]
pub struct LstmElemwiseBackward<B: Backend> {
    /// gradient of gate outputs
    pub gates_grad: FloatTensor<B>,
    /// intermediate cell state gradient
    pub c_int_grad: FloatTensor<B>,
}

/// LSTM cell operations trait
pub trait LstmOps<B: Backend> {
    /// LSTM-specific cell forward operations.
    /// Do not call directly, use [rnn](super::RnnOps::rnn)
    fn lstm_elemwise(
        mut wx_rh: FloatTensor<B>,
        mut c: FloatTensor<B>,
        size: &RnnSize,
        tracked: bool,
    ) -> LstmElemwise<B> {
        // FALLBACK IMPLEMENTATION
        // split into gate inputs
        let [ix, fx, cx, ox] =
            std::array::from_fn(|j| B::float_slice(wx_rh.clone(), &size.gate_range(0, j)));
        // calculate gate outputs
        let ig = B::sigmoid(ix);
        let fg = B::sigmoid(fx);
        let cg = B::tanh(cx);
        let og = B::sigmoid(ox);
        // transition states
        c = B::float_add(
            B::float_mul(fg.clone(), c),
            B::float_mul(ig.clone(), cg.clone()),
        );
        let h = B::float_mul(og.clone(), B::float_tanh(c.clone()));
        // store gate outputs required for accelerated backprop (reuse wx_rh)
        let gates = tracked.then_some({
            for (j, g) in [ig, fg, cg, og].into_iter().enumerate() {
                wx_rh = B::float_slice_assign(wx_rh, &size.gate_range(0, j), g);
            }
            wx_rh
        });
        LstmElemwise::new(h, c, gates)
    }

    /// LSTM-specific cell backward operations.
    /// Do not call directly, use [rnn_backward](super::RnnOps::rnn_backward)
    fn lstm_elemwise_backward(
        h_out_grad: FloatTensor<B>,
        h_int_grad: FloatTensor<B>,
        c: FloatTensor<B>,
        c_out: FloatTensor<B>,
        mut c_int_grad: FloatTensor<B>,
        gates: FloatTensor<B>,
        size: &RnnSize,
    ) -> LstmElemwiseBackward<B> {
        // FALLBACK IMPLEMENTATION
        // get gate outputs from cache
        let [ig, fg, cg, og] =
            std::array::from_fn(|j| B::float_slice(gates.clone(), &size.gate_range(0, j)));
        // calculate intermediate state gradients
        let h_grad_total = B::float_add(h_int_grad, h_out_grad);
        let c_out_tanh = B::float_tanh(c_out);
        let c_tanh_grad = B::float_mul(og.clone(), h_grad_total.clone());
        let c_grad_total = B::float_add(
            c_int_grad,
            B::tanh_backward(c_out_tanh.clone(), c_tanh_grad),
        );
        c_int_grad = B::float_mul(fg.clone(), c_grad_total.clone());
        // calculate gate grads
        let dog = B::sigmoid_backward(og, B::float_mul(h_grad_total, c_out_tanh));
        let dfg = B::sigmoid_backward(fg.clone(), B::float_mul(c.clone(), c_grad_total.clone()));
        let dig = B::sigmoid_backward(ig.clone(), B::float_mul(cg.clone(), c_grad_total.clone()));
        let dcg = B::tanh_backward(cg, B::float_mul(ig, c_grad_total));
        // write gate gradients (reuse gates tensor)
        let mut gates_grad = gates;
        for (j, dxg) in [dig, dfg, dcg, dog].into_iter().enumerate() {
            gates_grad = B::float_slice_assign(gates_grad, &size.gate_range(0, j), dxg);
        }
        LstmElemwiseBackward::new(gates_grad, c_int_grad)
    }
}
