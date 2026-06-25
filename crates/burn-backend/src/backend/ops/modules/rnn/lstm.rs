use crate::backend::Backend;
use crate::ops::rnn::RnnSize;
use crate::tensor::FloatTensor;

/// Result of [lstm_elemwise](LstmOps::lstm_elemwise)
#[derive(new)]
pub struct LstmElemwise<B: Backend> {
    /// output hidden state
    pub h_out: FloatTensor<B>,
    /// output cell state
    pub c_out: FloatTensor<B>,
    /// output of cell gates (required for backprop)
    pub g_out: Option<FloatTensor<B>>,
}

/// Result of [lstm_elemwise_backward](LstmOps::lstm_elemwise_backward)
#[derive(new)]
pub struct LstmElemwiseBackward<B: Backend> {
    /// gradient of gate outputs
    pub g_grad: FloatTensor<B>,
    /// cell state gradient
    pub c_grad: FloatTensor<B>,
}

/// LSTM cell operations trait
pub trait LstmOps<B: Backend> {
    /// LSTM-specific cell forward operations.
    /// Do not call directly, use [rnn](super::RnnOps::rnn)
    fn lstm_elemwise(
        g: FloatTensor<B>,
        c: FloatTensor<B>,
        size: &RnnSize,
        tracked: bool,
    ) -> LstmElemwise<B> {
        // FALLBACK IMPLEMENTATION
        // split into gate inputs
        let [ix, fx, cx, ox] =
            std::array::from_fn(|j| B::float_slice(g.clone(), &size.gate_range(j)));
        // calculate gate outputs
        let ig = B::sigmoid(ix);
        let fg = B::sigmoid(fx);
        let cg = B::tanh(cx);
        let og = B::sigmoid(ox);
        // transition states
        let c_out = B::float_add(
            B::float_mul(fg.clone(), c),
            B::float_mul(ig.clone(), cg.clone()),
        );
        let h_out = B::float_mul(og.clone(), B::float_tanh(c_out.clone()));
        // store gate outputs required for accelerated backprop (reuse g)
        let mut g_out = g;
        let g_out = tracked.then_some({
            for (j, sg) in [ig, fg, cg, og].into_iter().enumerate() {
                g_out = B::float_slice_assign(g_out, &size.gate_range(j), sg);
            }
            g_out
        });
        LstmElemwise::new(h_out, c_out, g_out)
    }

    /// LSTM-specific cell backward operations.
    /// Do not call directly, use [rnn_backward](super::RnnOps::rnn_backward)
    fn lstm_elemwise_backward(
        h_out_grad: FloatTensor<B>,
        c: FloatTensor<B>,
        c_out: FloatTensor<B>,
        c_out_grad: FloatTensor<B>,
        g_out: FloatTensor<B>,
        size: &RnnSize,
    ) -> LstmElemwiseBackward<B> {
        // FALLBACK IMPLEMENTATION
        // get gate outputs from cache
        let [ig, fg, cg, og] =
            std::array::from_fn(|j| B::float_slice(g_out.clone(), &size.gate_range(j)));
        // calculate intermediate state gradients
        let c_out_tanh = B::float_tanh(c_out);
        let c_tanh_grad = B::float_mul(og.clone(), h_out_grad.clone());
        let c_grad_total = B::float_add(
            c_out_grad,
            B::tanh_backward(c_out_tanh.clone(), c_tanh_grad),
        );
        let c_grad = B::float_mul(fg.clone(), c_grad_total.clone());
        // calculate gate grads
        let dog = B::sigmoid_backward(og, B::float_mul(h_out_grad, c_out_tanh));
        let dfg = B::sigmoid_backward(fg.clone(), B::float_mul(c.clone(), c_grad_total.clone()));
        let dig = B::sigmoid_backward(ig.clone(), B::float_mul(cg.clone(), c_grad_total.clone()));
        let dcg = B::tanh_backward(cg, B::float_mul(ig, c_grad_total));
        // write gate gradients (reuse gate outputs)
        let mut g_grad = g_out;
        for (j, dsg) in [dig, dfg, dcg, dog].into_iter().enumerate() {
            g_grad = B::float_slice_assign(g_grad, &size.gate_range(j), dsg);
        }
        LstmElemwiseBackward::new(g_grad, c_grad)
    }
}
