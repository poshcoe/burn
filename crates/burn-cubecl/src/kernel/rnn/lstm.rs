use crate::CubeRuntime;
use crate::kernel::into_contiguous_aligned;
use crate::kernel::utils::address_type;
use crate::tensor::CubeTensor;
use burn_backend::TensorMetadata;
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::ops::rnn::RnnSize;
use burn_std::Shape;
use cubecl::num_traits::One;
use cubecl::prelude::*;
use cubecl::std::tensor::layout::linear::{LinearView, LinearViewMut};

/// sigmoid (logistic function) helper
///
/// f(x) = 1 / (1 + e^(-x))
///
/// or equivalently:
///
/// f(x) = e^x / (e^x + 1)
#[cube]
fn _sigmoid<E: Float, N: Size>(x: Vector<E, N>, one: Vector<E, N>) -> Vector<E, N> {
    let ex = Vector::exp(x);
    ex / (ex + one)
}

/// LSTM accelerator for forward element-wise calculations
///
/// Expects precomputed Wx + Rh + b (sum of matrix multiplications)
///
/// If tracked, gate results will be stored in cache for efficient backprop
#[cube(launch_unchecked, address_type = "dynamic")]
fn lstm_elemwise_kernel<E: Float, N: Size>(
    mut ho: LinearViewMut<'_, Vector<E, N>>,
    mut co: LinearViewMut<'_, Vector<E, N>>,
    go: ComptimeOption<&mut Tensor<Vector<E, N>>>,
    g: &Tensor<Vector<E, N>>,
    c: LinearView<'_, Vector<E, N>>,
    #[comptime] hid_d: usize,
    #[define(E)] _dtype: StorageType,
) {
    // terminate extra units
    if !c.is_in_bounds(ABSOLUTE_POS) {
        terminate!()
    }
    // get indices of combined-gate inputs
    let vector_size = N::value().comptime();
    let gate_stride = comptime!(hid_d / vector_size);
    let ig_idx = ABSOLUTE_POS;
    let fg_idx = ig_idx + gate_stride;
    let cg_idx = fg_idx + gate_stride;
    let og_idx = cg_idx + gate_stride;
    // calculate gate activations
    let one = Vector::one();
    let ig = _sigmoid(g[ig_idx], one);
    let fg = _sigmoid(g[fg_idx], one);
    let cg = Vector::tanh(g[cg_idx]);
    let og = _sigmoid(g[og_idx], one);
    // if cache tensor provided store gate results
    #[comptime]
    match go {
        ComptimeOption::Some(go) => {
            go[ig_idx] = ig;
            go[fg_idx] = fg;
            go[cg_idx] = cg;
            go[og_idx] = og;
        }
        ComptimeOption::None => {}
    }
    // transition and store states
    let co_t = fg * c.read(ABSOLUTE_POS) + ig * cg;
    ho.write(ABSOLUTE_POS, og * Vector::tanh(co_t));
    co.write(ABSOLUTE_POS, co_t);
}

/// Accelerated LSTM forward
pub fn lstm_elemwise<R: CubeRuntime>(
    g: CubeTensor<R>,
    c: CubeTensor<R>,
    size: &RnnSize,
    tracked: bool,
) -> ([CubeTensor<R>; 2], Option<CubeTensor<R>>) {
    // check shape compat
    let RnnSize {
        seq_d: _,
        bat_d,
        inp_d: _,
        hid_d,
    } = size.clone();
    let state_shape = size.state_shape();
    let gates_shape = Shape::new([1, bat_d, hid_d * 4]);
    assert_eq!(state_shape, c.shape(), "incompatible shape of cell state");
    assert_eq!(gates_shape, g.shape(), "incompatible shape of gates input");
    // tensor args require contiguous
    let g = into_contiguous_aligned(g);
    // prepare output tensors (modified inplace by elemwise kernel)
    let client = &c.client.clone();
    let dtype = c.dtype();
    let new_empty = || {
        crate::ops::numeric::empty_device_dtype(client.clone(), c.device.clone(), c.shape(), dtype)
    };
    let h_out = new_empty();
    let c_out = new_empty();
    // prepare cube params for elemwise kernel
    let vector_size = crate::ops::max_vector_size(&c);
    let working_units = (bat_d * hid_d) / vector_size;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = cubecl::calculate_cube_count_elemwise(client, working_units, cube_dim);
    // run kernel calculating forward step
    unsafe {
        lstm_elemwise_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            address_type!(g, c),
            vector_size,
            // inplace outputs (reuse g for g_out)
            h_out.clone().into_linear_view(),
            c_out.clone().into_linear_view(),
            tracked.then_some(g.clone().into_tensor_arg()).into(),
            // inputs
            g.clone().into_tensor_arg(),
            c.into_linear_view(),
            hid_d,
            dtype_to_storage_type(dtype),
        );
    }
    // return outputs
    let g_out = tracked.then_some(g);
    ([h_out, c_out], g_out)
}

/// LSTM accelerator for backward element-wise calculations
#[cube(launch_unchecked, address_type = "dynamic")]
fn lstm_backward_elemwise_kernel<E: Float, N: Size>(
    g_grad: &mut Tensor<Vector<E, N>>,
    mut c_grad: LinearViewMut<'_, Vector<E, N>>,
    ho_grad: LinearView<'_, Vector<E, N>>,
    c: LinearView<'_, Vector<E, N>>,
    co: LinearView<'_, Vector<E, N>>,
    co_grad: LinearView<'_, Vector<E, N>>,
    go: &Tensor<Vector<E, N>>,
    #[comptime] hid_d: usize,
    #[define(E)] _dtype: StorageType,
) {
    // terminate extra units
    if !c.is_in_bounds(ABSOLUTE_POS) {
        terminate!()
    }
    // get indices of combined-gate inputs
    let vector_size = N::value().comptime();
    let gate_stride = comptime!(hid_d / vector_size);
    let ig_idx = ABSOLUTE_POS;
    let fg_idx = ig_idx + gate_stride;
    let cg_idx = fg_idx + gate_stride;
    let og_idx = cg_idx + gate_stride;
    // calculate cell state grad
    let one = Vector::one();
    let co_tanh = Vector::tanh(co.read(ABSOLUTE_POS));
    let (ig, fg, cg, og) = (go[ig_idx], go[fg_idx], go[cg_idx], go[og_idx]);
    let co_tanh_grad = og * ho_grad.read(ABSOLUTE_POS);
    // calculate temp values
    let co_grad_t = co_grad.read(ABSOLUTE_POS) + co_tanh_grad * (one - co_tanh * co_tanh);
    let fg_co_grad_t = fg * co_grad_t;
    let ig_co_grad_t = ig * co_grad_t;
    let cg_ig_co_grad_t = cg * ig_co_grad_t;
    // c_grad = (fg * co_grad_t)
    c_grad.write(ABSOLUTE_POS, fg_co_grad_t);
    // ig_grad = d_sigmoid(ig) * cg * co_grad_t
    //          -> (1 - ig) * (cg * ig * co_grad_t)
    g_grad[ig_idx] = (one - ig) * cg_ig_co_grad_t;
    // fg_grad = d_sigmoid(fg) * c * co_grad_t
    //          -> (1 - fg) * c * (fg * co_grad_t)
    g_grad[fg_idx] = (one - fg) * c.read(ABSOLUTE_POS) * fg_co_grad_t;
    // cg_grad = d_tanh(cg) * ig * co_grad_t
    //          -> (1 - cg * cg) * ig * co_grad_t
    //          -> (ig * co_grad_t) - cg * (cg * ig * co_grad_t)
    g_grad[cg_idx] = ig_co_grad_t - cg * cg_ig_co_grad_t;
    // og_grad = d_sigmoid(og) * co_tanh * ho_grad
    //          -> (1 - og) * co_tanh * (og * ho_grad)
    g_grad[og_idx] = (one - og) * co_tanh * co_tanh_grad;
}

/// Accelerated LSTM states backward
pub fn lstm_elemwise_backward<R: CubeRuntime>(
    h_out_grad: CubeTensor<R>,
    c: CubeTensor<R>,
    c_out: CubeTensor<R>,
    c_out_grad: CubeTensor<R>,
    g_out: CubeTensor<R>,
    size: &RnnSize,
) -> (CubeTensor<R>, CubeTensor<R>) {
    // check shape compat
    let RnnSize {
        seq_d: _,
        bat_d,
        inp_d: _,
        hid_d,
    } = size.clone();
    let shape = size.state_shape();
    let gates_shape = Shape::new([1, bat_d, hid_d * 4]);
    assert_eq!(shape.clone(), h_out_grad.shape());
    assert_eq!(shape.clone(), c.shape());
    assert_eq!(shape.clone(), c_out.shape());
    assert_eq!(shape, c_out_grad.shape());
    assert_eq!(gates_shape, g_out.shape());
    // tensor args require contiguous
    let g_out = into_contiguous_aligned(g_out);
    // prepare cube params for elemwise kernel
    let client = &h_out_grad.client.clone();
    let dtype = h_out_grad.dtype();
    let vector_size = crate::ops::max_vector_size(&h_out_grad);
    let working_units = (bat_d * hid_d) / vector_size;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = cubecl::calculate_cube_count_elemwise(client, working_units, cube_dim);
    // run kernel performing calculating backward step
    unsafe {
        lstm_backward_elemwise_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            address_type!(h_out_grad, c, c_out, c_out_grad, g_out),
            vector_size,
            // outputs (reuse g_out as g_grad, reuse c_out_grad as c_grad)
            g_out.clone().into_tensor_arg(),
            c_out_grad.clone().into_linear_view(),
            // inputs...
            h_out_grad.into_linear_view(),
            c.into_linear_view(),
            c_out.into_linear_view(),
            c_out_grad.clone().into_linear_view(),
            g_out.clone().into_tensor_arg(),
            hid_d,
            dtype_to_storage_type(dtype),
        );
    }
    let (g_grad, c_grad) = (g_out, c_out_grad);
    (g_grad, c_grad)
}
