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

/// sigmoid first derivative helper
///
/// f'(x) = f(x)(1 - f(x))
#[cube]
fn _d_sigmoid<E: Float, N: Size>(sigmoid: Vector<E, N>, one: Vector<E, N>) -> Vector<E, N> {
    sigmoid * (one - sigmoid)
}

/// tanh first derivative helper
///
/// tanh'(x) = 1 - tanh^2(x)
#[cube]
fn _d_tanh<E: Float, N: Size>(tanh: Vector<E, N>, one: Vector<E, N>) -> Vector<E, N> {
    one - (tanh * tanh)
}

/// LSTM accelerator for forward element-wise calculations
///
/// Expects precomputed Wx + Rh + b (sum of matrix multiplications)
///
/// If tracked, gate results will be stored in cache for efficient backprop
#[cube(launch_unchecked, address_type = "dynamic")]
fn lstm_elemwise_kernel<E: Float, N: Size>(
    mut h_out: LinearViewMut<'_, Vector<E, N>>,
    mut c_out: LinearViewMut<'_, Vector<E, N>>,
    gates: ComptimeOption<&mut Tensor<Vector<E, N>>>,
    wx_rh: &Tensor<Vector<E, N>>,
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
    let ig = _sigmoid(wx_rh[ig_idx], one);
    let fg = _sigmoid(wx_rh[fg_idx], one);
    let cg = Vector::tanh(wx_rh[cg_idx]);
    let og = _sigmoid(wx_rh[og_idx], one);
    // if cache tensor provided store gate results
    #[comptime]
    match gates {
        ComptimeOption::Some(gates) => {
            gates[ig_idx] = ig;
            gates[fg_idx] = fg;
            gates[cg_idx] = cg;
            gates[og_idx] = og;
        }
        ComptimeOption::None => {}
    }
    // transition and store states
    let c_out_t = fg * c.read(ABSOLUTE_POS) + ig * cg;
    h_out.write(ABSOLUTE_POS, og * Vector::tanh(c_out_t));
    c_out.write(ABSOLUTE_POS, c_out_t);
}

/// Accelerated LSTM forward
pub fn lstm_elemwise<R: CubeRuntime>(
    wx_rh: CubeTensor<R>,
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
    let gate_shape = Shape::new([1, bat_d, hid_d * 4]);
    assert_eq!(state_shape, c.shape(), "incompatible shape of cell state");
    assert_eq!(gate_shape, wx_rh.shape(), "incompatible shape of input");
    // tensor args require contiguous
    let wx_rh = into_contiguous_aligned(wx_rh);
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
            address_type!(wx_rh, c),
            vector_size,
            // inplace outputs (reuse wx_rh for gates)
            h_out.clone().into_linear_view(),
            c_out.clone().into_linear_view(),
            tracked.then_some(wx_rh.clone().into_tensor_arg()).into(),
            // inputs
            wx_rh.clone().into_tensor_arg(),
            c.into_linear_view(),
            hid_d,
            dtype_to_storage_type(dtype),
        );
    }
    // return outputs
    let gates = tracked.then_some(wx_rh);
    ([h_out, c_out], gates)
}

/// LSTM accelerator for backward element-wise calculations
#[cube(launch_unchecked, address_type = "dynamic")]
fn lstm_backward_elemwise_kernel<E: Float, N: Size>(
    gates_grad: &mut Tensor<Vector<E, N>>,
    mut c_int_grad_out: LinearViewMut<'_, Vector<E, N>>,
    h_out_grad: LinearView<'_, Vector<E, N>>,
    h_int_grad: LinearView<'_, Vector<E, N>>,
    c: LinearView<'_, Vector<E, N>>,
    c_out: LinearView<'_, Vector<E, N>>,
    c_int_grad: LinearView<'_, Vector<E, N>>,
    gates: &Tensor<Vector<E, N>>,
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
    let c_out_tanh = Vector::tanh(c_out.read(ABSOLUTE_POS));
    let h_grad_total = h_int_grad.read(ABSOLUTE_POS) + h_out_grad.read(ABSOLUTE_POS);
    let (ig, fg, cg, og) = (gates[ig_idx], gates[fg_idx], gates[cg_idx], gates[og_idx]);
    let c_tanh_grad = og * h_grad_total;
    let one = Vector::one();
    let c_grad_total = c_int_grad.read(ABSOLUTE_POS) + c_tanh_grad * _d_tanh(c_out_tanh, one);
    c_int_grad_out.write(ABSOLUTE_POS, fg * c_grad_total);
    // calculate & write gate gradients
    gates_grad[ig_idx] = _d_sigmoid(ig, one) * cg * c_grad_total;
    gates_grad[fg_idx] = _d_sigmoid(fg, one) * c.read(ABSOLUTE_POS) * c_grad_total;
    gates_grad[cg_idx] = _d_tanh(cg, one) * ig * c_grad_total;
    gates_grad[og_idx] = _d_sigmoid(og, one) * h_grad_total * c_out_tanh;
}

/// Accelerated LSTM states backward
pub fn lstm_elemwise_backward<R: CubeRuntime>(
    h_out_grad: CubeTensor<R>,
    h_int_grad: CubeTensor<R>,
    c: CubeTensor<R>,
    c_out: CubeTensor<R>,
    c_int_grad: CubeTensor<R>,
    gates: CubeTensor<R>,
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
    assert_eq!(shape.clone(), h_int_grad.shape());
    assert_eq!(shape.clone(), c.shape());
    assert_eq!(shape.clone(), c_out.shape());
    assert_eq!(shape, c_int_grad.shape());
    assert_eq!(gates_shape, gates.shape());
    // tensor args require contiguous
    let gates = into_contiguous_aligned(gates);
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
            address_type!(h_out_grad, h_int_grad, c, c_out, c_int_grad, gates),
            vector_size,
            // outputs (reuse gates as gates_grad, write c_int_grad inplace)
            gates.clone().into_tensor_arg(),
            c_int_grad.clone().into_linear_view(),
            // inputs...
            h_out_grad.into_linear_view(),
            h_int_grad.into_linear_view(),
            c.into_linear_view(),
            c_out.into_linear_view(),
            c_int_grad.clone().into_linear_view(),
            gates.clone().into_tensor_arg(),
            hid_d,
            dtype_to_storage_type(dtype),
        );
    }
    let (gates_grad, c_int_grad_out) = (gates, c_int_grad);
    (gates_grad, c_int_grad_out)
}
