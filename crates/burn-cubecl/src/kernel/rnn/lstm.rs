use crate::tensor::CubeTensor;
use crate::{CubeRuntime, FloatElement};
use cubecl::matmul::components::MatmulSetupError;
use cubecl::prelude::*;

/// sigmoid (logistic function) helper
///
/// f(x) = 1 / (1 + e^(-x))
///
/// or equivalently:
///
/// f(x) = e^x / (e^x + 1)
#[cube]
fn _sigmoid<E: Float>(x: Line<E>, one: Line<E>) -> Line<E> {
    let ex = Line::exp(x);
    ex / (ex + one)
}

/// sigmoid first derivative helper
///
/// f'(x) = f(x)(1 - f(x))
#[cube]
fn _d_sigmoid<E: Float>(sigmoid: Line<E>, one: Line<E>) -> Line<E> {
    sigmoid * (one - sigmoid)
}

/// tanh first derivative helper
///
/// tanh'(x) = 1 - tanh^2(x)
#[cube]
fn _d_tanh<E: Float>(tanh: Line<E>, one: Line<E>) -> Line<E> {
    one - (tanh * tanh)
}

/// LSTM accelerator for forward element-wise calculations
///
/// Expects precomputed Wx + b and rh matrix multiplications
///
/// If tracked, gate results will be stored in wx_b for efficient backprop
///
/// Resulting states are stored in-place on state arrays
#[cube(launch_unchecked)]
fn lstm_elemwise_kernel<E: Float>(
    h_out: &mut Array<Line<E>>,
    c_out: &mut Array<Line<E>>,
    iwx_b: &Array<Line<E>>,
    rwh: &Array<Line<E>>,
    cache: &mut Array<Line<E>>,
    seq_i: u32,
    #[comptime] bat_d: u32,
    #[comptime] hid_d: u32,
    #[comptime] tracked: bool,
) {
    let row = ABSOLUTE_POS_X;
    let col = ABSOLUTE_POS_Y;
    let line_size = h_out.line_size();
    let bat_stride = comptime!(hid_d / line_size);
    let seq_stride = comptime!(bat_d * bat_stride);
    // terminate extra units
    if row >= bat_stride || col >= bat_d {
        terminate!()
    }
    // get indices of combined-gate inputs
    let g_base_idx = col * comptime!(4 * bat_stride) + row;
    let ig_idx = g_base_idx;
    let fg_idx = ig_idx + bat_stride;
    let cg_idx = fg_idx + bat_stride;
    let og_idx = cg_idx + bat_stride;
    // get offset indices of input transformation
    let g_offset_idx = seq_i * comptime!(4 * seq_stride);
    let ig_offset_idx = ig_idx + g_offset_idx;
    let fg_offset_idx = fg_idx + g_offset_idx;
    let cg_offset_idx = cg_idx + g_offset_idx;
    let og_offset_idx = og_idx + g_offset_idx;
    // calculate gate activations
    let one = Line::empty(line_size).fill(E::new(1.));
    let ig = _sigmoid(iwx_b[ig_offset_idx] + rwh[ig_idx], one);
    let fg = _sigmoid(iwx_b[fg_offset_idx] + rwh[fg_idx], one);
    let cg = Line::tanh(iwx_b[cg_offset_idx] + rwh[cg_idx]);
    let og = _sigmoid(iwx_b[og_offset_idx] + rwh[og_idx], one);
    // if tracked store gate results in cache
    if tracked {
        cache[ig_offset_idx] = ig;
        cache[fg_offset_idx] = fg;
        cache[cg_offset_idx] = cg;
        cache[og_offset_idx] = og;
    }
    // get index of states
    let s_idx = col * bat_stride + row;
    let s_inp_idx = s_idx + (seq_i * seq_stride);
    let s_out_idx = s_inp_idx + seq_stride;
    // transition and store states
    c_out[s_out_idx] = fg * c_out[s_inp_idx] + ig * cg;
    h_out[s_out_idx] = og * Line::tanh(c_out[s_out_idx]);
}

/// Type containing LSTM output hidden states, cell states and optional cache for backprop
pub type CubeLstmOut<R> = ([CubeTensor<R>; 2], Option<CubeTensor<R>>);

/// Accelerated LSTM forward
pub fn lstm<R: CubeRuntime, E: FloatElement>(
    x: CubeTensor<R>,
    h: CubeTensor<R>,
    c: CubeTensor<R>,
    iw: CubeTensor<R>,
    rw: CubeTensor<R>,
    b: Option<CubeTensor<R>>,
    size: [usize; 4],
    tracked: bool,
) -> Result<CubeLstmOut<R>, MatmulSetupError> {
    let [seq_d, bat_d, _, hid_d] = size;
    // prepare output tensors (modified inplace by elemwise kernel)
    let h_out = crate::ops::base::expand(h, [seq_d + 1, bat_d, hid_d].into()).copy();
    let c_out = crate::ops::base::expand(c, [seq_d + 1, bat_d, hid_d].into()).copy();
    // calculate input transformation & add bias
    let iwx = crate::kernel::matmul::matmul::<R, E>(x, iw, None, Default::default())?;
    let iwx_b = match b {
        Some(b) => crate::ops::numeric::add::<R, E>(iwx, b),
        None => iwx,
    };
    // prepare cube params for elemwise kernel
    let cube_dim = Default::default();
    let line_size =
        cubecl::tensor_line_size(R::supported_line_sizes(), &[bat_d, hid_d], &[hid_d, 1], 1);
    let cube_count =
        cubecl::calculate_cube_count_elemwise((bat_d * hid_d) / line_size as usize, cube_dim);
    // perform in-sequence operations
    for i in 0..seq_d {
        // calculate recurrent transformation
        let h = crate::kernel::slice::<R, E>(h_out.clone(), &[i..i + 1, 0..bat_d, 0..hid_d]);
        let rwh = crate::kernel::matmul::matmul::<R, E>(h, rw.clone(), None, Default::default())?;
        // run kernel performing gate activations and inplace state transitions
        unsafe {
            lstm_elemwise_kernel::launch_unchecked::<E, R>(
                &iwx_b.client,
                cube_count.clone(),
                cube_dim,
                h_out.as_array_arg::<E>(line_size),
                c_out.as_array_arg::<E>(line_size),
                iwx_b.as_array_arg::<E>(line_size),
                rwh.as_array_arg::<E>(line_size),
                // reuse wx_b as cache
                iwx_b.as_array_arg::<E>(line_size),
                ScalarArg::new(i as u32),
                bat_d as u32,
                hid_d as u32,
                tracked,
            );
        }
    }
    Ok(([h_out, c_out], tracked.then_some(iwx_b)))
}

/// LSTM accelerator for backward element-wise calculations
#[cube(launch_unchecked)]
fn lstm_states_backward_elemwise<E: Float>(
    c_out: &Array<Line<E>>,
    cache: &mut Array<Line<E>>,
    h_grad: &Array<Line<E>>,
    c_grad: &mut Array<Line<E>>,
    h_out_grad: &Array<Line<E>>,
    seq_i: u32,
    #[comptime] bat_d: u32,
    #[comptime] hid_d: u32,
) {
    let row = ABSOLUTE_POS_X;
    let col = ABSOLUTE_POS_Y;
    let line_size = c_out.line_size();
    let bat_stride = comptime!(hid_d / line_size);
    let seq_stride = comptime!(bat_d * bat_stride);
    let one = Line::empty(line_size).fill(E::new(1.));
    // terminate extra units
    if row >= bat_stride || col >= bat_d {
        terminate!()
    }
    // get indices of cached activations
    let g_base_idx = col * comptime!(4 * bat_stride) + row;
    let g_offset_idx = g_base_idx + seq_i * comptime!(4 * seq_stride);
    let ig_offset_idx = g_offset_idx;
    let fg_offset_idx = ig_offset_idx + bat_stride;
    let cg_offset_idx = fg_offset_idx + bat_stride;
    let og_offset_idx = cg_offset_idx + bat_stride;
    // load cached activations
    let ig = cache[ig_offset_idx];
    let fg = cache[fg_offset_idx];
    let cg = cache[cg_offset_idx];
    let og = cache[og_offset_idx];
    // get indices of states
    let s_idx = col * bat_stride + row;
    let s_inp_idx = s_idx + seq_i * seq_stride;
    let s_out_idx = s_inp_idx + seq_stride;
    // calculate cell state grad
    let c_out_tanh = Line::tanh(c_out[s_out_idx]);
    let h_grad_total = h_grad[s_idx] + h_out_grad[s_out_idx];
    let c_tanh_grad = og * h_grad_total;
    let c_grad_total = c_grad[s_idx] + c_tanh_grad * _d_tanh(c_out_tanh, one);
    c_grad[s_idx] = fg * c_grad_total;
    // calculate & update cache with gate grads
    cache[ig_offset_idx] = _d_sigmoid(ig, one) * cg * c_grad_total;
    cache[fg_offset_idx] = _d_sigmoid(fg, one) * c_out[s_inp_idx] * c_grad_total;
    cache[cg_offset_idx] = _d_tanh(cg, one) * ig * c_grad_total;
    cache[og_offset_idx] = _d_sigmoid(og, one) * h_grad_total * c_out_tanh;
}

/// Accelerated LSTM states backward
pub fn lstm_states_backward<R: CubeRuntime, E: FloatElement>(
    rw: CubeTensor<R>,
    c_out: CubeTensor<R>,
    cache: CubeTensor<R>,
    h_out_grad: CubeTensor<R>,
    size: [usize; 4],
) -> Result<[CubeTensor<R>; 3], MatmulSetupError> {
    let device = &rw.device;
    let [seq_d, bat_d, _, hid_d] = size;
    // init storage for hidden and cell gradients
    let mut h_grad = crate::ops::numeric::zeros::<R, E>([1, bat_d, hid_d].into(), device);
    let c_grad = crate::ops::numeric::zeros::<R, E>([1, bat_d, hid_d].into(), device);
    // prepare transpose of recurrent weights
    let rw_t = crate::ops::swap_dims(rw, 1, 2);
    // prepare cube params for elemwise kernel
    let cube_dim = Default::default();
    let line_size =
        cubecl::tensor_line_size(R::supported_line_sizes(), &[bat_d, hid_d], &[hid_d, 1], 1);
    let cube_count =
        cubecl::calculate_cube_count_elemwise((bat_d * hid_d) / line_size as usize, cube_dim);
    // perform in-sequence operations backward
    for i in (0..seq_d).rev() {
        unsafe {
            lstm_states_backward_elemwise::launch_unchecked::<E, R>(
                &cache.client,
                cube_count.clone(),
                cube_dim,
                c_out.as_array_arg::<E>(line_size),
                cache.as_array_arg::<E>(line_size),
                h_grad.as_array_arg::<E>(line_size),
                c_grad.as_array_arg::<E>(line_size),
                h_out_grad.as_array_arg::<E>(line_size),
                ScalarArg::new(i as u32),
                bat_d as u32,
                hid_d as u32,
            );
        }
        // calculate hidden state grad
        let s_cache =
            crate::kernel::slice::<R, E>(cache.clone(), &[i..i + 1, 0..bat_d, 0..hid_d * 4]);
        h_grad =
            crate::kernel::matmul::matmul::<R, E>(s_cache, rw_t.clone(), None, Default::default())?;
    }
    // reshape cache for downstream grad cacls
    let cache_grad = crate::ops::reshape(cache, [1, seq_d * bat_d, hid_d * 4].into());
    Ok([h_grad, c_grad, cache_grad])
}
