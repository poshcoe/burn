use crate::{
    CubeBackend, CubeRuntime, FloatElement, IntElement,
    element::BoolElement,
    kernel::{
        self,
        conv::{ConvStrategy, ConvTranspose2dStrategy},
    },
};
use burn_tensor::ops::rnn::lstm::{LstmOut, LstmStateGrads};
use burn_tensor::ops::{
    ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions, InterpolateOptions,
    MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
};
use burn_tensor::ops::{FloatTensor, IntTensor};

impl<R, F, I, BT> ModuleOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv::<R, F, 1>(x, weight, bias, options, ConvStrategy::default()).unwrap()
    }

    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv::<R, F, 2>(x, weight, bias, options, ConvStrategy::default()).unwrap()
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::deform_conv2d::<R, F>(x, offset, weight, mask, bias, options).unwrap()
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        kernel::conv::deform_conv2d_backward::<R, F, I, BT>(
            x,
            offset,
            weight,
            mask,
            bias,
            output_grad,
            options,
        )
        .unwrap()
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv::<R, F, 3>(x, weight, bias, options, ConvStrategy::Direct).unwrap()
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_transpose2d::<R, F, I>(
            x,
            weight,
            bias,
            options,
            ConvTranspose2dStrategy::default(),
        )
        .unwrap()
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_transpose3d::<R, F>(x, weight, bias, options)
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::avg_pool2d::<R, F>(x, kernel_size, stride, padding, count_include_pad)
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::avg_pool2d_backward::<R, F>(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
        )
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<Self> {
        kernel::pool::max_pool2d::<R, F>(x, kernel_size, stride, padding, dilation)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<Self> {
        let (output, indices) = kernel::pool::max_pool2d_with_indices::<R, F, I>(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
        );

        MaxPool2dWithIndices::new(output, indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        MaxPool2dBackward::new(kernel::pool::max_pool2d_with_indices_backward::<R, F, I>(
            x,
            output_grad,
            indices,
            kernel_size,
            stride,
            padding,
            dilation,
        ))
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        kernel::pool::adaptive_avg_pool2d::<R, F>(x, output_size)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::pool::adaptive_avg_pool2d_backward::<R, F>(x, grad)
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        kernel::interpolate::interpolate::<R, F>(x, output_size, options)
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        kernel::interpolate::interpolate_backward::<R, F>(x, grad, output_size, options)
    }

    fn lstm(
        input: FloatTensor<Self>,
        hidden_state: FloatTensor<Self>,
        cell_state: FloatTensor<Self>,
        input_weights: FloatTensor<Self>,
        recurrent_weights: FloatTensor<Self>,
        biases: Option<FloatTensor<Self>>,
        tracked: bool,
    ) -> LstmOut<Self> {
        let ([hidden_states, cell_states], cache) = kernel::rnn::lstm::lstm::<R, F>(
            input,
            hidden_state,
            cell_state,
            input_weights,
            recurrent_weights,
            biases,
            tracked,
        )
        .unwrap();
        LstmOut {
            hidden_states,
            cell_states,
            cache,
        }
    }

    fn lstm_states_backward(
        recurrent_weights: FloatTensor<Self>,
        cell_states: FloatTensor<Self>,
        cache: FloatTensor<Self>,
        hidden_states_grad: FloatTensor<Self>,
    ) -> burn_tensor::ops::rnn::lstm::LstmStateGrads<Self> {
        let [hidden_state_grad, cell_state_grad, cache_grad] =
            kernel::rnn::lstm::lstm_states_backward::<R, F>(
                recurrent_weights,
                cell_states,
                cache,
                hidden_states_grad,
            )
            .unwrap();
        LstmStateGrads {
            hidden_state_grad,
            cell_state_grad,
            cache_grad,
        }
    }
}
