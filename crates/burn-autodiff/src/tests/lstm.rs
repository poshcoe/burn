#[burn_tensor_testgen::testgen(ad_lstm)]
mod tests {
    use super::*;
    use burn_tensor::ops::FloatElem;
    use burn_tensor::{TensorData, Tolerance, s};

    type FT = FloatElem<TestBackend>;

    pub const SEQ_D: usize = 2;
    const BAT_D: usize = 4;
    const INP_D: usize = 3;
    const HID_D: usize = 2;
    // known values sourced from PyTorch LSTM with zero-init state
    const INPUT: [[[f32; INP_D]; BAT_D]; SEQ_D] = [
        [
            [6.1599e-2, 1.4434e-1, 5.0515e-1],
            [4.5327e-1, 9.6461e-2, 9.4723e-1],
            [6.5902e-1, 8.3846e-1, 8.5444e-1],
            [7.3353e-1, 7.8060e-1, 2.7181e-1],
        ],
        [
            [4.5411e-1, 7.2665e-1, 4.0610e-1],
            [2.4880e-4, 2.0886e-1, 5.9652e-1],
            [4.7231e-1, 2.0873e-1, 8.2748e-1],
            [3.4535e-2, 8.7611e-2, 8.1984e-1],
        ],
    ];
    const INPUT_WEIGHTS_T: [[[f32; INP_D]; HID_D * 4]; 1] = [[
        [0.5515, -1.0005, -0.9981],
        [-0.3744, -0.6567, -0.8902],
        [-0.7804, 0.5981, -0.9637],
        [0.9862, -0.1910, 0.1298],
        [-0.5881, -0.6989, 0.5698],
        [-0.5157, -0.5139, 0.8493],
        [0.3309, -0.5226, -0.1362],
        [0.3765, 0.5327, 0.7626],
    ]];
    const RECURRENT_WEIGHTS_T: [[[f32; HID_D]; HID_D * 4]; 1] = [[
        [-0.4148, -0.1351],
        [0.2090, 0.4742],
        [-0.6913, 0.0456],
        [0.2024, 0.6486],
        [0.1362, -0.0956],
        [0.3370, -0.3085],
        [-0.2517, 0.0471],
        [-0.2644, 0.4771],
    ]];
    const BIASES: [[[f32; HID_D * 4]; 1]; 1] = [[[0., 0., 1., 1., 0., 0., 0., 0.]]];
    const INPUT_WEIGHTS_G_T: [[[f32; INP_D]; HID_D * 4]; 1] = [[
        [-7.1392e-3, -9.1484e-3, -8.3713e-5],
        [-1.8330e-3, -4.6515e-3, 1.3608e-2],
        [-2.5509e-4, 1.6488e-5, -1.5846e-3],
        [2.9855e-4, 6.5486e-4, -5.2864e-5],
        [2.6259e-2, 2.5132e-2, 5.2782e-2],
        [3.8327e-2, 4.1747e-2, 6.8519e-2],
        [-4.2946e-3, -5.0783e-3, -1.4337e-3],
        [1.2592e-4, 1.1887e-4, 5.2137e-3],
    ]];
    const RECURRENT_WEIGHTS_G_T: [[[f32; HID_D]; HID_D * 4]; 1] = [[
        [-2.7388e-4, -3.0253e-4],
        [-5.5798e-4, -3.4758e-4],
        [2.1925e-4, 2.1652e-4],
        [1.5562e-4, 2.2855e-4],
        [-7.7453e-4, 3.3847e-4],
        [-6.2271e-4, 9.5916e-4],
        [6.3325e-5, 5.1526e-5],
        [-3.4406e-5, 1.8998e-4],
    ]];
    const BIASES_G: [[[f32; HID_D * 4]; 1]; 1] = [[[
        -0.0054, 0.0125, -0.0017, 0.0006, 0.0790, 0.1081, -0.0045, 0.0066,
    ]]];

    #[test]
    fn test_lstm_backward() {
        let input = TestAutodiffTensor::<3>::from(INPUT).require_grad();
        let input_weights = TestAutodiffTensor::<3>::from(INPUT_WEIGHTS_T)
            .transpose()
            .require_grad();
        let recurrent_weights = TestAutodiffTensor::<3>::from(RECURRENT_WEIGHTS_T)
            .transpose()
            .require_grad();
        let biases = Some(TestAutodiffTensor::<3>::from(BIASES).require_grad());
        let hidden = TestAutodiffTensor::<3>::from([[[0.; HID_D]; BAT_D]; 1]);
        let cell = TestAutodiffTensor::<3>::from([[[0.; HID_D]; BAT_D]; 1]);
        let expected_input_weights_grad = TestTensor::<3>::from(INPUT_WEIGHTS_G_T)
            .transpose()
            .to_data();
        let expected_recurrent_weights_grad = TestTensor::<3>::from(RECURRENT_WEIGHTS_G_T)
            .transpose()
            .to_data();
        let expected_biases_grad = TensorData::from(BIASES_G);
        // run lstm under test
        let (hidden_states, cell_states) = burn_tensor::module::lstm(
            input,
            hidden.clone(),
            cell.clone(),
            input_weights.clone(),
            recurrent_weights.clone(),
            biases.clone(),
        );
        let out = hidden_states.slice(s![1.., .., ..]);
        // backprop with dummy loss
        let grads = out.mean().backward();
        let input_weights_grad = input_weights.grad(&grads).unwrap();
        let recurrent_weights_grad = recurrent_weights.grad(&grads).unwrap();
        let biases_grad = biases.as_ref().unwrap().grad(&grads).unwrap();
        // check backprop results
        input_weights_grad
            .to_data()
            .assert_approx_eq::<FT>(&expected_input_weights_grad, Default::default());
        recurrent_weights_grad
            .to_data()
            .assert_approx_eq::<FT>(&expected_recurrent_weights_grad, Default::default());
        biases_grad
            .to_data()
            .assert_approx_eq::<FT>(&expected_biases_grad, Tolerance::permissive());
    }
}
