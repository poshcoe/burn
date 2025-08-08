#[burn_tensor_testgen::testgen(module_lstm)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;
    use burn_tensor::ops::FloatElem;

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
    const OUT: [[[f32; HID_D]; BAT_D]; SEQ_D] = [
        [
            [0.0246, 0.0696],
            [0.0311, 0.0872],
            [-0.0393, -0.0062],
            [-0.1026, -0.0929],
        ],
        [
            [-0.0465, 0.0120],
            [0.0436, 0.1391],
            [-0.0161, 0.0557],
            [-0.01095, 0.0446],
        ],
    ];
    const CELL: [[[f32; HID_D]; BAT_D]; 1] = [[
        [-0.1090, 0.0169],
        [0.0968, 0.2190],
        [-0.0332, 0.0779],
        [-0.0234, 0.0676],
    ]];
    const HIDDEN: [[[f32; HID_D]; BAT_D]; 1] = [[
        [-0.0465, 0.0120],
        [0.0436, 0.1391],
        [-0.0161, 0.0557],
        [-0.01095, 0.0446],
    ]];

    #[test]
    fn test_lstm_forward() {
        let input = TestTensor::<3>::from(INPUT);
        let input_weights = TestTensor::<3>::from(INPUT_WEIGHTS_T).transpose();
        let recurrent_weights = TestTensor::<3>::from(RECURRENT_WEIGHTS_T).transpose();
        let biases = Some(TestTensor::<3>::from(BIASES));
        let cell = TestTensor::<3>::from([[[0.; HID_D]; BAT_D]; 1]);
        let hidden = TestTensor::<3>::from([[[0.; HID_D]; BAT_D]; 1]);
        let expected_output = TensorData::from(OUT);
        let expected_cell = TensorData::from(CELL);
        let expected_hidden = TensorData::from(HIDDEN);
        // test forward pass
        let (hidden_states, cell_states) = burn_tensor::module::lstm(
            input,
            hidden.clone(),
            cell.clone(),
            input_weights.clone(),
            recurrent_weights.clone(),
            biases.clone(),
            [SEQ_D, BAT_D, INP_D, HID_D],
        );
        let output = hidden_states
            .clone()
            .slice([1..SEQ_D + 1, 0..BAT_D, 0..HID_D]);
        let hidden = hidden_states.slice([SEQ_D..SEQ_D + 1, 0..BAT_D, 0..HID_D]);
        let cell = cell_states.slice([SEQ_D..SEQ_D + 1, 0..BAT_D, 0..HID_D]);
        // check forward results
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected_output, Default::default());
        hidden
            .to_data()
            .assert_approx_eq::<FT>(&expected_hidden, Default::default());
        cell.to_data()
            .assert_approx_eq::<FT>(&expected_cell, Default::default());
    }
}
