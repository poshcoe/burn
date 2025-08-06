use crate as burn;
use crate::config::Config;
use crate::module::{Module, Param};
use crate::nn::Initializer;
use crate::tensor::backend::Backend;
use crate::tensor::{Tensor, s};

/// An LstmState is used to store cell state and hidden state in LSTM.
#[derive(Module, Debug)]
pub struct LstmState<B: Backend> {
    /// The hidden state `[1, d_batch, d_hidden]`
    pub hidden: Tensor<B, 3>,
    /// The cell state `[1, d_batch, d_hidden]`
    pub cell: Tensor<B, 3>,
}

/// Configuration to create a [LSTM](Lstm) module using the [init function](LstmConfig::init).
#[derive(Config)]
pub struct LstmConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the Lstm transformation.
    pub bias: bool,
    /// Initializer for input weights
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub input_weight_init: Initializer,
    /// Initializer for recurrent weights
    #[config(default = "Initializer::Orthogonal{gain:1.0}")]
    pub recurrent_weight_init: Initializer,
    /// Initializer for biases
    #[config(default = "Initializer::Zeros")]
    pub bias_init: Initializer,
    /// Forget gate bias override
    #[config(default = "Initializer::Ones")]
    pub forget_bias_init: Initializer,
}

impl LstmConfig {
    /// Initialize a new [LSTM](Lstm) module
    pub fn init<B: Backend>(&self, device: &B::Device) -> Lstm<B> {
        // init input weight params
        let input_weights = self.input_weight_init.init_with(
            [1, self.d_input, self.d_hidden * 4],
            Some(self.d_input),
            Some(self.d_hidden * 4),
            device,
        );
        // init recurrent weight params
        let recurrent_weights = self.recurrent_weight_init.init_with(
            [1, self.d_hidden, self.d_hidden * 4],
            Some(self.d_hidden),
            Some(self.d_hidden * 4),
            device,
        );
        // init bias params if configured
        let biases = self.bias.then_some(self.bias_init.init_with(
            [1, 1, self.d_hidden * 4],
            Some(self.d_input),
            Some(self.d_hidden * 4),
            device,
        ));
        // override forget gate initialization
        let biases = biases.map(|b_param| {
            let forget_bias = self
                .forget_bias_init
                .init_with(
                    [1, 1, self.d_hidden],
                    Some(self.d_input),
                    Some(self.d_hidden),
                    device,
                )
                .val();
            b_param.map(|b| {
                b.slice_assign([0..1, 0..1, self.d_hidden..self.d_hidden * 2], forget_bias)
            })
        });
        Lstm {
            input_weights,
            recurrent_weights,
            biases,
            d_hidden: self.d_hidden,
        }
    }

    /// Initialize a new [Bidirectional LSTM](BiLstm) module
    pub fn init_bilstm<B: Backend>(&self, device: &B::Device) -> BiLstm<B> {
        BiLstm {
            forward: self.init(device),
            reverse: self.init(device),
        }
    }
}

/// The Lstm module. This implementation is for a unidirectional, stateless, Lstm.
///
/// Introduced in the paper: [Long Short-Term Memory](https://www.researchgate.net/publication/13853244).
///
/// Should be created with [LstmConfig].
///
/// # Details
///
/// The combined-gate weights are flattened in `(input, forget, cell, output)` gate order.
#[derive(Module, Debug)]
pub struct Lstm<B: Backend> {
    /// Combined-gate input weights (W) ``[1, d_input, d_hidden * 4]``
    pub input_weights: Param<Tensor<B, 3>>,
    /// Combined-gate recurrent weights (R) ``[1, d_hidden, d_hidden * 4]``
    pub recurrent_weights: Param<Tensor<B, 3>>,
    /// Combined-gate biases (b) ``[1, 1, d_hidden * 4]``
    pub biases: Option<Param<Tensor<B, 3>>>,
    /// The hidden dimension of the LSTM
    pub d_hidden: usize,
}

impl<B: Backend> Lstm<B> {
    /// Applies the forward pass on the input tensor. This LSTM implementation
    /// returns the state for each element in a sequence (i.e., across d_sequence) and a final state.
    ///
    /// ## Parameters:
    /// - input: The input tensor of shape `[d_sequence, d_batch, d_input]`.
    /// - state: An optional `LstmState` representing the initial cell state and hidden state.
    ///   Each state tensor has shape `[1, d_batch, d_hidden]`.
    ///   If no initial state is provided, these tensors are initialized to zeros.
    ///
    /// ## Returns:
    /// - output: A tensor representing the output features of LSTM. Shape: `[d_sequence, d_batch, d_hidden]`
    /// - state: A `LstmState` represents the final states. Both `state.cell` and `state.hidden` have the shape
    ///   `[1, d_batch, d_hidden]`.
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: Option<LstmState<B>>,
    ) -> (Tensor<B, 3>, LstmState<B>) {
        // unwrap or initialize state
        let state = state.unwrap_or_else(|| {
            let d_batch = input.shape().dims[1];
            let device = input.device();
            LstmState {
                hidden: Tensor::zeros([1, d_batch, self.d_hidden], &device),
                cell: Tensor::zeros([1, d_batch, self.d_hidden], &device),
            }
        });
        // forward
        let (hidden_states, cell_states) = crate::tensor::module::lstm(
            input,
            state.hidden,
            state.cell,
            self.input_weights.val(),
            self.recurrent_weights.val(),
            self.biases.as_ref().map(|b| b.val()),
        );
        let out = hidden_states.clone().slice(s![1.., .., ..]);
        let hidden = hidden_states.slice(s![-1, .., ..]);
        let cell = cell_states.slice(s![-1, .., ..]);
        (out, LstmState { hidden, cell })
    }
}

/// An LstmState is used to store cell state and hidden state in LSTM.
#[derive(Module, Debug)]
pub struct BiLstmState<B: Backend> {
    /// The forward LSTM state
    pub forward: LstmState<B>,
    /// The reverse LSTM state
    pub reverse: LstmState<B>,
}

/// The BiLstm module. This implementation is for Bidirectional LSTM.
///
/// Introduced in the paper: [Framewise phoneme classification with bidirectional LSTM and other neural network architectures](https://www.cs.toronto.edu/~graves/ijcnn_2005.pdf).
///
/// Wraps [Lstm] modules
#[derive(Module, Debug)]
pub struct BiLstm<B: Backend> {
    /// LSTM for the forward direction.
    pub forward: Lstm<B>,
    /// LSTM for the reverse direction.
    pub reverse: Lstm<B>,
}

impl<B: Backend> BiLstm<B> {
    /// Applies the forward pass on the input tensor. This Bidirectional LSTM implementation
    /// returns the state for each element in a sequence (i.e., across d_sequence) and a final state.
    ///
    /// ## Parameters:
    /// - input: The input tensor of shape `[d_sequence, d_batch, d_input]`.
    /// - state: An optional `BiLstmState`, representing the initial cell state and hidden state of each LSTM.
    ///   Each state tensor has shape `[1, d_batch, d_hidden]`.
    ///   If no initial state is provided, these tensors are initialized to zeros.
    ///
    /// ## Returns:
    /// - output: A tensor represents the output features of LSTM. Shape: `[d_sequence, d_batch, d_hidden * 2]`
    /// - state: A `BiLstmState` represents the final forward and reverse hidden & cell states. All states
    ///   have the shape `[1, d_batch, d_hidden]`.
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: Option<BiLstmState<B>>,
    ) -> (Tensor<B, 3>, BiLstmState<B>) {
        // split states
        let (forward_state, reverse_state) = if let Some(state) = state {
            (Some(state.forward), Some(state.reverse))
        } else {
            (None, None)
        };
        // forward direction
        let (forward_output, forward_state) = self.forward.forward(input.clone(), forward_state);
        // reverse direction
        let (reverse_output, reverse_state) = self.reverse.forward(input.flip([0]), reverse_state);
        let reverse_output = reverse_output.flip([0]);
        // build combined output and state
        let output = Tensor::cat(vec![forward_output, reverse_output], 2);
        let state = BiLstmState {
            forward: forward_state,
            reverse: reverse_state,
        };
        (output, state)
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::cast::ToElement;

    use super::*;
    use crate::tensor::ops::FloatElem;
    use crate::tensor::{Distribution, TensorData, Tolerance};

    #[cfg(feature = "std")]
    type B = crate::TestAutodiffBackend;
    #[cfg(not(feature = "std"))]
    type B = crate::TestBackend;

    type E = FloatElem<B>;

    const SEQ_D: usize = 2;
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
    const HIDDEN: [[[f32; HID_D]; BAT_D]; 1] = [[
        [-0.0465, 0.0120],
        [0.0436, 0.1391],
        [-0.0161, 0.0557],
        [-0.01095, 0.0446],
    ]];
    const CELL: [[[f32; HID_D]; BAT_D]; 1] = [[
        [-0.1090, 0.0169],
        [0.0968, 0.2190],
        [-0.0332, 0.0779],
        [-0.0234, 0.0676],
    ]];
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
    fn test_lstm_against_known_values() {
        // create tensors from known values
        let input = Tensor::<B, 3>::from_data(TensorData::from(INPUT), &Default::default());
        let input_rev = input.clone().flip([0]);
        let input_weights =
            Tensor::<B, 3>::from_data(TensorData::from(INPUT_WEIGHTS_T), &Default::default())
                .transpose();
        let recurrent_weights =
            Tensor::<B, 3>::from_data(TensorData::from(RECURRENT_WEIGHTS_T), &Default::default())
                .transpose();
        let biases = Tensor::<B, 3>::from_data(TensorData::from(BIASES), &Default::default());
        let expected_output = TensorData::from(OUT);
        let expected_output_rev = TensorData::from([OUT[1], OUT[0]]);
        let expected_hidden = TensorData::from(HIDDEN);
        let expected_cell = TensorData::from(CELL);
        let expected_input_weights_grad =
            Tensor::<B, 3>::from_data(TensorData::from(INPUT_WEIGHTS_G_T), &Default::default())
                .transpose()
                .to_data();
        let expected_recurrent_weights_grad =
            Tensor::<B, 3>::from_data(TensorData::from(RECURRENT_WEIGHTS_G_T), &Default::default())
                .transpose()
                .to_data();
        let expected_biases_grad = TensorData::from(BIASES_G);
        // create lstm under test
        let lstm = Lstm {
            input_weights: Param::from_tensor(input_weights.clone()),
            recurrent_weights: Param::from_tensor(recurrent_weights.clone()),
            biases: Some(Param::from_tensor(biases.clone())),
            d_hidden: HID_D,
        };
        let (output, state) = lstm.forward(input, None);
        // create bilstm under test with identical forward & reverse weights
        let bilstm = BiLstm {
            forward: lstm.clone(),
            reverse: lstm.clone(),
        };
        let (output_bi, state_bi) = bilstm.forward(input_rev, None);
        let output_rev = output_bi
            .clone()
            .slice([0..SEQ_D, 0..BAT_D, HID_D..HID_D * 2]);
        let state_rev = state_bi.reverse;
        // check lstm and bilstm
        for (output, state, expected_output, lstm, msg) in [
            (output, state, expected_output, &lstm, "LSTM"),
            (
                output_rev,
                state_rev,
                expected_output_rev,
                &bilstm.reverse,
                "BILSTM",
            ),
        ] {
            println!("Checking {msg} outputs");
            // check outputs
            output
                .to_data()
                .assert_approx_eq::<E>(&expected_output, Default::default());
            state
                .hidden
                .to_data()
                .assert_approx_eq::<E>(&expected_hidden, Default::default());
            state
                .cell
                .to_data()
                .assert_approx_eq::<E>(&expected_cell, Default::default());
            // check backprop
            #[cfg(feature = "std")]
            {
                println!("Checking {msg} backprop");
                // backprop with dummy loss
                let grads = output.mean().backward();
                let input_weights_grad = lstm.input_weights.grad(&grads).unwrap();
                let recurrent_weights_grad = lstm.recurrent_weights.grad(&grads).unwrap();
                let biases_grad = lstm.biases.as_ref().unwrap().grad(&grads).unwrap();
                // check backprop results
                input_weights_grad
                    .to_data()
                    .assert_approx_eq::<E>(&expected_input_weights_grad, Default::default());
                recurrent_weights_grad
                    .to_data()
                    .assert_approx_eq::<E>(&expected_recurrent_weights_grad, Default::default());
                biases_grad
                    .to_data()
                    .assert_approx_eq::<E>(&expected_biases_grad, Tolerance::permissive());
            }
        }
    }

    #[test]
    fn test_lstm_init() {
        let [d_input, d_hidden, d_sequence, d_batch] = [2, 3, 4, 5];
        let bilstm = LstmConfig::new(d_input, d_hidden, true).init_bilstm::<B>(&Default::default());
        // check biases are properly initialized
        assert!(
            bilstm
                .forward
                .biases
                .as_ref()
                .unwrap()
                .val()
                .equal(Tensor::<B, 3>::from_floats(
                    [[[0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.]]],
                    &Default::default()
                ))
                .all()
                .into_scalar()
                .to_bool()
        );
        // prepare random input and run
        let input = Tensor::<B, 3>::random(
            [d_sequence, d_batch, d_input],
            Distribution::Normal(0., 1.),
            &Default::default(),
        );
        let (output, _) = bilstm.forward(input, None);
        assert_eq!(output.shape().dims(), [d_sequence, d_batch, d_hidden * 2]);
        // test backprop
        #[cfg(feature = "std")]
        {
            let grads = output.mean().backward();
            // check grads exist on both LSTM
            assert!(
                bilstm
                    .forward
                    .input_weights
                    .grad(&grads)
                    .unwrap()
                    .equal_elem(0.)
                    .any()
                    .into_scalar()
                    .to_bool()
                    != true
            );
            assert!(
                bilstm
                    .reverse
                    .input_weights
                    .grad(&grads)
                    .unwrap()
                    .equal_elem(0.)
                    .any()
                    .into_scalar()
                    .to_bool()
                    != true
            );
        }
    }
}
