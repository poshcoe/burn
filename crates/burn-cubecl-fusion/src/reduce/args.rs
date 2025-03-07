use cubecl::reduce::args::ReduceArgs;
use cubecl::{prelude::*, reduce::args::ReduceDType};

use crate::shared::io::{ref_buffer_len, ref_len, ref_rank, ref_shape, ref_stride};
use crate::shared::ir::{
    Arg, ElemwiseConfig, GlobalArgs, GlobalArgsExpand, LocalArgs, LocalArgsExpand,
};
use crate::shared::kernel::{fuse_on_read, fuse_on_write, init_locals};

#[derive(Clone)]
pub struct FusedReduceArgs;

#[derive(CubeLaunch)]
pub struct FusedReduceInput {
    global: GlobalArgs,
    #[cube(comptime)]
    config: ElemwiseConfig,
    #[cube(comptime)]
    arg: Arg,
}

#[derive(CubeLaunch)]
pub struct FusedReduceOutput {
    global: GlobalArgs,
    #[cube(comptime)]
    config: ElemwiseConfig,
    #[cube(comptime)]
    arg: Arg,
}

pub struct FusedReduceState {
    inputs: *const GlobalArgs,
    outputs: *mut GlobalArgs,
    locals_on_read: *mut LocalArgs,
    locals_on_write: *mut LocalArgs,
    config_on_read: ElemwiseConfig,
    config_on_write: ElemwiseConfig,
    input: Arg,
    out: Arg,
}

#[derive(Clone)]
pub struct FusedReduceStateExpand {
    inputs: GlobalArgsExpand,
    outputs: GlobalArgsExpand,
    locals_on_read: LocalArgsExpand,
    locals_on_write: LocalArgsExpand,
    config_on_read: ElemwiseConfig,
    config_on_write: ElemwiseConfig,
    input: Arg,
    out: Arg,
}

#[cube]
impl ReduceArgs for FusedReduceArgs {
    type Input<E: Numeric> = FusedReduceInput;
    type Output<E: Numeric> = FusedReduceOutput;
    type State<P: ReduceDType> = FusedReduceState;

    fn init_state<P: ReduceDType>(
        input: &Self::Input<P::In>,
        output: &mut Self::Output<P::Out>,
    ) -> Self::State<P> {
        let mut locals_read = init_locals(&input.global, &mut output.global, &input.config);
        let mut locals_write = init_locals(&input.global, &mut output.global, &output.config);
        FusedReduceState::new(input, output, &mut locals_read, &mut locals_write)
    }

    fn read_input<P: ReduceDType>(state: &Self::State<P>, index: u32) -> Line<P::In> {
        *fuse_on_read::<P::In>(
            unsafe { &(*state.inputs) },
            unsafe { &mut (*state.outputs) },
            unsafe { &mut (*state.locals_on_read) },
            index,
            comptime! {
                let mut sequence = Sequence::new();
                sequence.push(state.input.clone());
                sequence
            },
            &state.config_on_read,
        )
        .index(0)
    }

    fn read_output<P: ReduceDType>(_state: &Self::State<P>, _index: u32) -> Line<P::Out> {
        Line::empty(1)
    }

    fn write_output<P: ReduceDType>(state: &mut Self::State<P>, index: u32, value: Line<P::Out>) {
        let mut values = Registry::<Arg, Line<P::Out>>::new();
        let mut args = comptime![Sequence::<Arg>::new()];

        values.insert(comptime![state.out.clone()], value);
        comptime![args.push(state.out.clone())];

        fuse_on_write(
            unsafe { &(*state.inputs) },
            unsafe { &mut (*state.outputs) },
            unsafe { &mut (*state.locals_on_write) },
            index,
            values,
            args,
            &state.config_on_write,
        );
    }

    fn len_input<P: ReduceDType>(state: &Self::State<P>) -> u32 {
        ref_len(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            unsafe { &(*state.locals_on_read) },
            &state.config_on_read,
        )
    }

    fn len_output<P: ReduceDType>(state: &Self::State<P>) -> u32 {
        ref_len(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            unsafe { &(*state.locals_on_write) },
            &state.config_on_write,
        )
    }

    fn buffer_len_input<P: ReduceDType>(state: &Self::State<P>) -> u32 {
        ref_buffer_len(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            unsafe { &(*state.locals_on_read) },
            &state.config_on_read,
        )
    }

    fn buffer_len_output<P: ReduceDType>(state: &Self::State<P>) -> u32 {
        ref_buffer_len(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            unsafe { &(*state.locals_on_write) },
            &state.config_on_write,
        )
    }

    fn rank_input<P: ReduceDType>(state: &Self::State<P>) -> u32 {
        ref_rank(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            &state.config_on_read,
        )
    }

    fn rank_output<P: ReduceDType>(state: &Self::State<P>) -> u32 {
        ref_rank(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            &state.config_on_write,
        )
    }

    fn shape_input<P: ReduceDType>(state: &Self::State<P>, dim: u32) -> u32 {
        ref_shape(unsafe { &(*state.locals_on_read) }, dim)
    }

    fn shape_output<P: ReduceDType>(state: &Self::State<P>, dim: u32) -> u32 {
        ref_shape(unsafe { &(*state.locals_on_write) }, dim)
    }

    fn stride_input<P: ReduceDType>(state: &Self::State<P>, dim: u32) -> u32 {
        ref_stride(unsafe { &(*state.locals_on_read) }, dim)
    }

    fn stride_output<P: ReduceDType>(state: &Self::State<P>, dim: u32) -> u32 {
        ref_stride(unsafe { &(*state.locals_on_write) }, dim)
    }
}

#[cube]
impl FusedReduceState {
    pub fn new(
        inputs: &FusedReduceInput,
        outputs: &mut FusedReduceOutput,
        locals_on_read: &mut LocalArgs,
        locals_on_write: &mut LocalArgs,
    ) -> FusedReduceState {
        FusedReduceState {
            inputs: &inputs.global,
            outputs: &mut outputs.global,
            locals_on_read,
            locals_on_write,
            config_on_read: comptime![inputs.config.clone()],
            config_on_write: comptime![outputs.config.clone()],
            input: comptime![inputs.arg.clone()],
            out: comptime![outputs.arg.clone()],
        }
    }
}

impl CubeType for FusedReduceState {
    type ExpandType = FusedReduceStateExpand;
}

impl Init for FusedReduceStateExpand {
    fn init(self, _context: &mut Scope) -> Self {
        self
    }
}

impl CubeDebug for FusedReduceStateExpand {}
