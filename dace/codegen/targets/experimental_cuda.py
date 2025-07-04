# Standard library imports
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

# Third-party imports
import networkx as nx
import sympy

# DaCe core imports
import dace
from dace import data as dt, Memlet
from dace import dtypes, registry, symbolic
from dace.config import Config
from dace.sdfg import SDFG, ScopeSubgraphView, SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView

# DaCe codegen imports
from dace.codegen import common
from dace.codegen.codeobject import CodeObject
from dace.codegen.dispatcher import DefinedType, TargetDispatcher
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.common import update_persistent_desc
from dace.codegen.targets.cpp import (
    codeblock_to_cpp, 
    memlet_copy_to_absolute_strides, 
    mangle_dace_state_struct_name,
    ptr,
    sym2cpp
)
from dace.codegen.targets.target import IllegalCopy, TargetCodeGenerator, make_absolute

# DaCe transformation imports
from dace.transformation.passes import analysis as ap
from dace.transformation.passes.gpustream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.shared_memory_synchronization import DefaultSharedMemorySync

# Experimental CUDA helper imports
from dace.codegen.targets.experimental_cuda_helpers.gpu_stream_manager import GPUStreamManager
from dace.codegen.targets.experimental_cuda_helpers.gpu_utils import symbolic_to_cpp, product, emit_sync_debug_checks

# Type checking imports (conditional)
if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator
    from dace.codegen.targets.cpu import CPUCodeGen


# TODO's easy:
# 3. Emit sync -> yea not easy

# add symbolic_to_cpp !

# TODO's harder:
# 1. Include constant expressions



@registry.autoregister_params(name='experimental_cuda')
class ExperimentalCUDACodeGen(TargetCodeGenerator):
    """ Experimental CUDA code generator."""
    target_name = 'experimental_cuda'
    title = 'CUDA'


    ###########################################################################
    # Initialization & Preprocessing 

    def __init__(self, frame_codegen: 'DaCeCodeGenerator', sdfg: SDFG):

        self._frame: DaCeCodeGenerator = frame_codegen # creates the frame code, orchestrates the code generation for targets
        self._dispatcher: TargetDispatcher = frame_codegen.dispatcher # responsible for dispatching code generation to the appropriate target


        ExperimentalCUDACodeGen._in_kernel_code = False  
        self._cpu_codegen: Optional['CPUCodeGen'] = None


        # NOTE: Moved from preprossessing to here
        self.backend: str = common.get_gpu_backend() 
        self.language = 'cu' if self.backend == 'cuda' else 'cpp'
        target_type = '' if self.backend == 'cuda' else self.backend
        self._codeobject = CodeObject(sdfg.name + '_' + 'cuda',
                                      '',
                                      self.language,
                                      ExperimentalCUDACodeGen,
                                      'CUDA',
                                      target_type=target_type)



        self._localcode = CodeIOStream()
        self._globalcode = CodeIOStream()

        # TODO: init and exitcode seem to serve no purpose actually.
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()

        self._global_sdfg: SDFG = sdfg
        self._toplevel_schedule = None


        # Positions at which to deallocate memory pool arrays
        self.pool_release: Dict[Tuple[SDFG, str], Tuple[SDFGState, Set[nodes.Node]]] = {}
        self.has_pool = False


        # INFO: 
        # Register GPU schedules and storage types for ExperimentalCUDACodeGen.
        # The dispatcher maps GPU-related schedules and storage types to the
        # appropriate code generation functions in this code generator.

        # Register dispatchers
        self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()

        self._dispatcher = frame_codegen.dispatcher
        self._dispatcher.register_map_dispatcher(dtypes.GPU_SCHEDULES_EXPERIMENTAL_CUDACODEGEN, self)
        self._dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)
        self._dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)

        # TODO: Add this to dtypes as well
        gpu_storage = [dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned]

        self._dispatcher.register_array_dispatcher(gpu_storage, self)
        self._dispatcher.register_array_dispatcher(dtypes.StorageType.CPU_Pinned, self)
        for storage in gpu_storage:
            for other_storage in dtypes.StorageType:
                self._dispatcher.register_copy_dispatcher(storage, other_storage, None, self)
                self._dispatcher.register_copy_dispatcher(other_storage, storage, None, self)

        
        # NOTE: 
        # "Register illegal copies" code NOT copied from cuda.py
        # Behavior unclear for me yet.


        ################## New variables ##########################

        self._current_kernel_spec: Optional[KernelSpec] = None
        self._gpu_stream_manager: Optional[GPUStreamManager]  = None

    def preprocess(self, sdfg: SDFG) -> None:
        """
        Preprocess the SDFG to prepare it for GPU code generation. This includes:
        - Handling GPU<->GPU strided copies.
        - Assigning backend GPU streams (e.g., CUDA streams) and creating the GPUStreamManager.
        - Handling memory pool management 
        """
        
        #------------------------- Hanlde GPU<->GPU strided copies --------------------------

        # Find GPU<->GPU strided copies that cannot be represented by a single copy command
        from dace.transformation.dataflow import CopyToMap
        for e, state in list(sdfg.all_edges_recursive()):
            if isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode):
                nsdfg = state.parent
                if (e.src.desc(nsdfg).storage == dtypes.StorageType.GPU_Global
                        and e.dst.desc(nsdfg).storage == dtypes.StorageType.GPU_Global):
                    copy_shape, src_strides, dst_strides, _, _ = memlet_copy_to_absolute_strides(
                        None, nsdfg, state, e, e.src, e.dst)
                    dims = len(copy_shape)

                    # Skip supported copy types
                    if dims == 1:
                        continue
                    elif dims == 2:
                        if src_strides[-1] != 1 or dst_strides[-1] != 1:
                            # NOTE: Special case of continuous copy
                            # Example: dcol[0:I, 0:J, k] -> datacol[0:I, 0:J]
                            # with copy shape [I, J] and strides [J*K, K], [J, 1]
                            try:
                                is_src_cont = src_strides[0] / src_strides[1] == copy_shape[1]
                                is_dst_cont = dst_strides[0] / dst_strides[1] == copy_shape[1]
                            except (TypeError, ValueError):
                                is_src_cont = False
                                is_dst_cont = False
                            if is_src_cont and is_dst_cont:
                                continue
                        else:
                            continue
                    elif dims > 2:
                        if not (src_strides[-1] != 1 or dst_strides[-1] != 1):
                            continue

                    # Turn unsupported copy to a map
                    try:
                        CopyToMap.apply_to(nsdfg, save=False, annotate=False, a=e.src, b=e.dst)
                    except ValueError:  # If transformation doesn't match, continue normally
                        continue


        #------------------------- GPU Stream related Logic --------------------------


        # Register GPU context in state struct
        self._frame.statestruct.append('dace::cuda::Context *gpu_context;')

        # Define backend stream access expression (e.g., CUDA stream handle)
        gpu_stream_access_template = "__state->gpu_context->streams[{gpu_stream}]"  

        # Initialize and configure GPU stream scheduling pass
        gpu_stream_pass = NaiveGPUStreamScheduler()
        gpu_stream_pass.set_gpu_stream_access_template(gpu_stream_access_template)
        assigned_streams = gpu_stream_pass.apply_pass(sdfg, None)

        # Initialize runtime GPU stream manager
        self._gpu_stream_manager = GPUStreamManager(sdfg, assigned_streams, gpu_stream_access_template)

        #----------------- Shared Memory Synchronization related Logic -----------------

        auto_sync = Config.get('compiler', 'cuda', 'auto_syncthreads_insertion')
        if auto_sync:
            DefaultSharedMemorySync().apply_pass(sdfg, None)

        #------------------------- Memory Pool related Logic --------------------------

        # Find points where memory should be released to the memory pool
        self._compute_pool_release(sdfg)

    def _compute_pool_release(self, top_sdfg: SDFG):
        """
        Computes positions in the code generator where a memory pool array is no longer used and
        ``backendFreeAsync`` should be called to release it.

        :param top_sdfg: The top-level SDFG to traverse.
        :raises ValueError: If the backend does not support memory pools.
        """
        # Find release points for every array in every SDFG
        reachability = access_nodes = None
        for sdfg in top_sdfg.all_sdfgs_recursive():
            # Skip SDFGs without memory pool hints
            pooled = set(aname for aname, arr in sdfg.arrays.items()
                         if getattr(arr, 'pool', False) is True and arr.transient)
            if not pooled:
                continue
            self.has_pool = True
            if self.backend != 'cuda':
                raise ValueError(f'Backend "{self.backend}" does not support the memory pool allocation hint')

            # Lazily compute reachability and access nodes
            if reachability is None:
                reachability = ap.StateReachability().apply_pass(top_sdfg, {})
                access_nodes = ap.FindAccessStates().apply_pass(top_sdfg, {})

            reachable = reachability[sdfg.cfg_id]
            access_sets = access_nodes[sdfg.cfg_id]
            for state in sdfg.nodes():
                # Find all data descriptors that will no longer be used after this state
                last_state_arrays: Set[str] = set(
                    s for s in access_sets
                    if s in pooled and state in access_sets[s] and not (access_sets[s] & reachable[state]) - {state})

                anodes = list(state.data_nodes())
                for aname in last_state_arrays:
                    # Find out if there is a common descendant access node.
                    # If not, release at end of state
                    ans = [an for an in anodes if an.data == aname]
                    terminator = None
                    for an1 in ans:
                        if all(nx.has_path(state.nx, an2, an1) for an2 in ans if an2 is not an1):
                            terminator = an1
                            break
                    
                    # Old logic below, now we use the gpu_stream manager which returns nullptr automatically
                    # to all nodes thatdid not got assigned a cuda stream
                    """
                    # Enforce a cuda_stream field so that the state-wide deallocation would work
                    if not hasattr(an1, '_cuda_stream'):
                        an1._cuda_stream = 'nullptr'
                    """

                    # If access node was found, find the point where all its reads are complete
                    terminators = set()
                    if terminator is not None:
                        parent = state.entry_node(terminator)
                        # If within a scope, once all memlet paths going out of that scope are complete,
                        # it is time to release the memory
                        if parent is not None:
                            # Just to be safe, release at end of state (e.g., if misused in Sequential map)
                            terminators = set()
                        else:
                            # Otherwise, find common descendant (or end of state) following the ends of
                            # all memlet paths (e.g., (a)->...->[tasklet]-->...->(b))
                            for e in state.out_edges(terminator):
                                if isinstance(e.dst, nodes.EntryNode):
                                    terminators.add(state.exit_node(e.dst))
                                else:
                                    terminators.add(e.dst)
                            # After all outgoing memlets of all the terminators have been processed, memory
                            # will be released

                    self.pool_release[(sdfg, aname)] = (state, terminators)

            # If there is unfreed pooled memory, free at the end of the SDFG
            unfreed = set(arr for arr in pooled if (sdfg, arr) not in self.pool_release)
            if unfreed:
                # Find or make single sink node
                sinks = sdfg.sink_nodes()
                if len(sinks) == 1:
                    sink = sinks[0]
                elif len(sinks) > 1:
                    sink = sdfg.add_state()
                    for s in sinks:
                        sdfg.add_edge(s, sink)
                else:  # len(sinks) == 0:
                    raise ValueError('End state not found when trying to free pooled memory')

                # Add sink as terminator state
                for arr in unfreed:
                    self.pool_release[(sdfg, arr)] = (sink, set())


    ###########################################################################
    # Determine wheter initializer and finalizer should be called

    @property
    def has_initializer(self) -> bool:
        return True
    @property
    def has_finalizer(self) -> bool:
        return True


    ###########################################################################
    # Scope generation 

    def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                       function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:


        # Import strategies here to avoid circular dependencies
        from dace.codegen.targets.experimental_cuda_helpers.scope_strategies import (
            ScopeGenerationStrategy,
            KernelScopeGenerator,
            ThreadBlockScopeGenerator,
            WarpScopeGenerator
        )


        #--------------- Start of Kernel Function Code Generation --------------------

        if not ExperimentalCUDACodeGen._in_kernel_code:

            # Prepare and cache kernel metadata (name, dimensions, arguments, etc.)
            self._current_kernel_spec = KernelSpec(
                cudaCodeGen=self, sdfg=sdfg, cfg=cfg, dfg_scope=dfg_scope, state_id=state_id
            )

            # Generate wrapper function
            self._generate_kernel_wrapper(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)

            # Enter kernel context and recursively generate device code
            ExperimentalCUDACodeGen._in_kernel_code = True
            kernel_stream = CodeIOStream()
            kernel_function_stream = self._globalcode

            kernel_scope_generator = KernelScopeGenerator(codegen=self)
            if kernel_scope_generator.applicable(sdfg, cfg, dfg_scope, state_id, kernel_function_stream, kernel_stream):
                kernel_scope_generator.generate(sdfg, cfg, dfg_scope, state_id, kernel_function_stream, kernel_stream)
            else:
                raise ValueError(
                    "Invalid kernel configuration: This strategy is only applicable if the "
                    "outermost GPU schedule is of type GPU_Device (most likely cause)."
                )

            # Append generated kernel code to localcode
            self._localcode.write(kernel_stream.getvalue() + '\n')

            # Exit kernel context
            ExperimentalCUDACodeGen._in_kernel_code = False

            # Generate kernel launch
            self._generate_kernel_launch(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)

            return


        #--------------- Nested GPU Scope --------------------

        supported_strategies: List[ScopeGenerationStrategy] = [
            ThreadBlockScopeGenerator(codegen=self),
            WarpScopeGenerator(codegen=self)
        ]

        for strategy in supported_strategies:
            if strategy.applicable(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream):
                strategy.generate(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)
                return

        #--------------- Unsupported Cases --------------------
        # Note: We are inside a nested GPU scope at this point.

        node = dfg_scope.source_nodes()[0]
        schedule_type = node.map.schedule

        if schedule_type == dace.ScheduleType.GPU_Device:
            raise NotImplementedError(
                "Dynamic parallelism (nested GPU_Device schedules) is not supported."
            )

        raise NotImplementedError(
            f"Scope generation for schedule type '{schedule_type}' is not implemented in ExperimentalCUDACodeGen. "
            "Please check for supported schedule types or implement the corresponding strategy."
        )
        
    def _generate_kernel_wrapper(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                             state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:


            scope_entry = dfg_scope.source_nodes()[0]

            kernel_spec: KernelSpec = self._current_kernel_spec
            kernel_name = kernel_spec.kernel_name
            kernel_wrapper_args = kernel_spec.kernel_wrapper_args
            kernel_wrapper_args_typed = kernel_spec.kernel_wrapper_args_typed

            # Declaration of the function which launches the kernel (C++ code)
            function_stream.write('DACE_EXPORTED void __dace_runkernel_%s(%s);\n' % 
                                (kernel_name, ', '.join(kernel_wrapper_args_typed)), cfg, state_id, scope_entry)

            # Calling the function which launches the kernel (C++ code)
            callsite_stream.write( '__dace_runkernel_%s(%s);\n' %
                                (kernel_name, ', '.join(kernel_wrapper_args)), cfg, state_id, scope_entry)
        
    def _generate_kernel_launch(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                                    state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
            
            # NOTE: This generates the function that launches the kernel.
            # Do not confuse it with CUDA's internal "LaunchKernel" API —
            # the generated function *calls* that API, but we also refer to it as a "launch function".

            scope_entry = dfg_scope.source_nodes()[0]

            kernel_spec: KernelSpec = self._current_kernel_spec
            kernel_name = kernel_spec.kernel_name
            kernel_args_as_input = kernel_spec.args_as_input
            kernel_launch_args_typed = kernel_spec.kernel_wrapper_args_typed

            # get kernel dimensions and transform into a c++ string
            grid_dims = kernel_spec.grid_dims
            block_dims = kernel_spec.block_dims
            gdims = ', '.join(symbolic_to_cpp(grid_dims))
            bdims = ', '.join(symbolic_to_cpp(block_dims))

            # cuda/hip stream the kernel belongs to
            gpu_stream = self._gpu_stream_manager.get_stream_node(scope_entry)

            # ----------------- Kernel Launch Function Declaration -----------------------
            self._localcode.write(
                """
                DACE_EXPORTED void __dace_runkernel_{fname}({fargs});
                void __dace_runkernel_{fname}({fargs})
                {{
                """.format(fname=kernel_name, fargs=', '.join(kernel_launch_args_typed)), 
                cfg, state_id, scope_entry
            )


            
            # ----------------- Guard Checks handling -----------------------

            # Ensure that iteration space is neither empty nor negative sized
            single_dimchecks = []
            for gdim in grid_dims:
                # Only emit a guard if we can't statically prove gdim > 0
                if (gdim > 0) != True:
                    single_dimchecks.append(f'(({symbolic_to_cpp(gdim)}) <= 0)')

            dimcheck = ' || '.join(single_dimchecks)

            if dimcheck:
                emptygrid_warning = ''
                if Config.get('debugprint') == 'verbose' or Config.get_bool('compiler', 'cuda', 'syncdebug'):
                    emptygrid_warning = (f'printf("Warning: Skipping launching kernel \\"{kernel_name}\\" '
                                        'due to an empty grid.\\n");')

                self._localcode.write(
                    f'''
                    if ({dimcheck}) {{
                        {emptygrid_warning}
                        return;
                    }}''', cfg, state_id, scope_entry)



            # ----------------- Kernel Launch Invocation -----------------------
            self._localcode.write(
                '''
                void  *{kname}_args[] = {{ {kargs} }};
                gpuError_t __err = {backend}LaunchKernel( (void*){kname}, dim3({gdims}), dim3({bdims}), {kname}_args, {dynsmem}, {stream}
                );
                '''.format(
                    kname=kernel_name,
                    kargs=', '.join(['(void *)&' + arg for arg in kernel_args_as_input]),
                    gdims=gdims,
                    bdims=bdims,
                    dynsmem='0',
                    stream=gpu_stream,
                    backend=self.backend
                ), 
                cfg, state_id, scope_entry
            )
            

            self._localcode.write(f'DACE_KERNEL_LAUNCH_CHECK(__err, "{kernel_name}", {gdims}, {bdims});')
            emit_sync_debug_checks(self.backend, self._localcode)
            self._localcode.write('}')

    ###########################################################################
    # Generation of Memory Copy Logic

    def copy_memory(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                    src_node: Union[nodes.Tasklet, nodes.AccessNode], dst_node: Union[nodes.CodeNode, nodes.AccessNode],
                    edge: Tuple[nodes.Node, str, nodes.Node, str, Memlet], 
                    function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        

        from dace.codegen.targets.experimental_cuda_helpers.copy_strategies import (
            CopyContext,
            CopyStrategy,
            OutOfKernelCopyStrategy,
            SyncCollaboritveGPUCopyStrategy,
            AsyncCollaboritveGPUCopyStrategy,
            FallBackGPUCopyStrategy
        )

        context = CopyContext(self, self._gpu_stream_manager, state_id, src_node, dst_node, edge,
                            sdfg, cfg, dfg, callsite_stream)

        # Order matters: fallback must come last
        strategies: List[CopyStrategy] = [
            OutOfKernelCopyStrategy(),
            SyncCollaboritveGPUCopyStrategy(),
            AsyncCollaboritveGPUCopyStrategy(),
            FallBackGPUCopyStrategy()
        ]

        for strategy in strategies:
            if strategy.applicable(context):
                strategy.generate_copy(context)
                return

        raise RuntimeError("No applicable GPU memory copy strategy found (this should not happen).")

    #############################################################################
    # Predicates for Dispatcher

    def state_dispatch_predicate(self, sdfg, state):
        """
        Determines whether a state should be handled by this
        code generator (`ExperimentalCUDACodeGen`).

        Returns True if the generator is currently generating kernel code.
        """
        return ExperimentalCUDACodeGen._in_kernel_code

    def node_dispatch_predicate(self, sdfg, state, node):
        """
        Determines whether a node should be handled by this
        code generator (`ExperimentalCUDACodeGen`).

        Returns True if:
        - The node has a GPU schedule handled by this backend, or
        - The generator is currently generating kernel code.
        """
        schedule = getattr(node, 'schedule', None)

        if schedule in dtypes.GPU_SCHEDULES_EXPERIMENTAL_CUDACODEGEN:
            return True
        
        if ExperimentalCUDACodeGen._in_kernel_code:
            return True
        
        return False
    
    #############################################################################
    # Nested SDFG related, testing phase

    def generate_state(self,
                       sdfg: SDFG,
                       cfg: ControlFlowRegion,
                       state: SDFGState,
                       function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream,
                       generate_state_footer: bool = False) -> None:
        
        # User frame code  to generate state
        self._frame.generate_state(sdfg, cfg, state, function_stream, callsite_stream)

        # Special: Release of pooled memory if not in device code that need to be released her
        if not ExperimentalCUDACodeGen._in_kernel_code:

            handled_keys = set()
            backend = self.backend
            for (pool_sdfg, name), (pool_state, _) in self.pool_release.items():

                if (pool_sdfg is not sdfg) or (pool_state is not state):
                    continue

                data_descriptor = pool_sdfg.arrays[name]
                ptrname = ptr(name, data_descriptor, pool_sdfg, self._frame)

                # Adjust if there is an offset
                if isinstance(data_descriptor, dt.Array) and data_descriptor.start_offset != 0:
                    ptrname = f'({ptrname} - {sym2cpp(data_descriptor.start_offset)})'

                # Free the memory
                callsite_stream.write(f'DACE_GPU_CHECK({backend}Free({ptrname}));\n', pool_sdfg)

                emit_sync_debug_checks(self.backend, callsite_stream)

                # We handled the key (pool_sdfg, name) and can remove it later
                handled_keys.add((pool_sdfg, name))

            # Delete the handled keys here (not in the for loop, which would cause issues)
            for key in handled_keys:
                del self.pool_release[key]
    
    def generate_node(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int, node: nodes.Node,
                      function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        

        # get the generating function's name
        gen = getattr(self, '_generate_' + type(node).__name__, False)

        # if it is not implemented, use generate node of cpu impl
        if gen is not False: 
            gen(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
        elif type(node).__name__ == 'MapExit' and node.schedule in dtypes.GPU_SCHEDULES_EXPERIMENTAL_CUDACODEGEN:
            # Special case: It is a MapExit but from a GPU_schedule- the MapExit is already
            # handled by a KernelScopeManager instance. Otherwise cpu_codegen will close it
            return
        else:
            self._cpu_codegen.generate_node(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    def generate_nsdfg_header(self, sdfg, cfg, state, state_id, node, memlet_references, sdfg_label):
        return 'DACE_DFI ' + self._cpu_codegen.generate_nsdfg_header(
            sdfg, cfg, state, state_id, node, memlet_references, sdfg_label, state_struct=False)

    def generate_nsdfg_call(self, sdfg, cfg, state, node, memlet_references, sdfg_label):
        return self._cpu_codegen.generate_nsdfg_call(sdfg, cfg, state, node, memlet_references,
                                                     sdfg_label, state_struct=False)

    def generate_nsdfg_arguments(self, sdfg, cfg, dfg, state, node):
        result = self._cpu_codegen.generate_nsdfg_arguments(sdfg, cfg, dfg, state, node)
        return result

    def _generate_NestedSDFG(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                             node: nodes.NestedSDFG, function_stream: CodeIOStream,
                             callsite_stream: CodeIOStream) -> None:
        old_schedule = self._toplevel_schedule
        self._toplevel_schedule = node.schedule
        old_codegen = self._cpu_codegen.calling_codegen
        self._cpu_codegen.calling_codegen = self

        self._cpu_codegen._generate_NestedSDFG(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

        self._cpu_codegen.calling_codegen = old_codegen
        self._toplevel_schedule = old_schedule
 

    #######################################################################
    # Array Declaration, Allocation and Deallocation

    def declare_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                    node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                    declaration_stream: CodeIOStream) -> None:
        

        ptrname = ptr(node.data, nodedesc, sdfg, self._frame)
        fsymbols = self._frame.symbols_and_constants(sdfg)

        # ----------------- Guard checks --------------------

        # NOTE: `dfg` is None iff `nodedesc` is non-free symbol dependent (see DaCeCodeGenerator.determine_allocation_lifetime).
        # We avoid `is_nonfree_sym_dependent` when dfg is None and `nodedesc` is a View.
        if dfg and not sdutil.is_nonfree_sym_dependent(node, nodedesc, dfg, fsymbols):
            raise NotImplementedError(
                "declare_array is only for variables that require separate declaration and allocation.")
        
        if nodedesc.storage == dtypes.StorageType.GPU_Shared:
            raise NotImplementedError("Dynamic shared memory unsupported")

        if nodedesc.storage == dtypes.StorageType.Register:
            raise ValueError("Dynamic allocation of registers is not allowed")

        if nodedesc.storage not in {dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned}:
            raise NotImplementedError(
                f"CUDA: Unimplemented storage type {nodedesc.storage.name}.")

        if self._dispatcher.declared_arrays.has(ptrname):
            return  # Already declared


        # ----------------- Declaration --------------------
        dataname = node.data
        array_ctype = f'{nodedesc.dtype.ctype} *'
        declaration_stream.write(f'{array_ctype} {dataname};\n', cfg, state_id, node)
        self._dispatcher.declared_arrays.add(dataname, DefinedType.Pointer, array_ctype)

    def allocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                       node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                       declaration_stream: CodeIOStream, allocation_stream: CodeIOStream) -> None:
        """
        Maybe document here that this also does declaration and that declare_array only declares specific
        kind of data
        """

        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

        # ------------------- Guard checks -------------------

        # Skip if variable is already defined
        if self._dispatcher.defined_vars.has(dataname):
            return

        if isinstance(nodedesc, (dace.data.View, dace.data.Reference)):
            return NotImplementedError("Pointers and References not implemented in ExperimentalCUDACodeGen")

        if isinstance(nodedesc, dace.data.Stream):
            raise NotImplementedError("allocate_stream not implemented in ExperimentalCUDACodeGen")

        # No clue what is happening here
        if nodedesc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
            nodedesc = update_persistent_desc(nodedesc, sdfg)

        # ------------------- Allocation/Declaration -------------------

        # Call the appropriate handler based on storage type
        gen = getattr(self, f'_prepare_{nodedesc.storage.name}_array', None)
        if gen:
            gen(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream, allocation_stream)
        else:
            raise NotImplementedError(f'CUDA: Unimplemented storage type {nodedesc.storage}')

    def _prepare_GPU_Global_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                      node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                                      declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):
        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

        # ------------------- Declaration -------------------
        array_ctype = f'{nodedesc.dtype.ctype} *'
        declared = self._dispatcher.declared_arrays.has(dataname)

        if not declared:
            declaration_stream.write(f'{array_ctype} {dataname};\n', cfg, state_id, node)

        self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)

        # ------------------- Allocation -------------------
        arrsize = nodedesc.total_size
        arrsize_malloc = f'{symbolic_to_cpp(arrsize)} * sizeof({nodedesc.dtype.ctype})'

        if nodedesc.pool:
            gpu_stream_manager = self._gpu_stream_manager
            gpu_stream = gpu_stream_manager.get_stream_node(node)
            if gpu_stream != 'nullptr':
                gpu_stream = f'__state->gpu_context->streams[{gpu_stream}]'
            allocation_stream.write(
                f'DACE_GPU_CHECK({self.backend}MallocAsync((void**)&{dataname}, {arrsize_malloc}, {gpu_stream}));\n',
                cfg, state_id, node
            )
            emit_sync_debug_checks(self.backend, allocation_stream)
        else:
            # Strides are left to the user's discretion
            allocation_stream.write(
                f'DACE_GPU_CHECK({self.backend}Malloc((void**)&{dataname}, {arrsize_malloc}));\n',
                cfg, state_id, node
            )

        # ------------------- Initialization -------------------
        if node.setzero:
            allocation_stream.write(
                f'DACE_GPU_CHECK({self.backend}Memset({dataname}, 0, {arrsize_malloc}));\n',
                cfg, state_id, node
            )

        if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
            allocation_stream.write(
                f'{dataname} += {symbolic_to_cpp(nodedesc.start_offset)};\n',
                cfg, state_id, node
            )

    def _prepare_CPU_Pinned_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                      node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                                      declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):
        
        dataname = ptr(node.data, nodedesc, sdfg, self._frame)
    
        # ------------------- Declaration -------------------
        array_ctype = f'{nodedesc.dtype.ctype} *'
        declared = self._dispatcher.declared_arrays.has(dataname)

        if not declared:
            declaration_stream.write(f'{array_ctype} {dataname};\n', cfg, state_id, node)

        self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)


        # ------------------- Allocation -------------------
        arrsize = nodedesc.total_size
        arrsize_malloc = f'{symbolic_to_cpp(arrsize)} * sizeof({nodedesc.dtype.ctype})'

        # Strides are left to the user's discretion
        allocation_stream.write(
            f'DACE_GPU_CHECK({self.backend}MallocHost(&{dataname}, {arrsize_malloc}));\n',
            cfg, state_id, node
            )
        if node.setzero:
            allocation_stream.write(
                f'memset({dataname}, 0, {arrsize_malloc});\n',
                cfg, state_id, node
                )
            
        if nodedesc.start_offset != 0:
            allocation_stream.write(
                f'{dataname} += {symbolic_to_cpp(nodedesc.start_offset)};\n',
                cfg, state_id, node
                )

    def _prepare_GPU_Shared_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                      node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                                      declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):

        dataname = ptr(node.data, nodedesc, sdfg, self._frame)
        arrsize = nodedesc.total_size


        # ------------------- Guard checks -------------------
        if symbolic.issymbolic(arrsize, sdfg.constants): 
            raise NotImplementedError('Dynamic shared memory unsupported')
        if nodedesc.start_offset != 0:
            raise NotImplementedError('Start offset unsupported for shared memory')
        

        # ------------------- Declaration -------------------
        array_ctype = f'{nodedesc.dtype.ctype} *'

        declaration_stream.write(
            f'__shared__ {nodedesc.dtype.ctype} {dataname}[{symbolic_to_cpp(arrsize)}];\n',
            cfg, state_id, node
            )
        
        self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)


        # ------------------- Initialization -------------------
        if node.setzero:
            allocation_stream.write(
                f'dace::ResetShared<{nodedesc.dtype.ctype}, {", ".join(symbolic_to_cpp(self._current_kernel_spec.block_dims))}, {symbolic_to_cpp(arrsize)}, '
                f'1, false>::Reset({dataname});\n',
                cfg, state_id, node
            )

    def _prepare_Register_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                      node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                                      declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):

        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

        # ------------------- Guard checks -------------------
        if symbolic.issymbolic(arrsize, sdfg.constants):
            raise ValueError('Dynamic allocation of registers not allowed')
        if nodedesc.start_offset != 0:
            raise NotImplementedError('Start offset unsupported for registers')
        

        # ------------------- Declaration & Initialization -------------------
        arrsize = nodedesc.total_size
        array_ctype = '{nodedesc.dtype.ctype} *'
        init_clause = ' = {0}' if node.setzero else ''

        declaration_stream.write(
            f'{nodedesc.dtype.ctype} {dataname}[{symbolic_to_cpp(arrsize)}]{init_clause};\n',
            cfg, state_id, node
            )
        
        self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)

    def deallocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                         node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                         callsite_stream: CodeIOStream) -> None:
        

        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

        # Adjust offset if needed
        if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
            dataname = f'({dataname} - {symbolic_to_cpp(nodedesc.start_offset)})'

        # Remove declaration info
        if self._dispatcher.declared_arrays.has(dataname):
            is_global = nodedesc.lifetime in (
                dtypes.AllocationLifetime.Global,
                dtypes.AllocationLifetime.Persistent,
                dtypes.AllocationLifetime.External,
            )
            self._dispatcher.declared_arrays.remove(dataname, is_global=is_global)


        # Special case: Stream
        if isinstance(nodedesc, dace.data.Stream):
            raise NotImplementedError('stream code is not implemented in ExperimentalCUDACodeGen (yet)')
        
        # Special case: View - no deallocation
        if isinstance(nodedesc, dace.data.View):
            return


        # Main deallocation logic by storage type
        if nodedesc.storage == dtypes.StorageType.GPU_Global:
            if not nodedesc.pool:  # If pooled, will be freed somewhere else
                callsite_stream.write(
                    f'DACE_GPU_CHECK({self.backend}Free({dataname}));\n', 
                    cfg, state_id, node
                    )

        elif nodedesc.storage == dtypes.StorageType.CPU_Pinned:
            callsite_stream.write(
                f'DACE_GPU_CHECK({self.backend}FreeHost({dataname}));\n', cfg, state_id, node)
            
        elif nodedesc.storage in {dtypes.StorageType.GPU_Shared, dtypes.StorageType.Register}:
            # No deallocation needed
            return
        
        else:
            raise NotImplementedError(f'Deallocation not implemented for storage type: {nodedesc.storage.name}')

    def get_generated_codeobjects(self):

        # My comment: first part creates the header and stores it in a object property
        fileheader = CodeIOStream()

        self._frame.generate_fileheader(self._global_sdfg, fileheader, 'cuda')

        # My comment: takes codeblocks and transforms it nicely to code
        initcode = CodeIOStream()
        for sd in self._global_sdfg.all_sdfgs_recursive():
            if None in sd.init_code:
                initcode.write(codeblock_to_cpp(sd.init_code[None]), sd)
            if 'cuda' in sd.init_code:
                initcode.write(codeblock_to_cpp(sd.init_code['cuda']), sd)
        initcode.write(self._initcode.getvalue())

        # My comment: takes codeblocks and transforms it nicely to code- probably same as before now for exit code
        exitcode = CodeIOStream()
        for sd in self._global_sdfg.all_sdfgs_recursive():
            if None in sd.exit_code:
                exitcode.write(codeblock_to_cpp(sd.exit_code[None]), sd)
            if 'cuda' in sd.exit_code:
                exitcode.write(codeblock_to_cpp(sd.exit_code['cuda']), sd)
        exitcode.write(self._exitcode.getvalue())


        # My comment: Uses GPU backend (NVIDIA or AMD) to get correct header files
        if self.backend == 'cuda':
            backend_header = 'cuda_runtime.h'
        elif self.backend == 'hip':
            backend_header = 'hip/hip_runtime.h'
        else:
            raise NameError('GPU backend "%s" not recognized' % self.backend)

        # My comment: Seems to get all function params, needed for later
        params_comma = self._global_sdfg.init_signature(free_symbols=self._frame.free_symbols(self._global_sdfg))
        if params_comma:
            params_comma = ', ' + params_comma

        #My comment looks life Memory information
        pool_header = ''
        if self.has_pool:
            poolcfg = Config.get('compiler', 'cuda', 'mempool_release_threshold')
            pool_header = f'''
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, 0);
    uint64_t threshold = {poolcfg if poolcfg != -1 else 'UINT64_MAX'};
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
'''

        # My comment: Looks like a "base" template, where more details will probably be added later
        self._codeobject.code = """
#include <{backend_header}>
#include <dace/dace.h>

// New, cooperative groups and asnyc copy
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

namespace cg = cooperative_groups;

{file_header}

DACE_EXPORTED int __dace_init_experimental_cuda({sdfg_state_name} *__state{params});
DACE_EXPORTED int __dace_exit_experimental_cuda({sdfg_state_name} *__state);

{other_globalcode}

int __dace_init_experimental_cuda({sdfg_state_name} *__state{params}) {{
    int count;

    // Check that we are able to run {backend} code
    if ({backend}GetDeviceCount(&count) != {backend}Success)
    {{
        printf("ERROR: GPU drivers are not configured or {backend}-capable device "
               "not found\\n");
        return 1;
    }}
    if (count == 0)
    {{
        printf("ERROR: No {backend}-capable devices found\\n");
        return 2;
    }}

    // Initialize {backend} before we run the application
    float *dev_X;
    DACE_GPU_CHECK({backend}Malloc((void **) &dev_X, 1));
    DACE_GPU_CHECK({backend}Free(dev_X));

    {pool_header}

    __state->gpu_context = new dace::cuda::Context({nstreams}, {nevents});

    // Create {backend} streams and events
    for(int i = 0; i < {nstreams}; ++i) {{
        DACE_GPU_CHECK({backend}StreamCreateWithFlags(&__state->gpu_context->internal_streams[i], {backend}StreamNonBlocking));
        __state->gpu_context->streams[i] = __state->gpu_context->internal_streams[i]; // Allow for externals to modify streams
    }}
    for(int i = 0; i < {nevents}; ++i) {{
        DACE_GPU_CHECK({backend}EventCreateWithFlags(&__state->gpu_context->events[i], {backend}EventDisableTiming));
    }}

    {initcode}

    return 0;
}}

int __dace_exit_experimental_cuda({sdfg_state_name} *__state) {{
    {exitcode}

    // Synchronize and check for CUDA errors
    int __err = static_cast<int>(__state->gpu_context->lasterror);
    if (__err == 0)
        __err = static_cast<int>({backend}DeviceSynchronize());

    // Destroy {backend} streams and events
    for(int i = 0; i < {nstreams}; ++i) {{
        DACE_GPU_CHECK({backend}StreamDestroy(__state->gpu_context->internal_streams[i]));
    }}
    for(int i = 0; i < {nevents}; ++i) {{
        DACE_GPU_CHECK({backend}EventDestroy(__state->gpu_context->events[i]));
    }}

    delete __state->gpu_context;
    return __err;
}}

DACE_EXPORTED bool __dace_gpu_set_stream({sdfg_state_name} *__state, int streamid, gpuStream_t stream)
{{
    if (streamid < 0 || streamid >= {nstreams})
        return false;

    __state->gpu_context->streams[streamid] = stream;

    return true;
}}

DACE_EXPORTED void __dace_gpu_set_all_streams({sdfg_state_name} *__state, gpuStream_t stream)
{{
    for (int i = 0; i < {nstreams}; ++i)
        __state->gpu_context->streams[i] = stream;
}}

{localcode}
""".format(params=params_comma,
           sdfg_state_name=mangle_dace_state_struct_name(self._global_sdfg),
           initcode=initcode.getvalue(),
           exitcode=exitcode.getvalue(),
           other_globalcode=self._globalcode.getvalue(),
           localcode=self._localcode.getvalue(),
           file_header=fileheader.getvalue(),
           nstreams=self._gpu_stream_manager.num_gpu_streams,
           nevents=self._gpu_stream_manager.num_gpu_events,
           backend=self.backend,
           backend_header=backend_header,
           pool_header=pool_header,
           sdfg=self._global_sdfg)

        return [self._codeobject]

    #######################################################################
    # Compilation Related

    @staticmethod
    def cmake_options():
        options = []

        # Override CUDA toolkit
        if Config.get('compiler', 'cuda', 'path'):
            options.append("-DCUDA_TOOLKIT_ROOT_DIR=\"{}\"".format(
                Config.get('compiler', 'cuda', 'path').replace('\\', '/')))

        # Get CUDA architectures from configuration
        backend = common.get_gpu_backend()
        if backend == 'cuda':
            cuda_arch = Config.get('compiler', 'cuda', 'cuda_arch').split(',')
            cuda_arch = [ca for ca in cuda_arch if ca is not None and len(ca) > 0]

            cuda_arch = ';'.join(cuda_arch)
            options.append(f'-DDACE_CUDA_ARCHITECTURES_DEFAULT="{cuda_arch}"')

            flags = Config.get("compiler", "cuda", "args")
            options.append("-DCMAKE_CUDA_FLAGS=\"{}\"".format(flags))

        if backend == 'hip':
            hip_arch = Config.get('compiler', 'cuda', 'hip_arch').split(',')
            hip_arch = [ha for ha in hip_arch if ha is not None and len(ha) > 0]

            flags = Config.get("compiler", "cuda", "hip_args")
            flags += ' ' + ' '.join(
                '--offload-arch={arch}'.format(arch=arch if arch.startswith("gfx") else "gfx" + arch)
                for arch in hip_arch)
            options.append("-DEXTRA_HIP_FLAGS=\"{}\"".format(flags))

        if Config.get('compiler', 'cpu', 'executable'):
            host_compiler = make_absolute(Config.get("compiler", "cpu", "executable"))
            options.append("-DCUDA_HOST_COMPILER=\"{}\"".format(host_compiler))

        return options  

    #######################################################################
    # Callback to CPU codegen

    def define_out_memlet(self, sdfg: SDFG, cfg: ControlFlowRegion, state_dfg: StateSubgraphView, state_id: int,
                          src_node: nodes.Node, dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet],
                          function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        self._cpu_codegen.define_out_memlet(sdfg, cfg, state_dfg, state_id, src_node, dst_node, edge, function_stream,
                                            callsite_stream)

    def process_out_memlets(self, *args, **kwargs):
        # Call CPU implementation with this code generator as callback
        self._cpu_codegen.process_out_memlets(*args, codegen=self, **kwargs)





#########################################################################
# helper classes and functions

# NOTE: I had to redefine this function locally to not modify other files 
# and ensure backwards compatibility with the old cudacodegen
def ptr(name: str, desc: dace.data.Data, sdfg: SDFG = None, framecode=None) -> str:
    """
    Returns a string that points to the data based on its name and descriptor.

    This function should be in cpp.py, but for ExperimentalCUDACodeGen I defined 
    it here to not modify it there, s.t. we have backwards compatibility.

    :param name: Data name.
    :param desc: Data descriptor.
    :return: C-compatible name that can be used to access the data.
    """
    from dace.codegen.targets.framecode import DaCeCodeGenerator  # Avoid import loop
    framecode: DaCeCodeGenerator = framecode

    if '.' in name:
        root = name.split('.')[0]
        if root in sdfg.arrays and isinstance(sdfg.arrays[root], dace.data.Structure):
            name = name.replace('.', '->')

    # Special case: If memory is persistent and defined in this SDFG, add state
    # struct to name
    if (desc.transient and desc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External)):

        if desc.storage == dtypes.StorageType.CPU_ThreadLocal:  # Use unambiguous name for thread-local arrays
            return f'__{sdfg.cfg_id}_{name}'
        elif not ExperimentalCUDACodeGen._in_kernel_code:  # GPU kernels cannot access state
            return f'__state->__{sdfg.cfg_id}_{name}'
        elif (sdfg, name) in framecode.where_allocated and framecode.where_allocated[(sdfg, name)] is not sdfg:
            return f'__{sdfg.cfg_id}_{name}'
    elif (desc.transient and sdfg is not None and framecode is not None and (sdfg, name) in framecode.where_allocated
          and framecode.where_allocated[(sdfg, name)] is not sdfg):
        # Array allocated for another SDFG, use unambiguous name
        return f'__{sdfg.cfg_id}_{name}'

    return name


# This one is closely linked to the ExperimentalCUDACodeGen. In fact,
# it only exists to not have to much attributes and methods in the ExperimentalCUDACodeGen
# and to group Kernel specific methods & information. Thus, KernelSpec should remain in this file
class KernelSpec:
    """
    A helper class to encapsulate information required for working with kernels.
    This class provides a structured way to store and retrieve kernel parameters.
    """

    def __init__(self, cudaCodeGen: ExperimentalCUDACodeGen, sdfg: SDFG, cfg: ControlFlowRegion,
                 dfg_scope: ScopeSubgraphView, state_id: int):
        
        
        kernel_entry_node = dfg_scope.source_nodes()[0]
        kernel_exit_node = dfg_scope.sink_nodes()[0]
        state: SDFGState = cfg.state(state_id)

        self._kernel_entry_node: nodes.MapEntry  = kernel_entry_node
        self._kernel_map: nodes.Map = kernel_entry_node.map

        # Kernel name
        self._kernel_name: str = '%s_%d_%d_%d' % (kernel_entry_node.map.label, cfg.cfg_id, state.block_id, state.node_id(kernel_entry_node))

        # Kernel arguments
        arglist = {}
        for state, node, defined_syms in sdutil.traverse_sdfg_with_defined_symbols(sdfg, recursive=True):
            if node is kernel_entry_node:
                shared_transients = state.parent.shared_transients()
                arglist = state.scope_subgraph(node).arglist(defined_syms, shared_transients)
                break
        self._args: Dict = arglist

        """
        # const args
        input_params = set(e.data.data for e in state.in_edges(kernel_entry_node))
        output_params = set(e.data.data for e in state.out_edges(kernel_exit_node))
        toplevel_params = set(node.data for node in dfg_scope.nodes()
                            if isinstance(node, nodes.AccessNode) and sdfg.arrays[node.data].toplevel)
        dynamic_inputs = set(e.data.data for e in dace.sdfg.dynamic_map_inputs(state, kernel_entry_node))

        const_args = input_params - (output_params | toplevel_params | dynamic_inputs)
        self._args_typed: list[str] = [('const ' if aname in const_args else '') + adata.as_arg(name=aname) for aname, adata in self._args.items()]
        """

        # args typed correctly and as input
        self._args_typed: list[str] = [adata.as_arg(name=aname) for aname, adata in self._args.items()]
        self._args_as_input: list[str] = [ptr(aname, adata, sdfg, cudaCodeGen._frame) for aname, adata in self._args.items()]

        # Used for the kernel wrapper function, be careful: a change in the name __state will probably lead to compilation errors
        state_param: list[str] = [f'{mangle_dace_state_struct_name(cudaCodeGen._global_sdfg)} *__state']

        self._kernel_wrapper_args: list[str] = ['__state'] + self._args_as_input
        self._kernel_wrapper_args_typed: list[str] = state_param + self._args_typed

        # Kernel dimensions
        self._grid_dims, self._block_dims, self._has_tbmap = self._get_kernel_dimensions(dfg_scope)

        # C type (as string) of thread, block and warp indices
        self._gpu_index_ctype: str = self.get_gpu_index_ctype()

        # Set warp size of the kernel
        if cudaCodeGen.backend not in ['cuda', 'hip']:
            raise ValueError(
                f"Unsupported backend '{cudaCodeGen.backend}' in ExperimentalCUDACodeGen. "
                "Only 'cuda' and 'hip' are supported."
                )
        
        warp_size_key = 'cuda_warp_size' if cudaCodeGen.backend == 'cuda' else 'hip_warp_size'
        self._warpSize = Config.get('compiler', 'cuda', warp_size_key)


    def get_gpu_index_ctype(self, config_key='gpu_index_type') -> str:
        """
        Retrieves the GPU index data type as a C type string (for thread, block, warp indices)
        from the configuration and if it matches a DaCe data type.

        Raises:
            ValueError: If the configured type does not match a DaCe data type.

        Returns:
            str: 
                The C type string corresponding to the configured GPU index type. 
                Used for defining thread, block, and warp indices in the generated code.
        """
        type_name = Config.get('compiler', 'cuda', config_key)
        dtype = getattr(dtypes, type_name, None)
        if not isinstance(dtype, dtypes.typeclass):
            raise ValueError(
                f'Invalid {config_key} "{type_name}" configured (used for thread, block, and warp indices): '
                'no matching DaCe data type found.\n'
                'Please use a valid type from dace.dtypes (e.g., "int32", "uint64").'
            )
        return dtype.ctype


    def _get_kernel_dimensions(self, dfg_scope: ScopeSubgraphView):
        """
        Determines a GPU kernel's grid/block dimensions from map scopes.

        Ruleset for kernel dimensions:

            1. If only one map (device-level) exists, of an integer set ``S``,
                the block size is ``32x1x1`` and grid size is ``ceil(|S|/32)`` in
                1st dimension.
            2. If nested thread-block maps exist ``(T_1,...,T_n)``, grid
                size is ``|S|`` and block size is ``max(|T_1|,...,|T_n|)`` with
                block specialization.
            3. If block size can be overapproximated, it is (for
                dynamically-sized blocks that are bounded by a
                predefined size).
            4. If nested device maps exist, behavior is an error is thrown 
               in the generate_scope function. Nested device maps are not supported
               anymore.

        :note: Kernel dimensions are separate from the map
               variables, and they should be treated as such.
        :note: To make use of the grid/block 3D registers, we use multi-
               dimensional kernels up to 3 dimensions, and flatten the
               rest into the third dimension.
        """


        # Extract the subgraph of the kernel entry map
        launch_scope = dfg_scope.scope_subgraph(self._kernel_entry_node)

        # Collect all relevant maps affecting launch (i.e. grid and block) dimensions
        affecting_maps = self._get_maps_affecting_launch_dims(launch_scope)

        # Filter for ThreadBlock maps
        threadblock_maps = [(tbmap, sym_map) for tbmap, sym_map in affecting_maps
                                if tbmap.schedule == dtypes.ScheduleType.GPU_ThreadBlock]
        
        # Determine if we fall back to default block size (which also affects grid size)
        no_block_info: bool = len(threadblock_maps) == 0 and self._kernel_map.gpu_block_size is None
        
        if no_block_info:
            block_size, grid_size = self._compute_default_block_and_grid()
        else:
            block_size, grid_size = self._compute_block_and_grid_from_maps(threadblock_maps)


        return grid_size, block_size, len(threadblock_maps) > 0


    def _compute_default_block_and_grid(self):
        """
        Fallback when no gpu_block_size (i.e. self._kernel_map.gpu_block_size is None)
        or GPU_ThreadBlock maps are defined:

        Uses default_block_size (e.g. [32,1,1]) on the whole domain S (assuming 1 dimensional),
        producing block=[32,1,1] and grid=[ceil(|S|/32),1,1].

        Special case: if the block has more active (non-1) dimensions than S,
        extra block dimensions are collapsed into the last active slot.
        """

        kernel_map_label = self._kernel_entry_node.map.label
        default_block_size_config = Config.get('compiler', 'cuda', 'default_block_size')

        # 1) Reject unsupported 'max' setting
        if default_block_size_config == 'max':
            # TODO: does this make sense? what is meant with dynamic here?
            raise NotImplementedError('max dynamic block size unimplemented') 
        
        # 2) Warn that we're falling back to config
        warnings.warn(
            f'No `gpu_block_size` property specified on map "{kernel_map_label}". '
            f'Falling back to the configuration entry `compiler.cuda.default_block_size`: {default_block_size_config}. '
            'You can either specify the block size to use with the gpu_block_size property, '
            'or by adding nested `GPU_ThreadBlock` maps, which map work to individual threads. '
            'For more information, see https://spcldace.readthedocs.io/en/latest/optimization/gpu.html')
        

        # 3) Normalize the total iteration space size (len(X),len(Y),len(Z)…) to 3D
        raw_domain = list(self._kernel_map.range.size(True))[::-1]
        kernel_domain_size = self._to_3d_dims(raw_domain)

        # 4) Parse & normalize the default block size to 3D
        default_block_size = [int(x) for x in default_block_size_config.split(',')]
        default_block_size = self._to_3d_dims(default_block_size)

        # 5) If block has more "active" dims than the domain, collapse extras
        active_block_dims = max(1, sum(1 for b in default_block_size if b != 1))
        active_grid_dims  = max(1, sum(1 for g in kernel_domain_size  if g != 1))

        if active_block_dims > active_grid_dims:
            tail_product = product(default_block_size[active_grid_dims:])
            block_size = default_block_size[:active_grid_dims] + [1] * (3 - active_grid_dims)
            block_size[active_grid_dims - 1] *= tail_product
            warnings.warn(f'Default block size has more dimensions ({active_block_dims}) than kernel dimensions '
                            f'({active_grid_dims}) in map "{kernel_map_label}". Linearizing block '
                            f'size to {block_size}. Consider setting the ``gpu_block_size`` property.')
        else:
            block_size = default_block_size

        # 6) Compute the final grid size per axis: ceil(domain / block)
        grid_size = [symbolic.int_ceil(gs, bs) for gs, bs in zip(kernel_domain_size, block_size)]


        # 7) Check block size against configured CUDA hardware limits
        self._validate_block_size_limits(block_size)

        return block_size, grid_size


    def _compute_block_and_grid_from_maps(self, tb_maps_sym_map):
        # TODO: also provide a description here in docstring


        kernel_entry_node = self._kernel_entry_node
        
        # Compute kernel grid size
        raw_grid_size = self._kernel_map.range.size(True)[::-1]
        grid_size = self._to_3d_dims(raw_grid_size)

        # Determine block size, using gpu_block_size override if specified
        # NOTE: this must be done on the original list! otherwise error
        block_size = self._kernel_map.gpu_block_size
        if block_size is not None:
            block_size = self._to_3d_dims(block_size)


        # Find all thread-block maps to determine overall block size
        detected_block_sizes = [block_size] if block_size is not None else []
        for tbmap, sym_map in tb_maps_sym_map:
            tbsize = [s.subs(list(sym_map.items())) for s in tbmap.range.size()[::-1]]

            # Over-approximate block size (e.g. min(N,(i+1)*32)-i*32 --> 32)
            # The partial trailing thread-block is emitted as an if-condition
            # that returns on some of the participating threads
            tbsize = [symbolic.overapproximate(s) for s in tbsize]

            # To Cuda compatible block dimension description
            tbsize = self._to_3d_dims(tbsize)
 
            if len(detected_block_sizes) == 0:
                block_size = tbsize
            else:
                block_size = [sympy.Max(sz, bbsz) for sz, bbsz in zip(block_size, tbsize)]

            if block_size != tbsize or len(detected_block_sizes) == 0:
                detected_block_sizes.append(tbsize)



        #-------------- Error handling and warnings ------------------------

        # TODO: If grid/block sizes contain elements only defined within the
        #       kernel, raise an invalid SDFG exception and recommend
        #       overapproximation.

        kernel_map_label = kernel_entry_node.map.label
        if len(detected_block_sizes) > 1:
            # Error when both gpu_block_size and thread-block maps were defined and conflict
            if kernel_entry_node.map.gpu_block_size is not None:
                raise ValueError('Both the `gpu_block_size` property and internal thread-block '
                                    'maps were defined with conflicting sizes for kernel '
                                    f'"{kernel_map_label}" (sizes detected: {detected_block_sizes}). '
                                    'Use `gpu_block_size` only if you do not need access to individual '
                                    'thread-block threads, or explicit block-level synchronization (e.g., '
                                    '`__syncthreads`). Otherwise, use internal maps with the `GPU_Threadblock` or '
                                    '`GPU_ThreadBlock_Dynamic` schedules. For more information, see '
                                    'https://spcldace.readthedocs.io/en/latest/optimization/gpu.html')

            warnings.warn('Multiple thread-block maps with different sizes detected for '
                            f'kernel "{kernel_map_label}": {detected_block_sizes}. '
                            f'Over-approximating to block size {block_size}.\n'
                            'If this was not the intent, try tiling one of the thread-block maps to match.')
        
        # Check block size against configured CUDA hardware limits
        self._validate_block_size_limits(block_size)
        
        return block_size, grid_size


    def _validate_block_size_limits(self, block_size):
        """
        Check block size against configured maximum values, if those can be determined
        """

        kernel_map_label = self._kernel_map.label

        total_block_size = product(block_size)
        limit = Config.get('compiler', 'cuda', 'block_size_limit')
        lastdim_limit = Config.get('compiler', 'cuda', 'block_size_lastdim_limit')
        
        if (total_block_size > limit) == True:
            raise ValueError(f'Block size for kernel "{kernel_map_label}" ({block_size}) '
                             f'is larger than the possible number of threads per block ({limit}). '
                             'The kernel will potentially not run, please reduce the thread-block size. '
                             'To increase this limit, modify the `compiler.cuda.block_size_limit` '
                             'configuration entry.')
        if (block_size[-1] > lastdim_limit) == True:
            raise ValueError(f'Last block size dimension for kernel "{kernel_map_label}" ({block_size}) '
                             'is larger than the possible number of threads in the last block dimension '
                             f'({lastdim_limit}). The kernel will potentially not run, please reduce the '
                             'thread-block size. To increase this limit, modify the '
                             '`compiler.cuda.block_size_lastdim_limit` configuration entry.')


    def _to_3d_dims(self, dim_sizes: List) -> List:
        """
        Given a list representing the size of each dimension, this function modifies
        the list in-place by collapsing all dimensions beyond the second into the
        third entry. If the list has fewer than three entries, it is padded with 1's
        to ensure it always contains exactly three elements. This is used to format
        grid and block size parameters for a kernel launch.

        Examples:
            [x]             → [x, 1, 1]
            [x, y]          → [x, y, 1]
            [x, y, z]       → [x, y, z]
            [x, y, z, u, v] → [x, y, z * u * v]
        """
        
        if len(dim_sizes) > 3:
            # multiply everything from the 3rd onward into d[2]
            dim_sizes[2] = product(dim_sizes[2:])
            dim_sizes = dim_sizes[:3]

        # pad with 1s if necessary
        dim_sizes += [1] * (3 - len(dim_sizes))

        return dim_sizes


    def _get_maps_affecting_launch_dims(self, graph: ScopeSubgraphView) -> List[Tuple[nodes.MapEntry, Dict[dace.symbol, dace.symbol]]]:
        """
        Recursively collects all GPU_Device and GPU_ThreadBlock maps within the given graph,
        including those inside nested SDFGs. For each relevant map, returns a tuple containing
        the map object and an identity mapping of its free symbols.

        Args:
            graph (ScopeSubgraphView): The subgraph to search for relevant maps.

        Returns:
            List[Tuple[nodes.MapEntry, Dict[dace.symbol, dace.symbol]]]: 
                A list of tuples, each consisting of a MapEntry object and a dictionary mapping 
                each free symbol in the map's range to itself (identity mapping).
        
        NOTE:
            Currently, dynamic parallelism (nested GPU_Device schedules) is not supported.
            The GPU_Device is only used for the top level map, where it is allowed and required.
        """

        relevant_maps = []

        for node in graph.nodes():

            # Recurse into nested SDFGs
            if isinstance(node, nodes.NestedSDFG):
                for state in node.sdfg.states():
                    relevant_maps.extend(self._get_maps_affecting_launch_dims(state))
                continue

            # MapEntry with schedule affecting launch dimensions
            if (isinstance(node, nodes.MapEntry) and
                node.schedule in {dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock}):
                identity_map = { dace.symbol(sym): dace.symbol(sym) for sym in node.map.range.free_symbols}
                relevant_maps.append((node.map, identity_map))

        return relevant_maps
    


    @property
    def kernel_name(self) -> list[str]:
        """Returns the kernel name."""
        return self._kernel_name

    @property
    def kernel_entry_node(self) -> nodes.MapEntry:
        """Returns the kernels entry node"""
        return self._kernel_entry_node
    
    @property
    def kernel_map(self) -> nodes.Map:
        """Returns the kernel map node"""
        return self._kernel_map
    
    @property
    def args_as_input(self) -> list[str]:
        """Returns the kernel function arguments
        that can be used as an input for calling the function.
        It is the __global__ kernel function, NOT the kernel launch function."""
        return self._args_as_input

    @property
    def args_typed(self) -> list[str]:
        """Returns the typed kernel function arguments
        that can be used for declaring the __global__ kernel function.
        These arguments include their respective data types."""
        return self._args_typed
    
    @property
    def kernel_wrapper_args(self) -> list[str]:
        return self._kernel_wrapper_args

    @property
    def kernel_wrapper_args_typed(self) -> list[str]:
        return self._kernel_wrapper_args_typed

    @property
    def grid_dims(self) -> list:
        """Returns the grid dimensions of the kernel."""
        return self._grid_dims

    @property
    def block_dims(self) -> list:
        """Returns the block dimensions of the kernel."""
        return self._block_dims

    @property
    def has_tbmap(self) -> bool:
        """Returns whether the kernel has a thread-block map."""
        return self._has_tbmap

    @property
    def warpSize(self) -> int:
        """
        Returns the warp size used in this kernel.
        This value depends on the selected backend (CUDA or HIP)
        and is retrieved from the configuration.
        """
        return self._warpSize

    @property
    def gpu_index_ctype(self) -> str:
        """
        Returns the C data type used for GPU indices (thread, block, warp)
        in generated code. This type is determined by the 'gpu_index_type'
        setting in the configuration and matches with a DaCe typeclass.
        """
        return self._gpu_index_ctype

