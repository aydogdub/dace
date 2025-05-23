# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import TYPE_CHECKING
import dace
from dace import registry, symbolic, dtypes
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets.target import TargetCodeGenerator, make_absolute
from dace.codegen.targets.cpp import mangle_dace_state_struct_name
from dace.sdfg import nodes, SDFG
from dace.config import Config

from dace.codegen import cppunparse
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator


@registry.autoregister_params(name='mpi')
class MPICodeGen(TargetCodeGenerator):
    """ An MPI code generator. """
    target_name = 'mpi'
    title = 'MPI'
    language = 'cpp'

    def __init__(self, frame_codegen: 'DaCeCodeGenerator', sdfg: SDFG):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        self._global_sdfg = sdfg

        # Register dispatchers
        self._dispatcher.register_map_dispatcher(dtypes.ScheduleType.MPI, self)

    def get_generated_codeobjects(self):
        fileheader = CodeIOStream()
        sdfg = self._global_sdfg
        self._frame.generate_fileheader(sdfg, fileheader, 'mpi')

        params_comma = self._global_sdfg.init_signature(free_symbols=self._frame.free_symbols(self._global_sdfg))
        if params_comma:
            params_comma = ', ' + params_comma

        codeobj = CodeObject(
            sdfg.name + '_mpi', """
#include <dace/dace.h>
#include <mpi.h>

MPI_Comm __dace_mpi_comm;
int __dace_comm_size = 1;
int __dace_comm_rank = 0;

{file_header}

DACE_EXPORTED int __dace_init_mpi({sdfg_state_name} *__state{params});
DACE_EXPORTED int __dace_exit_mpi({sdfg_state_name} *__state);

int __dace_init_mpi({sdfg_state_name} *__state{params}) {{
    int isinit = 0;
    if (MPI_Initialized(&isinit) != MPI_SUCCESS)
        return 1;
    if (!isinit) {{
        if (MPI_Init(NULL, NULL) != MPI_SUCCESS)
            return 1;
    }}

    MPI_Comm_dup(MPI_COMM_WORLD, &__dace_mpi_comm);
    MPI_Comm_rank(__dace_mpi_comm, &__dace_comm_rank);
    MPI_Comm_size(__dace_mpi_comm, &__dace_comm_size);

    printf(\"MPI was initialized on proc %i of %i\\n\", __dace_comm_rank,
           __dace_comm_size);
    return 0;
}}

int __dace_exit_mpi({sdfg_state_name} *__state) {{
    MPI_Comm_free(&__dace_mpi_comm);
    MPI_Finalize();

    printf(\"MPI was finalized on proc %i of %i\\n\", __dace_comm_rank,
           __dace_comm_size);
    return 0;
}}
""".format(params=params_comma,
           sdfg=sdfg,
           sdfg_state_name=mangle_dace_state_struct_name(sdfg),
           file_header=fileheader.getvalue()), 'cpp', MPICodeGen, 'MPI')
        return [codeobj]

    @staticmethod
    def cmake_options():
        options = ['-DDACE_ENABLE_MPI=ON']

        if Config.get("compiler", "mpi", "executable"):
            compiler = make_absolute(Config.get("compiler", "mpi", "executable"))
            options.append("-DMPI_CXX_COMPILER=\"{}\"".format(compiler))

        return options

    @property
    def has_initializer(self):
        return True

    @property
    def has_finalizer(self):
        return True

    def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: StateSubgraphView, state_id: int,
                       function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        # Take care of map header
        assert len(dfg_scope.source_nodes()) == 1
        map_header: nodes.MapEntry = dfg_scope.source_nodes()[0]

        function_stream.write('extern int __dace_comm_size, __dace_comm_rank;', cfg, state_id, map_header)

        # Add extra opening brace (dynamic map ranges, closed in MapExit
        # generator)
        callsite_stream.write('{', cfg, state_id, map_header)

        if len(map_header.map.params) > 1:
            raise NotImplementedError('Multi-dimensional MPI maps are not supported')

        state = cfg.state(state_id)
        symtypes = map_header.new_symbols(sdfg, state, state.symbols_defined_at(map_header))

        for var, r in zip(map_header.map.params, map_header.map.range):
            begin, end, skip = r

            callsite_stream.write('{\n', cfg, state_id, map_header)
            callsite_stream.write(
                '%s %s = %s + __dace_comm_rank * (%s);\n' %
                (symtypes[var], var, cppunparse.pyexpr2cpp(symbolic.symstr(begin, cpp_mode=True)),
                 cppunparse.pyexpr2cpp(symbolic.symstr(skip, cpp_mode=True))), cfg, state_id, map_header)

        self._frame.allocate_arrays_in_scope(sdfg, cfg, map_header, function_stream, callsite_stream)

        self._dispatcher.dispatch_subgraph(sdfg,
                                           cfg,
                                           dfg_scope,
                                           state_id,
                                           function_stream,
                                           callsite_stream,
                                           skip_entry_node=True)
