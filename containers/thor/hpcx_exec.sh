#!/usr/bin/env bash
set -euo pipefail

export HPCX_DIR="${HPCX_DIR:-/opt/hpcx}"
export HPCX_MPI_DIR="${HPCX_MPI_DIR:-${HPCX_DIR}/ompi}"
export HPCX_UCC_DIR="${HPCX_UCC_DIR:-${HPCX_DIR}/ucc}"
export HPCX_UCX_DIR="${HPCX_UCX_DIR:-${HPCX_DIR}/ucx}"

for required_dir in \
    "${HPCX_MPI_DIR}/lib" \
    "${HPCX_UCC_DIR}/lib" \
    "${HPCX_UCX_DIR}/lib"
do
    if [[ ! -d "${required_dir}" ]]; then
        echo "Missing NVIDIA HPC-X component directory: ${required_dir}" >&2
        exit 1
    fi
done

# The 26.07 ARM64 PyTorch image contains the HPC-X component tree but omits
# hpcx-init.sh. Reproduce the version-matched library order set by hpcx_load;
# loading only UCX mixes HPC-X with Ubuntu OpenMPI and breaks torch imports.
hpcx_ld_paths=(
    "${HPCX_DIR}/nccl_rdma_sharp_plugin/lib"
    "${HPCX_UCC_DIR}/lib/ucc"
    "${HPCX_UCC_DIR}/lib"
    "${HPCX_UCX_DIR}/lib/ucx"
    "${HPCX_UCX_DIR}/lib"
    "${HPCX_DIR}/sharp/lib"
    "${HPCX_DIR}/hcoll/lib"
    "${HPCX_MPI_DIR}/lib"
)
hpcx_bin_paths=(
    "${HPCX_DIR}/sharp/bin"
    "${HPCX_DIR}/hcoll/bin"
    "${HPCX_UCC_DIR}/bin"
    "${HPCX_UCX_DIR}/bin"
    "${HPCX_MPI_DIR}/bin"
)

prepend_existing_paths()
{
    local variable_name="$1"
    shift
    local combined=""
    local candidate
    for candidate in "$@"; do
        if [[ -d "${candidate}" ]]; then
            combined+="${combined:+:}${candidate}"
        fi
    done
    local existing="${!variable_name:-}"
    printf -v "${variable_name}" '%s' "${combined}${existing:+:${existing}}"
    export "${variable_name}"
}

prepend_existing_paths LD_LIBRARY_PATH "${hpcx_ld_paths[@]}"
prepend_existing_paths LIBRARY_PATH \
    "${HPCX_MPI_DIR}/lib" "${HPCX_UCC_DIR}/lib" "${HPCX_UCX_DIR}/lib"
prepend_existing_paths PATH "${hpcx_bin_paths[@]}"
prepend_existing_paths CPATH \
    "${HPCX_MPI_DIR}/include" "${HPCX_UCC_DIR}/include" "${HPCX_UCX_DIR}/include"
prepend_existing_paths PKG_CONFIG_PATH \
    "${HPCX_MPI_DIR}/lib/pkgconfig" "${HPCX_UCX_DIR}/lib/pkgconfig"

export MPI_HOME="${HPCX_MPI_DIR}"
export OMPI_HOME="${HPCX_MPI_DIR}"
export OPAL_PREFIX="${HPCX_MPI_DIR}"
export OSHMEM_HOME="${HPCX_MPI_DIR}"

if [[ $# -eq 0 ]]; then
    exec bash
fi
exec "$@"
