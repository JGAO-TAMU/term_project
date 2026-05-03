#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-"${ROOT_DIR}/build_a100"}"
OUT_DIR="${OUT_DIR:-"${ROOT_DIR}/experiments/a100_scaling/$(date +%Y%m%d_%H%M%S)"}"
EXE="${EXE:-"${BUILD_DIR}/sim_a100"}"

TICKS="${TICKS:-96}"
TICKS_PER_DAY="${TICKS_PER_DAY:-96}"
REPEATS="${REPEATS:-3}"
SEED_BASE="${SEED_BASE:-11}"
SUITES="${SUITES:-capacity grid radius}"

CAPACITY_CASES="${CAPACITY_CASES:-10000 20000 40000 80000 120000 160000 200000}"
GRID_CASES="${GRID_CASES:-24 32 48 64 96 128}"
RADIUS_CASES="${RADIUS_CASES:-0.12 0.20 0.28 0.35}"
MULTIDAY_CAPACITY="${MULTIDAY_CAPACITY:-50000}"
MULTIDAY_TICKS="${MULTIDAY_TICKS:-960}"

DEFAULT_GRID="${DEFAULT_GRID:-64}"
DEFAULT_DETECTION_RADIUS="${DEFAULT_DETECTION_RADIUS:-0.28}"
DEFAULT_MATE_RADIUS="${DEFAULT_MATE_RADIUS:-0.075}"
DEFAULT_BASS_MATE_RADIUS="${DEFAULT_BASS_MATE_RADIUS:-0.14}"

mkdir -p "${BUILD_DIR}" "${OUT_DIR}"

if [[ "${LOAD_CUDA_MODULE:-0}" == "1" ]] && type module >/dev/null 2>&1; then
    module load CUDA/12.0
fi

choose_host_compiler() {
    if [[ -n "${HOST_COMPILER:-}" ]]; then
        printf '%s\n' "${HOST_COMPILER}"
        return
    fi
    if command -v icc >/dev/null 2>&1; then
        command -v icc
        return
    fi
    if command -v icx >/dev/null 2>&1; then
        command -v icx
        return
    fi
    if command -v gcc >/dev/null 2>&1; then
        command -v gcc
        return
    fi
    printf '\n'
}

build_sim() {
    if [[ "${SKIP_BUILD:-0}" == "1" && -x "${EXE}" ]]; then
        return
    fi

    local host_compiler
    host_compiler="$(choose_host_compiler)"

    local nvcc_flags=("-std=c++14" "-arch=sm_80" "-lineinfo")
    if [[ "${DEBUG:-0}" == "1" ]]; then
        nvcc_flags+=("-g" "-G" "-O0")
    else
        nvcc_flags+=("-O3")
    fi

    local host_args=()
    if [[ -n "${host_compiler}" ]]; then
        host_args=("-ccbin=${host_compiler}")
    fi

    nvcc "${nvcc_flags[@]}" "${host_args[@]}" \
        -I"${ROOT_DIR}" \
        -o "${EXE}" \
        "${ROOT_DIR}/main.cu" \
        "${ROOT_DIR}/simulation_app.cu" \
        "${ROOT_DIR}/kernels.cu" \
        "${ROOT_DIR}/cpu_rules.cpp" \
        "${ROOT_DIR}/output.cpp"
}

suite_enabled() {
    local suite="$1"
    [[ "${SUITES}" == "all" || " ${SUITES} " == *" ${suite} "* ]]
}

initial_bluegill() {
    local capacity="$1"
    printf '%d\n' $((capacity / 5))
}

initial_minnow() {
    local capacity="$1"
    printf '%d\n' $((capacity / 4))
}

initial_bass() {
    local capacity="$1"
    local value=$((capacity / 100))
    if (( value < 1 )); then
        value=1
    fi
    printf '%d\n' "${value}"
}

food_count() {
    local capacity="$1"
    printf '%d\n' $((capacity / 3))
}

cover_count() {
    local capacity="$1"
    printf '%d\n' $((capacity / 20))
}

write_metadata() {
    {
        printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'hostname=%s\n' "$(hostname)"
        printf 'root_dir=%s\n' "${ROOT_DIR}"
        printf 'exe=%s\n' "${EXE}"
        printf 'suites=%s\n' "${SUITES}"
        printf 'ticks=%s\n' "${TICKS}"
        printf 'ticks_per_day=%s\n' "${TICKS_PER_DAY}"
        printf 'repeats=%s\n' "${REPEATS}"
        printf 'seed_base=%s\n' "${SEED_BASE}"
        printf 'nvcc=%s\n' "$(command -v nvcc || true)"
        nvcc --version 2>/dev/null || true
        nvidia-smi 2>/dev/null || true
    } > "${OUT_DIR}/metadata.txt"
}

csv_header() {
    printf 'suite,case_id,repeat,seed,agent_capacity,bluegill_initial,minnow_initial,bass_initial,food,cover,ticks,ticks_per_day,spatial_grid,detection_radius,mate_radius,bass_mate_radius,program_ms,kernel_total_ms,kernel_avg_ms,kernel_max_ms,kernel_launches,wall_seconds,final_day,final_bluegill,final_minnow,final_bass,final_food,day_births,day_deaths,day_predation,status\n'
}

run_case() {
    local suite="$1"
    local case_id="$2"
    local repeat="$3"
    local seed="$4"
    local capacity="$5"
    local ticks="$6"
    local grid="$7"
    local detection_radius="$8"
    local mate_radius="$9"
    local bass_mate_radius="${10}"

    local bluegill minnow bass food cover
    bluegill="$(initial_bluegill "${capacity}")"
    minnow="$(initial_minnow "${capacity}")"
    bass="$(initial_bass "${capacity}")"
    food="$(food_count "${capacity}")"
    cover="$(cover_count "${capacity}")"

    local prefix="${suite}_${case_id}_r${repeat}_seed${seed}"
    local daily_csv="${OUT_DIR}/${prefix}_daily.csv"
    local summary_txt="${OUT_DIR}/${prefix}_summary.txt"
    local time_txt="${OUT_DIR}/${prefix}_time.txt"
    local status="ok"

    set +e
    /usr/bin/time -f 'wall_seconds=%e' -o "${time_txt}" \
        "${EXE}" \
        --stats \
        --ticks "${ticks}" \
        --ticks-per-day "${TICKS_PER_DAY}" \
        --seed "${seed}" \
        --agent-capacity "${capacity}" \
        --bluegill "${bluegill}" \
        --minnow "${minnow}" \
        --bass "${bass}" \
        --food "${food}" \
        --cover "${cover}" \
        --food-pattern random \
        --cover-pattern random \
        --spatial-grid "${grid}" \
        --detection-radius "${detection_radius}" \
        --mate-radius "${mate_radius}" \
        --bass-mate-radius "${bass_mate_radius}" \
        > "${daily_csv}" \
        2> "${summary_txt}"
    local exit_code=$?
    set -e

    if (( exit_code != 0 )); then
        status="failed_${exit_code}"
    fi

    local program_ms kernel_total_ms kernel_avg_ms kernel_max_ms kernel_launches wall_seconds
    program_ms="$(awk '/Program time:/ {print $3}' "${summary_txt}" | tail -n 1)"
    kernel_total_ms="$(awk '/Kernel total:/ {print $3}' "${summary_txt}" | tail -n 1)"
    kernel_avg_ms="$(awk '/Kernel total:/ {print $7}' "${summary_txt}" | tail -n 1)"
    kernel_max_ms="$(awk '/Kernel total:/ {print $11}' "${summary_txt}" | tail -n 1)"
    kernel_launches="$(awk '/Kernel total:/ {print $15}' "${summary_txt}" | tail -n 1)"
    wall_seconds="$(awk -F= '/wall_seconds=/ {print $2}' "${time_txt}" | tail -n 1)"

    local final_day final_bluegill final_minnow final_bass final_food day_births day_deaths day_predation
    IFS=',' read -r final_day final_bluegill final_minnow final_bass final_food day_births day_deaths day_predation \
        < <(tail -n 1 "${daily_csv}")

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${suite}" "${case_id}" "${repeat}" "${seed}" "${capacity}" \
        "${bluegill}" "${minnow}" "${bass}" "${food}" "${cover}" \
        "${ticks}" "${TICKS_PER_DAY}" "${grid}" "${detection_radius}" \
        "${mate_radius}" "${bass_mate_radius}" \
        "${program_ms}" "${kernel_total_ms}" "${kernel_avg_ms}" "${kernel_max_ms}" \
        "${kernel_launches}" "${wall_seconds}" "${final_day}" "${final_bluegill}" \
        "${final_minnow}" "${final_bass}" "${final_food}" "${day_births}" \
        "${day_deaths}" "${day_predation}" "${status}" \
        >> "${OUT_DIR}/results.csv"
}

run_repeated_case() {
    local suite="$1"
    local case_id="$2"
    local capacity="$3"
    local ticks="$4"
    local grid="$5"
    local detection_radius="$6"
    local mate_radius="$7"
    local bass_mate_radius="$8"

    local repeat seed
    for repeat in $(seq 1 "${REPEATS}"); do
        seed=$((SEED_BASE + repeat * 101))
        run_case "${suite}" "${case_id}" "${repeat}" "${seed}" "${capacity}" "${ticks}" \
            "${grid}" "${detection_radius}" "${mate_radius}" "${bass_mate_radius}"
    done
}

build_sim
write_metadata
csv_header > "${OUT_DIR}/results.csv"

if suite_enabled capacity; then
    for capacity in ${CAPACITY_CASES}; do
        run_repeated_case capacity "cap${capacity}" "${capacity}" "${TICKS}" \
            "${DEFAULT_GRID}" "${DEFAULT_DETECTION_RADIUS}" \
            "${DEFAULT_MATE_RADIUS}" "${DEFAULT_BASS_MATE_RADIUS}"
    done
fi

if suite_enabled grid; then
    for grid in ${GRID_CASES}; do
        run_repeated_case grid "grid${grid}" 100000 "${TICKS}" \
            "${grid}" "${DEFAULT_DETECTION_RADIUS}" \
            "${DEFAULT_MATE_RADIUS}" "${DEFAULT_BASS_MATE_RADIUS}"
    done
fi

if suite_enabled radius; then
    for radius in ${RADIUS_CASES}; do
        run_repeated_case radius "r${radius}" 100000 "${TICKS}" \
            "${DEFAULT_GRID}" "${radius}" \
            "${DEFAULT_MATE_RADIUS}" "${DEFAULT_BASS_MATE_RADIUS}"
    done
fi

if suite_enabled multiday; then
    run_repeated_case multiday "cap${MULTIDAY_CAPACITY}_${MULTIDAY_TICKS}ticks" \
        "${MULTIDAY_CAPACITY}" "${MULTIDAY_TICKS}" \
        "${DEFAULT_GRID}" "${DEFAULT_DETECTION_RADIUS}" \
        "${DEFAULT_MATE_RADIUS}" "${DEFAULT_BASS_MATE_RADIUS}"
fi

printf 'A100 scaling results written to %s\n' "${OUT_DIR}"
