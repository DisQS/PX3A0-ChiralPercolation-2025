import os
import numpy as np
from numba import njit
import time
import pandas as pd
import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import scipy.fft as fft
import pyfftw
from scipy.special import erfinv,erf
from tqdm import tqdm
from scipy.stats import norm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
#this also measures xi and has the fixed tails

rng = np.random.default_rng()


worker_parent = None
worker_rank = None
worker_size = None
worker_left = None
worker_bottom = None
base_parent = None
base_size = None

pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = 1          # use all cores
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

def init_worker_memory(M):
    global worker_parent, worker_rank, worker_size, worker_left, worker_bottom
    global base_parent, base_size, sum_x, sum_y,sum_x2,sum_y2

    num_nodes = 2 * M * (M + 1)
    worker_parent = np.empty(num_nodes, dtype=np.int32)
    worker_rank = np.empty(num_nodes, dtype=np.int32)
    worker_size = np.empty(num_nodes, dtype=np.int32)
    worker_left = np.empty(num_nodes, dtype=np.bool_)
    worker_bottom = np.empty(num_nodes, dtype=np.bool_)

    base_parent = np.arange(num_nodes, dtype=np.int32)
    base_size = np.ones(num_nodes, dtype=np.int32)
    sum_x = np.zeros(num_nodes, dtype=np.float32)
    sum_y = np.zeros(num_nodes, dtype=np.float32)
    sum_x2 = np.zeros(num_nodes, dtype=np.float32)
    sum_y2 = np.zeros(num_nodes, dtype=np.float32)


def compute_filter(correlated,M, alpha):
    M = M * 2
    ix = np.arange(-M / 2, M / 2)
    Ix, Iy = np.meshgrid(ix, ix, indexing='ij')
    r_sq = Ix ** 2 + Iy ** 2
    C_U = (1 + r_sq) ** (-alpha)
    if not correlated:
        C_U[0::2, 1::2] *= -1
        C_U[1::2, 0::2] *= -1

    C_U_shifted = fft.ifftshift(C_U)  # zero-lag → [0,0]


    S_q = fft.rfft2(C_U_shifted)  # now real and non-negative
    amp_filter_r = np.sqrt(S_q).astype(np.float32)  # real filter

    return amp_filter_r


def compute_threshold_g(p):
    p_clamped = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.sqrt(2.0) * erfinv(2.0 * p_clamped - 1.0)



@njit(fastmath=True)
def dsu_find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

@njit(fastmath=True)
def dsu_union_gyration(parent, rank, size, sum_x, sum_y, sum_x2, sum_y2, i, j):

    root_i = dsu_find(parent, i)
    root_j = dsu_find(parent, j)
    if root_i != root_j:
        if rank[root_i] < rank[root_j]:
            parent[root_i] = root_j
            size[root_j]   += size[root_i]
            sum_x[root_j]  += sum_x[root_i]
            sum_y[root_j]  += sum_y[root_i]
            sum_x2[root_j] += sum_x2[root_i]
            sum_y2[root_j] += sum_y2[root_i]
        elif rank[root_i] > rank[root_j]:
            parent[root_j] = root_i
            size[root_i]   += size[root_j]
            sum_x[root_i]  += sum_x[root_j]
            sum_y[root_i]  += sum_y[root_j]
            sum_x2[root_i] += sum_x2[root_j]
            sum_y2[root_i] += sum_y2[root_j]
        else:
            parent[root_i] = root_j
            rank[root_j]   += 1
            size[root_j]   += size[root_i]
            sum_x[root_j]  += sum_x[root_i]
            sum_y[root_j]  += sum_y[root_i]
            sum_x2[root_j] += sum_x2[root_i]
            sum_y2[root_j] += sum_y2[root_i]

@njit(fastmath=True)
def node_to_xy(node_id, M):
    stride = 2 * M + 1
    row    = node_id // stride
    rem    = node_id  % stride
    if rem < M:

        x = np.float64(rem) + 0.5
        y = np.float64(row)
    else:
        # vertical edge
        col = rem - M - 1
        x   = np.float64(col) + 1.0
        y   = np.float64(row) + 0.5
    return x, y

@njit(fastmath=True)
def run_single_trial_jit(M, field_2d, G_thresh,
                              parent, rank, size,
                              sum_x, sum_y, sum_x2, sum_y2,
                              left_reached, bottom_reached,
                              b_parent, b_size):

    num_nodes = 2 * M * (M + 1)
    stride    = 2 * M + 1

    parent[:]       = b_parent
    size[:]         = b_size
    rank[:]         = 0
    left_reached[:] = False
    bottom_reached[:] = False


    for k in range(num_nodes):
        x, y        = node_to_xy(k, M)
        sum_x[k]    = x
        sum_y[k]    = y
        sum_x2[k]   = x * x
        sum_y2[k]   = y * y


    i = 0
    j = 0
    row_offset_1 = 0
    row_offset_2 = stride

    for _ in range(M * M):
        is_type_0 = (i % 2) == (j % 2)
        is_t      = field_2d[i, j] < G_thresh

        u_id = row_offset_1 + M + j + 1
        d_id = row_offset_1 + M + j
        l_id = row_offset_1 + j
        r_id = row_offset_2 + j

        if is_type_0:
            if is_t:
                dsu_union_gyration(parent, rank, size, sum_x, sum_y, sum_x2, sum_y2, l_id, u_id)
                dsu_union_gyration(parent, rank, size, sum_x, sum_y, sum_x2, sum_y2, r_id, d_id)
            else:
                dsu_union_gyration(parent, rank, size, sum_x, sum_y, sum_x2, sum_y2, l_id, d_id)
                dsu_union_gyration(parent, rank, size, sum_x, sum_y, sum_x2, sum_y2, r_id, u_id)
        else:
            if is_t:
                dsu_union_gyration(parent, rank, size, sum_x, sum_y, sum_x2, sum_y2, d_id, l_id)
                dsu_union_gyration(parent, rank, size, sum_x, sum_y, sum_x2, sum_y2, u_id, r_id)
            else:
                dsu_union_gyration(parent, rank, size, sum_x, sum_y, sum_x2, sum_y2, u_id, l_id)
                dsu_union_gyration(parent, rank, size, sum_x, sum_y, sum_x2, sum_y2, d_id, r_id)

        j += 1
        if j == M:
            j = 0
            i += 1
            row_offset_1 += stride
            row_offset_2 += stride

    for k in range(num_nodes):
        parent[k] = dsu_find(parent, k)

    for k in range(M):
        left_reached[parent[k]] = True

    h_perc = False
    v_perc = False

    for k in range(M):
        if left_reached[parent[M * stride + k]]:
            h_perc = True
            break

    if not h_perc:
        for k in range(M):
            bottom_reached[parent[k * stride + M]] = True
        for k in range(M):
            if bottom_reached[parent[k * stride + 2 * M]]:
                v_perc = True
                break

    max_size   = 0
    span_root  = -1
    for k in range(num_nodes):
        if parent[k] == k:
            s = size[k]
            if s > max_size:
                max_size  = s
                span_root = k


    numerator   = 0.0
    denominator = 0.0

    for k in range(num_nodes):
        if parent[k] == k and k != span_root:
            s = np.float64(size[k])
            if s < 2.0:
                continue
            cx  = sum_x[k]  / s
            cy  = sum_y[k]  / s
            Rg2 = sum_x2[k] / s + sum_y2[k] / s - cx * cx - cy * cy
            if Rg2 < 0.0:
                Rg2 = 0.0
            numerator   += s * s * Rg2
            denominator += s * s

    xi = np.sqrt(numerator / denominator) if denominator > 0.0 else 0.0

    p_sum = 1 if (h_perc or v_perc) else 0
    p_inf = np.float64(max_size) / np.float64(num_nodes)

    return h_perc, v_perc, p_sum, max_size, p_inf, xi



def worker_run_trials(M, p_t, q, trials_chunk, amp_filter,correlated=True, use_linear_mix=False):
    global worker_parent, worker_rank, worker_size, worker_left, worker_bottom
    global base_parent, base_size

    h_perc_count = 0
    v_perc_count = 0
    p_sum_count = 0
    avg_mass_sum = 0.0
    p_inf_sum = 0.0
    p_inf_sq_sum = 0.0
    xi_sum = 0.0
    xi_sq_sum = 0.0

    #compute_threshold_g(p_t)
    if use_linear_mix:
        G_thresh = p_t
    else:
        G_thresh = norm.ppf(p_t)

    max_batch = 5
    trials_done = 0

    last_batch = -1
    noise_complex = None
    field_c = None
    fft_object = None
    M2 = 2*M
    while trials_done < trials_chunk:
        current_batch = min(max_batch, trials_chunk - trials_done)
        if current_batch != last_batch:
            noise_in = pyfftw.empty_aligned((current_batch, M2, M2), dtype='float32')
            noise_out = pyfftw.empty_aligned((current_batch, M2, M2 // 2 + 1), dtype='complex64')
            field_out = pyfftw.empty_aligned((current_batch, M2, M2), dtype='float32')

            fft_plan = pyfftw.FFTW(noise_in, noise_out, axes=(1, 2), direction='FFTW_FORWARD')
            ifft_plan = pyfftw.FFTW(noise_out, field_out, axes=(1, 2), direction='FFTW_BACKWARD')
            last_batch = current_batch

        noise_in[:] = rng.standard_normal((current_batch, M2, M2), dtype=np.float32)
        fft_plan()
        noise_out *= amp_filter
        ifft_plan()
        field_c = (field_out[:, 0:M, 0:M].real).copy()


        if use_linear_mix:
            field_c = 0.5 * (1 + erf(field_c / np.sqrt(2)))
            field_w = rng.random((current_batch, M, M), dtype=np.float32)
            fields_batch = (q * field_c + (1.0 - q) * field_w)
        else:
            field_w = rng.standard_normal((current_batch, M, M), dtype=np.float32)
            mask = rng.random((current_batch,M,M), dtype=np.float32) < q
            fields_batch = np.where(mask, field_c, field_w)

        for t in range(current_batch):
            perch, percv, p_sum, avg_mass, p_inf, xi = run_single_trial_jit(
                M, fields_batch[t], G_thresh,
                worker_parent, worker_rank, worker_size,
                sum_x,sum_y,sum_x2,sum_y2,worker_left, worker_bottom, base_parent, base_size
            )


            h_perc_count += perch
            v_perc_count += percv
            p_sum_count += p_sum
            avg_mass_sum += avg_mass
            p_inf_sum += p_inf
            p_inf_sq_sum += p_inf*p_inf
            xi_sum += xi
            xi_sq_sum += xi*xi

        trials_done += current_batch



    return M, p_t, h_perc_count, v_perc_count, p_sum_count, avg_mass_sum, p_inf_sum, p_inf_sq_sum, xi_sum, xi_sq_sum


class Simulator:
    def __init__(self,correlated, system_sizes, p_vals, trials, alpha=2, q=0):
        self.system_sizes = system_sizes
        self.p_vals = p_vals
        self.trials = trials
        self.alpha = alpha
        self.q = q
        self.correlated = correlated
        self.results = []

    def run_sweep(self):
        # This detects SLURM/cgroup restricted cores correctly
        try:
            num_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            # Fallback for Windows/macOS where sched_getaffinity doesn't exist
            num_cores = os.cpu_count()
        print(f"Detected {num_cores} logical cores. Parallelizing trials...")

        for M in self.system_sizes:

            amp_filter = compute_filter(self.correlated,M, self.alpha)


            base_chunk = self.trials // num_cores
            remainder = self.trials % num_cores
            chunks = [base_chunk + (1 if i < remainder else 0) for i in range(num_cores)]
            chunks = [c for c in chunks if c > 0]

            aggregated = {}
            tasks = []

            for p in self.p_vals:
                for c in chunks:
                    tasks.append((M, p, self.q, c, amp_filter,self.correlated))

            with ProcessPoolExecutor(max_workers=num_cores, initializer=init_worker_memory, initargs=(M,)) as executor:
                futures = {executor.submit(worker_run_trials, *task): task for task in tasks}

                with tqdm(total=len(futures), desc=f"Scanning M={M}", unit="chunk") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            res_M, res_p, h_c, v_c, ps_c, am_s, pi_s,pi_s_s,xi_s,xi_s_s = future.result()

                            if res_p not in aggregated:
                                aggregated[res_p] = [0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0]

                            aggregated[res_p][0] += h_c
                            aggregated[res_p][1] += v_c
                            aggregated[res_p][2] += ps_c
                            aggregated[res_p][3] += am_s
                            aggregated[res_p][4] += pi_s
                            aggregated[res_p][5] += pi_s_s
                            aggregated[res_p][6] += xi_s
                            aggregated[res_p][7] += xi_s_s


                        except Exception as e:
                            print(f"\nWorker failed: {e}")
                        finally:
                            pbar.update(1)

            for p, metrics in aggregated.items():
                h_c, v_c, ps_c, am_s, pi_s, pi_s_s,xi_s,xi_s_s = metrics
                N = self.trials

                p_sum_mean = ps_c / N
                p_inf_mean = pi_s / N
                p_inf_sq_mean = pi_s_s / N
                xi_sq_mean = xi_s_s / N
                xi_mean = xi_s / N

                p_sum_sem = np.sqrt(p_sum_mean * (1.0 - p_sum_mean) / N)


                variance_p_inf = max(0, p_inf_sq_mean - (p_inf_mean ** 2))
                p_inf_sem = np.sqrt(variance_p_inf / (N - 1)) if N > 0 else 0.0

                variance_xi = max(0, xi_sq_mean - (xi_mean ** 2))
                xi_sem = np.sqrt(variance_xi / (N-1)) if N > 0 else 0.0

                self.results.append({
                    'M': M, 'p': p, 'alpha': self.alpha, 'q': self.q,
                    'trials': N,
                    'P_v': v_c / N,
                    'P_h': h_c / N,
                    'P_sum': p_sum_mean,
                    'P_sum_sem': p_sum_sem,
                    'Avg_mass': am_s / N,
                    'P_inf': p_inf_mean,
                    'P_inf_sem': p_inf_sem,
                    'xi': xi_mean,
                    'xi_sem' : xi_sem,
                })

        return self.results

    def save_to_csv(self):
        if not self.results:
            print("No data to save. Run run_sweep() first.")
            return

        os.makedirs("results/anti_correlated", exist_ok=True)
        os.makedirs("results/correlated", exist_ok=True)

        if self.correlated:
            filename = f"results/correlated/a={self.alpha},q={self.q}.csv"
        else:
            filename = f"results/anti_correlated/a={self.alpha},q={self.q}.csv"

        df_new = pd.DataFrame(self.results)

        if os.path.exists(filename):
            df_old = pd.read_csv(filename)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)

            def compute_weighted_stats(group):
                total_n = group['trials'].sum()
                res = group.iloc[0].copy()

                # Weighted means for all simple mean metrics
                for m in ['P_v', 'P_h', 'P_sum', 'Avg_mass', 'P_inf','xi']:
                    res[m] = (group[m] * group['trials']).sum() / total_n

                # SEM for binomial P_sum
                p_avg = res['P_sum']
                res['P_sum_sem'] = np.sqrt(p_avg * (1 - p_avg) / total_n)


                variance_i = group['P_inf_sem'] ** 2 * group['trials']
                ex2_i = variance_i + group['P_inf'] ** 2
                global_ex2 = (ex2_i * group['trials']).sum() / total_n
                global_var = (global_ex2 - res['P_inf'] ** 2) * (total_n / (total_n - 1))
                res['P_inf_sem'] = np.sqrt(global_var / total_n)

                variance_xi = group['xi_sem'] ** 2 * group['trials']
                ex2_xi = variance_xi + group['xi'] ** 2
                global_exi2 = (ex2_xi * group['trials']).sum() / total_n
                global_var_xi = (global_exi2 - res['xi'] ** 2) * (total_n / (total_n - 1))
                res['xi_sem'] = np.sqrt(global_var_xi / total_n)

                res['trials'] = total_n
                return res

            df_final = (
                df_combined
                .groupby(['M', 'p', 'alpha', 'q'], as_index=False, group_keys=False)
                .apply(compute_weighted_stats)
                .reset_index(drop=True)
            )
        else:
            df_final = df_new

        cols = ['M', 'p', 'alpha', 'q', 'trials', 'P_v', 'P_h', 'P_sum', 'P_sum_sem', 'Avg_mass', 'P_inf', 'P_inf_sem','xi','xi_sem']
        df_final.to_csv(filename, index=False, columns=cols)
        print(f"Results saved to {filename}")


def scan_bounds(correlated,M=200, alpha=2, q=0, trials=50):
    p_scan = np.linspace(0, 1, 100)
    sim = Simulator(correlated,system_sizes=[M], p_vals=p_scan, trials=trials, alpha=alpha, q=q)
    results = sim.run_sweep()
    max_xi = 0.0
    for res in results:
        max_xi = res['xi']/M if res['xi']/M > max_xi else max_xi

    print(max_xi)
    active_p_min = [res['p'] for res in results if res['xi']/M > 0.02]
    active_p_rel = [res['p'] for res in results if  res['xi']/M > max_xi*0.25]

    if not active_p_min and not active_p_rel:
        active_p = [res['p'] for res in results if res['P_sum'] > 0]

        if not active_p:
            return None

        delta = max(0.5 - min(active_p), max(active_p) - 0.5)
        return min(0.49, delta)
    if active_p_min:
        delta_min = max(0.5 - min(active_p_min), max(active_p_min) - 0.5)
    if active_p_rel:
        delta_rel = max(0.5 - min(active_p_rel), max(active_p_rel) - 0.5)
    delta = min(delta_min,delta_rel)
    return min(0.49,delta)

