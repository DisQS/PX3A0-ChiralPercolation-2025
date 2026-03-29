import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
import cupy as cp
import cupyx.scipy.special



cu_source = r'''
extern "C" {

__device__ void uf_union(int* parent, int u, int v, int max_nodes) {
    int max_iters = max_nodes;
    int iter = 0;
    while (iter++ < max_iters) {
        int root_u = u;
        int inner = 0;
        while (root_u != parent[root_u] && inner++ < max_iters) root_u = parent[root_u];

        int root_v = v;
        inner = 0;
        while (root_v != parent[root_v] && inner++ < max_iters) root_v = parent[root_v];

        if (root_u == root_v) return;

        if (root_u < root_v) {
            int old = atomicCAS(&parent[root_v], root_v, root_u);
            if (old == root_v) return;
        } else {
            int old = atomicCAS(&parent[root_u], root_u, root_v);
            if (old == root_u) return;
        }
    }
}

__global__ void build_and_hook(const float* field, float G_thresh, int* parent, int M, int num_nodes) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= M || j >= M) return;

    bool is_type_0 = (i % 2) == (j % 2);
    bool is_t = field[i * M + j] < G_thresh;

    int row_offset_1 = i * (2 * M + 1);
    int row_offset_2 = (i + 1) * (2 * M + 1);

    int u_id = row_offset_1 + M + j + 1;
    int d_id = row_offset_1 + M + j;
    int l_id = row_offset_1 + j;
    int r_id = row_offset_2 + j;

    if (u_id >= num_nodes || d_id >= num_nodes || l_id >= num_nodes || r_id >= num_nodes) return;

    if (is_type_0) {
        if (is_t) {
            uf_union(parent, l_id, u_id, num_nodes);
            uf_union(parent, r_id, d_id, num_nodes);
        } else {
            uf_union(parent, l_id, d_id, num_nodes);
            uf_union(parent, r_id, u_id, num_nodes);
        }
    } else {
        if (is_t) {
            uf_union(parent, d_id, l_id, num_nodes);
            uf_union(parent, u_id, r_id, num_nodes);
        } else {
            uf_union(parent, u_id, l_id, num_nodes);
            uf_union(parent, d_id, r_id, num_nodes);
        }
    }
}

// FIX: jump_pointers is called twice in the pipeline to guarantee full path
// compression to roots. A single pass only jumps one level; two passes give
// O(log* N) depth guarantee needed for propagate_tags to work in one scatter.
__global__ void jump_pointers(int* parent, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_nodes) {
        int p = parent[tid];
        int max_iters = num_nodes;
        int iter = 0;
        while (p != parent[p] && iter++ < max_iters) {
            p = parent[p];
        }
        parent[tid] = p;
    }
}

__global__ void init_boundaries(int* tags, int M, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M) {
        int row_offset_1 = tid * (2 * M + 1);
        if (row_offset_1 < num_nodes) atomicOr(&tags[row_offset_1], 1);

        int row_offset_2 = (tid + 1) * (2 * M + 1) + M - 1;
        if (row_offset_2 < num_nodes) atomicOr(&tags[row_offset_2], 2);

        int top_node = M + tid + 1;
        if (top_node < num_nodes) atomicOr(&tags[top_node], 4);

        int bottom_node = (M - 1) * (2 * M + 1) + M + tid;
        if (bottom_node < num_nodes) atomicOr(&tags[bottom_node], 8);
    }
}

// OPT: propagate_tags and check_percolation fused into one kernel.
// This saves one full kernel launch + one pass over num_nodes per trial.
// Safe because check_percolation was a pure OR-reduction over the tag array,
// and the atomicOr into the root's tag is visible to the same thread
// immediately after via a second load.
// DEPENDENCY: must run after jump_pointers has fully flattened the tree,
// otherwise the single-scatter is incomplete (roots are only one hop away).
__global__ void propagate_tags_and_check(const int* parent, int* tags,
                                          int num_nodes, int* out_flags) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    int my_tag = tags[tid];
    if (my_tag > 0) {
        int root = parent[tid];
        if (root != tid && root >= 0 && root < num_nodes) {
            atomicOr(&tags[root], my_tag);
        }
    }

    // Every thread also checks its own tag (post-propagation value).
    // Non-roots contribute their pre-propagation tag, roots get their
    // fully merged value. The OR-reduction into out_flags is idempotent
    // so double-counting is harmless.
    int tag = tags[tid];
    if ((tag & 3)  == 3)  atomicOr(&out_flags[0], 1);
    if ((tag & 12) == 12) atomicOr(&out_flags[1], 1);
}

__global__ void compute_cluster_props(
        const int* parent, int* counts,
        unsigned long long* sum_x,  unsigned long long* sum_y,
        unsigned long long* sum_x2, unsigned long long* sum_y2,
        int M, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_nodes) {
        int root = parent[tid];
        if (root >= 0 && root < num_nodes) {
            atomicAdd(&counts[root], 1);

            // Coordinates scaled by 2 so all values are exact integers,
            // preventing float32 absorption errors for large M.
            int stride = 2 * M + 1;
            int row = tid / stride;
            int rem = tid % stride;
            unsigned long long x2, y2;

            if (rem < M) {
                x2 = (unsigned long long)(rem * 2 + 1);
                y2 = (unsigned long long)(row * 2);
            } else {
                int col = rem - M - 1;
                x2 = (unsigned long long)(col * 2 + 2);
                y2 = (unsigned long long)(row * 2 + 1);
            }

            atomicAdd(&sum_x[root],  x2);
            atomicAdd(&sum_y[root],  y2);
            atomicAdd(&sum_x2[root], x2 * x2);
            atomicAdd(&sum_y2[root], y2 * y2);
        }
    }
}

__global__ void compute_xi_components(
        const int* counts, const int* tags,
        const unsigned long long* sum_x,  const unsigned long long* sum_y,
        const unsigned long long* sum_x2, const unsigned long long* sum_y2,
        double* xi_comps, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    double s = (double)counts[tid];
    if (s < 2.0) return;

    int tag = tags[tid];
    // Exclude percolating clusters by topology (tag bits), not by mass.
    if (((tag & 3) == 3) || ((tag & 12) == 12)) return;

    // Unpack ×2 integer scale back to native coordinate values.
    double cx  = ((double)sum_x[tid]  / 2.0) / s;
    double cy  = ((double)sum_y[tid]  / 2.0) / s;
    double sx2 =  (double)sum_x2[tid] / 4.0;
    double sy2 =  (double)sum_y2[tid] / 4.0;

    double Rg2 = sx2 / s + sy2 / s - cx * cx - cy * cy;

    // FIX: clamp Rg2 to zero rather than gating both atomicAdds on Rg2 > 0.
    // The old code dropped the s^2 denominator contribution for compact
    // clusters, systematically overestimating xi. The denominator must always
    // be accumulated regardless of whether Rg2 happens to round negative.
    if (Rg2 < 0.0) Rg2 = 0.0;
    atomicAdd(&xi_comps[0], s * s * Rg2);
    atomicAdd(&xi_comps[1], s * s);
}

} // extern "C"
'''

module = cp.RawModule(code=cu_source, options=('-std=c++11',))
build_and_hook_kernel             = module.get_function('build_and_hook')
jump_pointers_kernel              = module.get_function('jump_pointers')
init_boundaries_kernel            = module.get_function('init_boundaries')
propagate_tags_and_check_kernel   = module.get_function('propagate_tags_and_check')
compute_cluster_props_kernel      = module.get_function('compute_cluster_props')
compute_xi_components_kernel      = module.get_function('compute_xi_components')


def invert(y_target):
    """
    Analytically invert the 4-parameter logistic correction formula.
    Returns the input alpha corresponding to the target measured alpha.
    """
    popt = [0.04575988, 0.68692398, -2.23607366, 1.50915253]
    a, b, c, d = popt
    eps = 1e-12
    lower_bound = min(a, a + d) + eps
    upper_bound = max(a, a + d) - eps
    y_safe = np.clip(y_target, lower_bound, upper_bound)
    base = (d / (y_safe - a)) - 1.0
    base = np.maximum(base, 0.0)
    return (base / b) ** (1.0 / c)


class Simulator:
    def __init__(self, correlated, system_sizes, p_vals, trials, alpha=2.0, q=0.0):
        self.system_sizes = system_sizes
        self.p_vals       = p_vals
        self.trials       = trials
        self.alpha        = alpha
        self.q            = q
        self.correlated   = correlated
        self.results      = []

    def compute_filter_cp(self, M):
        N = 2 * M
        idx = cp.arange(N)
        ei = cp.minimum(idx, N - idx)
        Ei, Ej = cp.meshgrid(ei, ei, indexing='ij')
        r_sq = Ei ** 2 + Ej ** 2
        #true_alpha = invert(self.alpha)
        C_U = (1 + r_sq) ** (-self.alpha / 2)

        if not self.correlated:
            C_U[0::2, 1::2] *= -1
            C_U[1::2, 0::2] *= -1

        S_q = cp.fft.rfft2(C_U)
        amp_filter = cp.sqrt(cp.maximum(S_q.real, 0.0))
        return amp_filter

    def run_sweep(self):
        for M in self.system_sizes:
            amp_filter = self.compute_filter_cp(M)
            M2        = 2 * M
            num_nodes = 2 * M * (M + 1)

            base_parent = cp.arange(num_nodes, dtype=cp.int32)
            parent      = cp.empty(num_nodes, dtype=cp.int32)
            tags        = cp.empty(num_nodes, dtype=cp.int32)
            counts      = cp.empty(num_nodes, dtype=cp.int32)
            out_flags   = cp.zeros(2, dtype=cp.int32)


            moments_buf = cp.empty(4 * num_nodes, dtype=cp.uint64)
            sum_x  = moments_buf[0 * num_nodes : 1 * num_nodes]
            sum_y  = moments_buf[1 * num_nodes : 2 * num_nodes]
            sum_x2 = moments_buf[2 * num_nodes : 3 * num_nodes]
            sum_y2 = moments_buf[3 * num_nodes : 4 * num_nodes]

            xi_comps = cp.zeros(2, dtype=cp.float64)

            blocks_x     = math.ceil(M / 16)
            blocks_y     = math.ceil(M / 16)
            blocks_nodes = math.ceil(num_nodes / 256)
            blocks_M     = math.ceil(M / 256)

            with tqdm(total=self.trials * len(self.p_vals),
                      desc=f"M={M}", leave=True) as pbar:

                for p in self.p_vals:
                    h_perc_count = 0
                    v_perc_count = 0
                    p_sum_count  = 0
                    avg_mass_sum = 0.0
                    p_inf_sum    = 0.0
                    p_inf_sq_sum = 0.0
                    xi_sum       = 0.0
                    xi_sq_sum    = 0.0
                    thresh       = norm.ppf(p)

                    for _ in range(self.trials):

                        noise     = cp.random.standard_normal((M2, M2), dtype=cp.float32)
                        noise_fft = cp.fft.rfft2(noise)
                        noise_fft *= amp_filter

                        field_c = cp.fft.irfft2(noise_fft, s=(M2, M2))[0:M, 0:M]

                        field_w     = cp.random.standard_normal((M, M), dtype=cp.float32)
                        mask        = cp.random.random((M, M), dtype=cp.float32) < self.q

                        field_batch = cp.ascontiguousarray(cp.where(mask, field_c, field_w))


                        parent[:] = base_parent
                        tags.fill(0)
                        counts.fill(0)
                        out_flags.fill(0)

                        moments_buf.fill(0)
                        xi_comps.fill(0)

                        # ---- CCL pipeline ----
                        init_boundaries_kernel(
                            (blocks_M,), (256,),
                            (tags, np.int32(M), np.int32(num_nodes)))

                        build_and_hook_kernel(
                            (blocks_x, blocks_y), (16, 16),
                            (field_batch, cp.float32(thresh), parent,
                             np.int32(M), np.int32(num_nodes)))

                        jump_pointers_kernel(
                            (blocks_nodes,), (256,),
                            (parent, np.int32(num_nodes)))

                        propagate_tags_and_check_kernel(
                            (blocks_nodes,), (256,),
                            (parent, tags, np.int32(num_nodes), out_flags))

                        compute_cluster_props_kernel(
                            (blocks_nodes,), (256,),
                            (parent, counts, sum_x, sum_y, sum_x2, sum_y2,
                             np.int32(M), np.int32(num_nodes)))

                        compute_xi_components_kernel(
                            (blocks_nodes,), (256,),
                            (counts, tags, sum_x, sum_y, sum_x2, sum_y2,
                             xi_comps, np.int32(num_nodes)))

                        # ---- read results ----
                        flags_host = out_flags.get()
                        xi_host    = xi_comps.get()

                        h_perc = bool(flags_host[0])
                        v_perc = bool(flags_host[1])

                        num_xi = xi_host[0]
                        den_xi = xi_host[1]
                        xi     = np.sqrt(num_xi / den_xi) if den_xi > 0.0 else 0.0

                        h_perc_count += int(h_perc)
                        v_perc_count += int(v_perc)
                        p_sum_count  += 1 if (h_perc or v_perc) else 0

                        max_size      = int(counts.max())
                        avg_mass_sum += max_size
                        p_inf         = max_size / num_nodes
                        p_inf_sum    += p_inf
                        p_inf_sq_sum += p_inf * p_inf
                        xi_sum       += xi
                        xi_sq_sum    += xi * xi

                        pbar.update(1)

                    # ---- aggregate over trials ----
                    N             = self.trials
                    p_sum_mean    = p_sum_count  / N
                    p_inf_mean    = p_inf_sum    / N
                    p_inf_sq_mean = p_inf_sq_sum / N
                    xi_mean       = xi_sum       / N
                    xi_sq_mean    = xi_sq_sum    / N

                    p_sum_sem   = np.sqrt(p_sum_mean * (1.0 - p_sum_mean) / N)
                    variance_p_inf = max(0, p_inf_sq_mean - p_inf_mean ** 2)
                    p_inf_sem   = np.sqrt(variance_p_inf  / (N - 1)) if N > 1 else 0.0
                    variance_xi = max(0, xi_sq_mean - xi_mean ** 2)
                    xi_sem      = np.sqrt(variance_xi     / (N - 1)) if N > 1 else 0.0

                    self.results.append({
                        'M':         M,
                        'p':         p,
                        'alpha':     self.alpha,
                        'q':         self.q,
                        'trials':    N,
                        'P_v':       v_perc_count / N,
                        'P_h':       h_perc_count / N,
                        'P_sum':     p_sum_mean,
                        'P_sum_sem': p_sum_sem,
                        'Avg_mass':  avg_mass_sum / N,
                        'P_inf':     p_inf_mean,
                        'P_inf_sem': p_inf_sem,
                        'xi':        xi_mean,
                        'xi_sem':    xi_sem,
                    })

        return self.results

    def save_to_csv(self):
        if not self.results:
            return

        os.makedirs("results/anti_correlated", exist_ok=True)
        os.makedirs("results/correlated",      exist_ok=True)

        folder   = "correlated" if self.correlated else "anti_correlated"
        filename = f"results/{folder}/a={self.alpha},q={self.q}.csv"

        df_new = pd.DataFrame(self.results)

        if os.path.exists(filename):
            df_old      = pd.read_csv(filename)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)

            def compute_weighted_stats(group):
                total_n = group['trials'].sum()
                res     = group.iloc[0].copy()

                for m in ['P_v', 'P_h', 'P_sum', 'Avg_mass', 'P_inf', 'xi']:
                    res[m] = (group[m] * group['trials']).sum() / total_n

                p_avg            = res['P_sum']
                res['P_sum_sem'] = np.sqrt(p_avg * (1 - p_avg) / total_n)

                variance_p_inf_i   = group['P_inf_sem'] ** 2 * group['trials']
                ex2_p_inf_i        = variance_p_inf_i + group['P_inf'] ** 2
                global_ex2_p_inf   = (ex2_p_inf_i * group['trials']).sum() / total_n
                global_var_p_inf   = (global_ex2_p_inf - res['P_inf'] ** 2) * (total_n / (total_n - 1))
                res['P_inf_sem']   = np.sqrt(max(0, global_var_p_inf) / total_n)

                variance_xi_i  = group['xi_sem'] ** 2 * group['trials']
                ex2_xi_i       = variance_xi_i + group['xi'] ** 2
                global_ex2_xi  = (ex2_xi_i * group['trials']).sum() / total_n
                global_var_xi  = (global_ex2_xi - res['xi'] ** 2) * (total_n / (total_n - 1))
                res['xi_sem']  = np.sqrt(max(0, global_var_xi) / total_n)

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

        cols = [
            'M', 'p', 'alpha', 'q', 'trials',
            'P_v', 'P_h', 'P_sum', 'P_sum_sem',
            'Avg_mass', 'P_inf', 'P_inf_sem',
            'xi', 'xi_sem',
        ]
        df_final.to_csv(filename, index=False, columns=cols)


def scan_bounds(correlated, M=200, alpha=2, q=0, trials=50):
    p_scan  = np.linspace(0, 1, 100)
    sim     = Simulator(correlated, system_sizes=[M], p_vals=p_scan,
                        trials=trials, alpha=alpha, q=q)
    results = sim.run_sweep()

    max_xi = 0.0
    for res in results:
        max_xi = res['xi'] / M if res['xi'] / M > max_xi else max_xi

    print(max_xi)
    active_p_min = [res['p'] for res in results if res['xi'] / M > 0.02]
    active_p_rel = [res['p'] for res in results if res['xi'] / M > max_xi * 0.25]

    if not active_p_min and not active_p_rel:
        active_p = [res['p'] for res in results if res['P_sum'] > 0]
        if not active_p:
            return None
        delta = max(0.5 - min(active_p), max(active_p) - 0.5)
        return min(0.49, delta)

    delta_min = max(0.5 - min(active_p_min), max(active_p_min) - 0.5)
    delta_rel = max(0.5 - min(active_p_rel), max(active_p_rel) - 0.5)
    delta     = min(delta_min, delta_rel)
    return min(0.49, delta)


if __name__ == '__main__':
    scan_bounds(True, M=200, trials=50)