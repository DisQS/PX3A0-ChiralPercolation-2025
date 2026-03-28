import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.fft as fft
from scipy.special import erf, erfinv,gamma,kv
from scipy.stats import norm,pearsonr,rankdata,spearmanr
from scipy.special import kv, gamma as gamma_func
import pyfftw
from tqdm import tqdm
import scipy.stats
import scipy.spatial.distance
from scipy.signal import correlate
def negate_checkerboard(arr):
    result = arr # Omit if mutating in-place is acceptable
    result[0::2, 1::2] *= -1
    result[1::2, 0::2] *= -1
    return result

def make_field_corr(M,alpha):
    M = M * 2
    ix = np.arange(-M / 2, M / 2)
    Ix, Iy = np.meshgrid(ix, ix)
    r_sq = Ix ** 2 + Iy ** 2
    C_U = np.zeros_like(r_sq)
    C_U[r_sq > 0 ] = (np.sqrt(r_sq[r_sq>0])) ** (-alpha)
    C_U[r_sq == 0] = 0.0
    noise_f = np.random.standard_normal((M, M))
    field_c = correlate(C_U,noise_f,mode='same')
    field_c = field_c[0:int(M / 2), 0:int(M / 2)].copy()
    field_c = (field_c - np.mean(field_c)) / np.std(field_c)
    field_c = 0.5 * (1 + erf(field_c / np.sqrt(2)))

    return field_c


def make_field_current(M,alpha):
    M = M*2
    ix = np.arange(-M/2,M/2)
    Ix, Iy = np.meshgrid(ix, ix, indexing='ij')
    r_sq = Ix ** 2 + Iy ** 2
    C_U = (1 + r_sq) ** (-alpha/2)
    #negate_checkerboard(C_U)
    C_U_shifted = fft.ifftshift(C_U)
    S_q = fft.fft2(C_U_shifted)
    amplitude = np.sqrt(np.abs(S_q))

    # can return amplitude here

    dist = M // 2
    noise_f = fft.fft2(np.random.standard_normal((M, M)))
    field_c = fft.ifft2(noise_f * amplitude).real
    field_c = field_c[0:dist, 0:dist].copy()
    #map this guy to uniform
    field_c = 0.5 * (1 + erf(field_c / np.sqrt(2)))
    return field_c


def make_field_power_thing(M,alpha):
    L = 2*M
    kx = np.fft.fftfreq(L) * L  # integer wavenumbers: 0, 1, ..., L/2, -L/2+1, ..., -1
    ky = np.fft.fftfreq(L) * L
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.hypot(KX, KY)

    S = np.where(k_mag > 0, np.abs(k_mag) ** (alpha - 2.0), 0.0)
    #noise = (rng.standard_normal((L, L)) + 1j * rng.standard_normal((L, L)))
    noise = fft.fft2(np.random.standard_normal((L, L)))
    field_k = noise * np.sqrt(S)
    field_k = np.fft.ifftshift(np.fft.ifftshift(field_k))
    field = np.real(np.fft.ifft2(field_k))[0:M,0:M].real
    field = field /  field.std()
    field = 0.5 * (1 + erf(field/ np.sqrt(2)))
    return field



pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = -1          # use one cores
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
rng = np.random.default_rng()


def make_field(M, alpha):
    'ppyfftw'
    global noise_in
    global noise_out
    global field_out
    global batch_size

    M_ext = M * 2
    noise_in[:] = rng.standard_normal((batch_size,M_ext, M_ext),dtype = np.float32)
    fft_plan()
    noise_out *= amp_filter_r
    ifft_plan()
    field_c = field_out[:,0:M, 0:M].copy()

    field_c = 0.5 * (1 + erf(field_c / np.sqrt(2)))

    return field_c

def make_field_me_2(M,alpha):
    'me but trying c~r^-alpha'
    M = M*2
    ix = np.arange(-M/2,M/2)
    Ix, Iy = np.meshgrid(ix, ix, indexing='ij')
    r_sq = Ix ** 2 + Iy ** 2
    r = np.sqrt(r_sq)
    C = np.ones_like(r)
    C[r>0] = r[r>0] ** (-alpha)
    noise = np.random.standard_normal((M, M))
    noise_f = fft.fft2(noise)

    S_q = fft.fft2(C)

    field_c = fft.ifft2(noise_f * np.sqrt(S_q))
    dist = M // 2
    field_c = field_c[0:dist, 0:dist].copy()
    return field_c


def make_field_uni(M, alpha):
    'hopefully uniform'
    N = M * 2
    ix = np.arange(-N / 2, N / 2)
    Ix, Iy = np.meshgrid(ix, ix, indexing='ij')


    r_sq = Ix ** 2 + Iy ** 2
    C_U = (1 + r_sq) ** (-alpha / 2)

    # 2. Pre-distort to find the required Gaussian correlation C_Z(r)
    C_Z = 2 * np.sin((np.pi / 6) * C_U)


    S_q = fft.fft2(C_Z)


    S_q = np.abs(np.real(S_q))
    amplitude = np.sqrt(S_q)

    # 4. Synthesize the complex Gaussian field
    noise = np.random.standard_normal((N, N))
    noise_f = fft.fft2(noise)
    field_c = fft.ifft2(noise_f * amplitude)

    # 5. Extract the real field and crop to eliminate periodic boundary artifacts
    field_real = np.real(field_c)
    field_crop = field_real[0:M, 0:M]

    # 6. Standardize strictly to N(0,1) to satisfy CDF domain requirements
    #field_norm = (field_crop - np.mean(field_crop)) / np.std(field_crop)

    # 7. Map to uniform distribution U(0,1) via the standard normal CDF
    field_uniform = 0.5 * (1 + erf(field_crop / np.sqrt(2)))

    return field_uniform

def make_field_old(M,alpha):
    'current model'
    M = 2*M
    kx = fft.fft(np.arange(-M/2,M/2))

    Kx, Ky = np.meshgrid(kx, kx, indexing='ij')


    K_mag_sq = Kx ** 2 + Ky ** 2

    with np.errstate(divide='ignore', invalid='ignore'):
        K_mag = np.sqrt(K_mag_sq)
        amplitude = (K_mag_sq) ** -((alpha)/4)

    amplitude[K_mag_sq == 0] = 0.0

    noise_shape = (M,M)
    noise_complex = np.random.random(noise_shape)
    noise_f = fft.fft2(noise_complex)
    field_c = fft.ifft2(noise_f * amplitude)
    dist = M // 2
    field_c = field_c[0:dist, 0:dist].copy().real
    #field_uniform = 0.5 * (1 + erf(field_c / np.sqrt(2))).real
    return field_c


def make_field_analytic(N, alpha):
    """Analytical FFM based on Makse et al. (1996)"""
    rng = np.random
    d = 2

    # 1. Correct beta_d definition: (gamma - d) / 2
    beta_d = (alpha - d) / 2.0

    # 2. Generate Gaussian noise in the expanded domain
    u = rng.normal(0, 1, (N * 2, N * 2))
    u_q = fft.fft2(u)

    # 3. Build the exact frequency grid
    freq_y = fft.fftfreq(N * 2) * 2 * np.pi
    freq_x = fft.fftfreq(N * 2) * 2 * np.pi
    qy, qx = np.meshgrid(freq_x, freq_y, indexing='ij')
    q = np.sqrt(qx ** 2 + qy ** 2)

    # 4. Calculate Analytical Spectral Density S(q)
    # Using d=2 in the prefactor 2 * pi^(d/2)
    prefactor = (2 * np.pi) / gamma(beta_d + 1)

    S_q = np.zeros_like(q)
    mask = q > 0  # Avoid singularity at q=0

    # S(q) = prefactor * (q/2)^beta * K_beta(q)
    S_q[mask] = prefactor * ((q[mask] / 2) ** beta_d) * kv(beta_d, q[mask])

    # Set q=0 to 0 to ensure the field has zero global mean
    S_q[0, 0] = 0.0

    # 5. Filter the noise and invert
    # CRITICAL: s=(N*2, N*2) must match the original expanded domain
    eta = fft.ifft2(np.sqrt(np.maximum(S_q, 0.0)) * u_q, s=(N * 2, N * 2))

    # 6. Extract the target N x N region
    eta = eta[0:N, 0:N].copy()

    # 7. Standardize the field (CRITICAL for norm.ppf thresholding)
    eta = (eta - np.mean(eta)) / np.std(eta)
    eta = 0.5 * (1 + erf(eta / np.sqrt(2))).real
    return eta.real
def make_field_atest(N, alpha):
    """analytical"""
    rng = np.random
    beta_d = (alpha - 2) / 2.0
    u = rng.normal(0, 1, (N*2, N*2))
    u_q = np.fft.rfftn(u)
    freq_y = np.fft.fftfreq(N*2) * 2 * np.pi
    freq_x = np.fft.rfftfreq(N*2) * 2 * np.pi
    qy, qx = np.meshgrid(freq_x, freq_y)


    q = np.sqrt(qx ** 2 + qy ** 2)


    prefactor = (2 * np.pi) / gamma_func(beta_d + 1)
    S_q = np.zeros_like(q)
    mask = q > 0
    S_q[mask] = prefactor * (q[mask] / 2) ** beta_d * kv(beta_d, q[mask])
    S_q[~mask] = np.nanmean(S_q[mask])

    eta = np.fft.irfftn(np.sqrt(np.maximum(S_q, 0.0)) * u_q, s=(N, N))

    eta = eta[0:N, 0:N].copy()
    eta = 0.5 * (1 + erf(eta / np.sqrt(2))).real
    return eta

def compute_filter(M, alpha):
    """Precomputes the amplitude filter for rfft2 (Power-law C(r) ~ r^-alpha)."""
    M = int(M / 2)
    ky = fft.rfftfreq(M)
    kx = fft.fftfreq(M)
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

    K_mag_sq = Kx ** 2 + Ky ** 2

    with np.errstate(divide='ignore', invalid='ignore'):
        amplitude = K_mag_sq ** -((alpha - 2.0) / 4.0)

    amplitude[0, 0] = 0.0
    return amplitude.astype(np.float32)





def generate_correlated_field(M, alpha, pr=1.0, use_linear_mix=False, precomputed_filter=None, batch_size=1):
    # 1. Generate Raw Correlated Field (Gaussian)
    # -------------------------------------------
    if precomputed_filter is not None:
        amp = precomputed_filter
    else:
        amp = compute_filter(M, alpha)

    # Ensure amp is float32
    amp = amp.astype(np.float32)
    M = int(M / 2)
    noise_shape = (batch_size, M, M // 2 + 1)

    # complex64 is enough if we use float32
    noise_complex = (np.random.standard_normal(noise_shape).astype(np.float32) +
                     1j * np.random.standard_normal(noise_shape).astype(np.float32))

    # irfft2 over last two axes
    field_c = fft.irfft2(noise_complex * amp, s=(M, M), workers=-1)

    # Standardize Field C to N(0, 1) per sample
    # mean/std along (1, 2)
    means = np.mean(field_c, axis=(1, 2), keepdims=True)
    stds = np.std(field_c, axis=(1, 2), keepdims=True)

    # Avoid div by zero
    stds[stds == 0] = 1.0



    # 2. Generate Raw White Noise Field (Gaussian)
    # --------------------------------------------
    field_w = np.random.standard_normal((batch_size, M, M)).astype(np.float32)
    # Standardize
    w_means = np.mean(field_w, axis=(1, 2), keepdims=True)
    w_stds = np.std(field_w, axis=(1, 2), keepdims=True)
    w_stds[w_stds == 0] = 1.0
    field_w = (field_w - w_means) / w_stds

    # 3. Combine Fields
    # -----------------
    if use_linear_mix:
        # METHOD A: Linear Superposition
        final_field = (pr * field_c) + ((1.0 - pr) * field_w)

        # Re-standardize
        f_means = np.mean(final_field, axis=(1, 2), keepdims=True)
        f_stds = np.std(final_field, axis=(1, 2), keepdims=True)
        f_stds[f_stds == 0] = 1.0
        final_field = (final_field - f_means) / f_stds

    else:
        # METHOD B: Spatial Masking
        mask_probs = np.random.rand(batch_size, M, M).astype(np.float32)
        mask = mask_probs < pr
        final_field = np.where(mask, field_c, field_w)

        # Re-standardize

    final_field = np.repeat(np.repeat(final_field, 2, axis=1), 2, axis=-1)
    # 4. Return Gaussian (No erf)
    return final_field.astype(np.float32)[0]


def manual_correlation_sim(M,alpha,trials = 100):
    #if alpha > 0.6:
    #    alpha = invert(alpha,popt)
    global batch_size
    C_diag = np.zeros(M)
    diag_elm = np.zeros((trials,M))
    row_elm = np.zeros((trials, M))
    r_count  = 0
    r_max_a = 0
    r_max = 0
    r_min = 0
    r_min_a = 0
    thresh = norm.ppf(0.5)
    thresh = 0.5
    with tqdm(total=trials, desc="running trials", unit="trials") as pbar:
        for t in range(0,int(trials/batch_size)):
            field =make_field(M, alpha)
            #field = np.random.random((batch_size,M,M))

            r_count += (field > thresh).sum()
            if field.max() > r_max:
                r_max = field.max()
            if field.min() < r_min:
                r_min = field.min()

            for i in range(0,batch_size):
                r_max_a += field.max()
                r_min_a += field.min()

                diag_elm[t*batch_size+i,:]= np.diagonal(field[i])
                row_elm[t*batch_size+i,:] = field[i][0]
                pbar.update(1)

    for i in range(0,M):
        #C_diag[i] = pearsonr(diag_elm[:, 0], diag_elm[:, i])[0]
        C_diag[i] = spearmanr(row_elm[:, 0], row_elm[:, i],alternative = 'less')[0]
    r_count /= (M**2)*trials
    r_max_a /=  trials
    r_min_a /= trials
    print(r_max_a,r_min_a,'avg')
    print(r_max,r_min, 'extr')
    return C_diag, r_count

def invert(y_target,popt):
    """
    Analytically invert the 4-parameter logistic correction formula.
    Returns the input alpha corresponding to the target measured alpha.
    """

    a, b, c, d = popt
    # Constrain y_target to the open interval bounded by the asymptotes (a, a+d)
    # to maintain a strictly positive base for the fractional exponentiation.
    eps = 1e-12
    lower_bound = min(a, a + d) + eps
    upper_bound = max(a, a + d) - eps

    # np.clip handles both scalar and array inputs
    y_safe = np.clip(y_target, lower_bound, upper_bound)

    base = (d / (y_safe - a)) - 1.0

    # Catch floating point edge cases near the asymptote
    base = np.maximum(base, 0.0)

    return (base / b) ** (1.0 / c)


M = 100
M2 = M*2
alpha =0.1

popt = [0.05555555,  0.71304513, - 2.36285607,  1.49229705]
true_alpha = invert(alpha,popt)




global noise_in
global noise_out
global field_out
global batch_size
batch_size = 10



#probs  = make_field(M, true_alpha)
#probs = np.random.random((batch_size, M, M))[0]


'''plt.imshow(probs, cmap='plasma')
plt.colorbar(label='Probability Value')
plt.show()

C_r,rden = manual_correlation_sim(M,alpha,trials = 2000)
x_vals = np.arange(M)


y_vals = (1+x_vals**2+x_vals**2)**(-alpha/2)

#y_vals[1::2] *= -1


#corr_line[x_vals%2!=0] =-(1+odd_vals**2+ odd_vals**2)**(-alpha/2)

#y_vals = (x_vals)**-(float(alpha))
#y_vals[0] = 1.0


plt.loglog(x_vals,y_vals)

plt.loglog(x_vals,C_r)
plt.show()
print(rden)'''



M_ext = M * 2
ix = np.arange(-M_ext / 2, M_ext / 2)
Ix, Iy = np.meshgrid(ix, ix, indexing='ij')
r_sq = Ix ** 2 + Iy ** 2



cm_to_inch = 1 / 2.54
width_inch = 8 * cm_to_inch
height_inch = 8 * cm_to_inch

# Global font settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 12
})


M = 100  # Adjust based on your simulation scale
trials = 50000
alphas = [0.5,1,1.5]
colors = ['r', 'g', 'b']
markers = ['o', 's', '^']

# Log-spaced sampling
log_indices = np.unique(np.geomspace(1, M - 1, num=50, dtype=int))

fig, ax = plt.subplots(figsize=(width_inch, height_inch))






for alpha, color, marker in zip(alphas, colors, markers):

    # Run simulation
    C_U = (1 + r_sq) ** (-alpha / 2)
    negate_checkerboard(C_U)
    C_U_shifted = fft.ifftshift(C_U)  # zero-lag → [0,0]
    S_q = fft.rfft2(C_U_shifted)  # now real and non-negative
    global amp_filter_r
    amp_filter_r = np.sqrt(S_q.real).astype(np.float32)

    noise_in = pyfftw.empty_aligned((batch_size, M2, M2), dtype='float32')
    noise_out = pyfftw.empty_aligned((batch_size, M2, M2 // 2 + 1), dtype='complex64')
    field_out = pyfftw.empty_aligned((batch_size, M2, M2), dtype='float32')

    # Build plans once
    fft_plan = pyfftw.FFTW(noise_in, noise_out, axes=(1, 2), direction='FFTW_FORWARD')
    ifft_plan = pyfftw.FFTW(noise_out, field_out, axes=(1, 2), direction='FFTW_BACKWARD')

    C_r, rden = manual_correlation_sim(M, alpha, trials=trials)

    x_full = np.arange(M)
    # Analytic expression
    y_analytic_full = (1 + 2 * x_full ** 2) ** (-alpha / 2)

    # 1. Plot Analytic (Dashed Line) - use full range for smoothness
    plt.loglog(x_full[1:], y_analytic_full[1:], linestyle='--', color=color,
               alpha=0.7, label=rf'Analytic $\alpha={alpha}$')

    # 2. Sample log-spaced points from simulation
    x_sampled = x_full[log_indices]
    C_sampled = C_r[log_indices]
    y_analytic_sampled = y_analytic_full[log_indices]

    # 3. Filter points: Only plot if within 50% relative error of the analytic line
    # (Adjust tolerance as needed for your specific noise floor)
    tolerance = 0.8
    mask = np.abs(C_sampled - y_analytic_sampled) < (tolerance * y_analytic_sampled)

    plt.loglog(x_sampled, C_sampled, marker=marker, linestyle='None',
               color=color, markersize=5, label=rf'Sim $\alpha={alpha}$ (Filtered)')

ax.set_xlabel('$r$')
ax.set_ylabel('$C(r)$')
ax.grid(False)

ax.margins(0)

# 3. Use constrained_layout if subplots are used, or tight_layout with 0 pad
plt.tight_layout(pad=0)

# 4. The "Nuclear Option" for the right and top white space:
# This manually tells the subplots to take up 100% of the figure area.
fig.subplots_adjust(top=1, right=1, left=0.12, bottom=0.12)

# 5. Save with 0 pad_inches and 'tight' bbox
plt.savefig("correlation.png", format='png', dpi=600, bbox_inches='tight')
plt.show()
