import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.fft as fft
from scipy.special import erf, erfinv,gamma,kv
from scipy.stats import norm,rankdata
from scipy.special import kv, gamma as gamma_func
import pyfftw

def negate_checkerboard(arr):
    result = arr # Omit if mutating in-place is acceptable
    result[0::2, 1::2] *= -1
    result[1::2, 0::2] *= -1
    return result

def to_uniform(field):
    """Maps field values to [0,1] via rank transform, preserving correlation structure."""
    shape = field.shape
    ranks = rankdata(field.real.ravel(), method='ordinal')
    # Divide by N+1 to avoid exact 0 and 1
    uniform = ranks / (ranks.size + 1)
    return uniform.reshape(shape).astype(np.float32)


def make_field(M,alpha,correlated):
    M = M*2
    ix = np.arange(-M/2,M/2)
    Ix, Iy = np.meshgrid(ix, ix, indexing='ij')
    r_sq = Ix ** 2 + Iy ** 2
    C_U = (1 + r_sq) ** (-alpha/2)
    negate_checkerboard(C_U)
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









def radial_profile(data, center=None):
    """Computes the radial average of a 2D array."""
    y, x = np.indices((data.shape))
    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile


def compute_autocorrelation_old(field):
    """Computes the 2D two-point spatial correlation function C(r)."""
    f = np.fft.fft2(field)
    acf_2d = np.real(np.fft.ifft2(f * np.conj(f)))
    acf_2d = np.fft.fftshift(acf_2d)
    return radial_profile(acf_2d)


def compute_autocorrelation(field):
    """
    Computes the 1D radial autocorrelation profile C(r) from a 2D scalar field
    using the Wiener-Khinchin theorem and azimuthal integration.
    """
    # Isolate fluctuations
    delta_field = field - np.mean(field)

    # Compute 2D power spectral density (PSD)
    F = np.fft.fft2(delta_field)
    S = np.abs(F) ** 2

    # Inverse FFT yields the spatial autocorrelation
    C_2d = np.fft.fftshift(np.fft.ifft2(S).real)
    C_2d /= np.max(C_2d)

    # Generate coordinate grid relative to the center
    ny, nx = C_2d.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.ogrid[-cy:ny - cy, -cx:nx - cx]

    # Compute radial distance metric r = sqrt(x^2 + y^2)
    r = np.sqrt(x ** 2 + y ** 2)
    r_int = np.round(r).astype(int)

    # Perform azimuthal averaging via spatial binning
    tbin = np.bincount(r_int.ravel(), weights=C_2d.ravel())
    nr = np.bincount(r_int.ravel())

    C_1d = tbin / np.maximum(nr, 1)

    return C_1d

def get_field(M, alpha, q):
    field_c = make_field(M,alpha,True)

    field_ac = make_field(M, alpha, False)

    #field_m = make_field(M,alpha,True)
    field_m = np.random.uniform(0,1,(M,M))
    mask = np.random.uniform(0, 1, (M, M)) < q
    #final_field = field_c
    final_field = np.where(mask, field_c, field_m)

    if True:
        C_fc = compute_autocorrelation(field_c)
        C_fac = compute_autocorrelation(field_ac)
        C_fm = compute_autocorrelation(field_m)
        C_ff = compute_autocorrelation(final_field)

        '''C_fc /= C_fc.max()
        C_fac /= C_fac.max()
        C_fm /= C_fm.max()
        C_ff /= C_ff.max()'''

        fig, axs = plt.subplots(1, 4,figsize=(24, 10))
        x_vals = np.linspace(0, M)
        y_vals = (1+(x_vals**2+x_vals**2))**(-alpha/2)
        r_bins = np.arange(len(C_fc))
        mask_r = r_bins
        axs[0].plot(r_bins[mask_r], C_fc[mask_r], label="correlated_base", alpha=0.8)
        axs[0].plot(x_vals,y_vals, label="correlation funcion", alpha=0.8)
        #axs[1].plot(r_bins[mask_r], C_fac[mask_r], label="anticorrelated base", alpha=0.8)
        #axs[2].plot(r_bins[mask_r], C_fm[mask_r], label="correlated mixing", alpha=0.8)
        #axs[3].plot(r_bins[mask_r], C_ff[mask_r], label="correlated mixing", alpha=0.8)
        plt.show()

    #field_w = np.random.uniform(0, 1, (M, M))
    #final_field = q * final_field + (1 - q) * field_w
    #print(final_field.shape)
    return final_field

def generate_percolation_network(M, p_t, alpha, q):
    # Generate M*M grid of random probabilities
    probs = get_field(M, alpha, q)
    print(probs.shape)

    # Create two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # --- Plot 1: The Probability Field ---
    im = ax1.imshow(probs.T, origin='lower', extent=[0, M, 0, M], cmap='plasma')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Probability Value')
    ax1.set_title(f"Scalar Field (probs): M={M}")
    ax1.set_aspect('equal')

    # --- Plot 2: The Network ---
    # Optional: background field at low alpha for context in the network plot
    ax2.imshow(probs.T, origin='lower', extent=[0, 2*M, 0, 2*M],
               cmap='plasma', alpha=0.15, zorder=0)

    for i in range(M):
        for j in range(M):
            x, y = 2 * i + 1, 2 * j + 1
            node_type = 0 if (x % 4 == y % 4) else 1
            is_t = probs[i, j] < p_t
            color = 'green' if is_t else 'red'

            up, down = (x, y + 1), (x, y - 1)
            left, right = (x - 1, y), (x + 1, y)

            if node_type == 0:
                bonds = [(left, up), (right, down)] if is_t else [(left, down), (right, up)]
            else:
                bonds = [(down, left), (up, right)] if is_t else [(up, left), (down, right)]

            for start, end in bonds:
                ax2.annotate('', xy=(end[0], end[1]), xytext=(start[0], start[1]),
                             arrowprops=dict(arrowstyle='->', color=color, lw=1.2,
                                             shrinkA=2, shrinkB=2), zorder=2)

    ax2.set_title(f"Percolation Network: $p_t$={p_t}")
    #ax2.set_xticks(range(0, 2 * M + 1, 2))
    #ax2.set_yticks(range(0, 2 * M + 1, 2))
    ax2.grid(True, linestyle=':', alpha=0.4, zorder=1)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig("network.svg", format='svg')
    plt.show()


def graph_me(M, p_t, alpha, q):
    # Field generation (Assuming get_field is defined elsewhere)
    probs = get_field(M, alpha, q)

    # Conversion for 8cm square
    cm_to_inch = 1 / 2.54
    size_inch = 8 * cm_to_inch

    # Single plot, 8cm x 8cm
    fig, ax = plt.subplots(figsize=(size_inch, size_inch))

    for i in range(M):
        for j in range(M):
            x, y = 2 * i + 1, 2 * j + 1
            node_type = 0 if (x % 4 == y % 4) else 1
            is_t = probs[i, j] < p_t
            color = 'green' if is_t else 'red'

            up, down = (x, y + 1), (x, y - 1)
            left, right = (x - 1, y), (x + 1, y)

            if node_type == 0:
                bonds = [(left, up), (right, down)] if is_t else [(left, down), (right, up)]
            else:
                bonds = [(down, left), (up, right)] if is_t else [(up, left), (down, right)]

            for start, end in bonds:
                ax.annotate('', xy=(end[0], end[1]), xytext=(start[0], start[1]),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.0,
                                            shrinkA=0, shrinkB=0), zorder=2)

    # Explicitly set limits to match the coordinate system used (0 to 2M)
    ax.set_xlim(0, 2 * M)
    ax.set_ylim(0, 2 * M)

    # Remove all labels, ticks, and spines
    ax.set_axis_off()
    ax.set_aspect('equal')

    # Tighten layout for 8cm square export
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0)

    plt.savefig("network_fixed.png", format='png',dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()

# Execution

graph_me(M=15, p_t=0.5,alpha=1 ,q =1)
