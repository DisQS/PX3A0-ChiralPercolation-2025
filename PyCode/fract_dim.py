def analyze_fractal_scaling(df, ax=None, label="", clean_style=False):
    """
    Method C: Fractal Dimension Scaling.
    Measures scaling between system size (L) and avg mass of the largest cluster at pc.
    Returns Df and its uncertainty (standard error).
    """
    if ax is None:
        return 0, 0

    pc = 0.5
    df['p_dist'] = abs(df['p'] - pc)
    # Find run closest to pc for each M
    df_pc = df.loc[df.groupby('M')['p_dist'].idxmin()].copy()

    log_L = np.log(df_pc['M'])
    log_Mass = np.log(df_pc['Avg_Mass'])

    slope_D, intercept, r_value, p_value, std_err = linregress(log_L, log_Mass)
    Df = slope_D

    # Visualization
    apply_clean_style(ax, clean_style)

    if not clean_style:
        color = 'tab:red'
        ax.set_xlabel('ln(L)')
        ax.set_ylabel('ln(Mass)', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_title(f'Fractal Scaling')

    # Points: Red crosses
    ax.plot(log_L, log_Mass, 'rx', markersize=8, markeredgewidth=2, rasterized=clean_style)

    # Line: Black line of best fit
    x_range = np.linspace(min(log_L), max(log_L), 100)
    y_fit = slope_D * x_range + intercept
    lbl_line = f'Df={Df:.2f}±{std_err:.3f}' if not clean_style else None
    ax.plot(x_range, y_fit, 'k-', linewidth=1.5, label=lbl_line, rasterized=clean_style)

    if not clean_style:
        ax.legend()

    return Df, std_err