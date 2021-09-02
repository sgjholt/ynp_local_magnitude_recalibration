import matplotlib
params = {
    # 'text.usetex': True,
    'text.latex.preamble': '\\usepackage{gensymb}',
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 20, # fontsize for x and y labels (was 10)
    'axes.titlesize': 12,
    'font.size': 15, # was 10
    'legend.fontsize': 12, # was 10
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'font.family': 'serif',
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'mathtext.fontset': 'stix',
}
matplotlib.rcParams.update(params)
