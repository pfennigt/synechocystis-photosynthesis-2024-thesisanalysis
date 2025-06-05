import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

figsize_default = 7 # inches
element_scaling = 0.5


# Function for setting the default figure size
def figsize(rel_width, rel_height):
    return (rel_width * figsize_default, rel_height * figsize_default)


# Function for labelling figures with letters
def label_figure(fig, pos, label, size=13):
    if isinstance(pos, str):
        if pos == "top left":
            pos = (0, 1.01)
        elif pos == "top right":
            pos = (0.98, 1.01)
    
    if isinstance(pos, tuple):
        fig.text(*pos, label, size=size, weight="bold")

def label_axis(ax, pos, label, size=13, ha=None, va="top"):
    if isinstance(pos, str):
        if pos == "top left":
            pos = (0.05, 0.95)
            ha="left"
        elif pos == "top right":
            pos = (0.95, 0.95)
            ha="right"
    
    if isinstance(pos, tuple):
        ax.text(*pos, label, size=size, transform=ax.transAxes, ha=ha, va=va, weight="bold")

def label_axes(axes, pos, start_label_int=0, size=13, ha=None):
    for i,ax in enumerate(axes.flatten()):
        label_axis(ax, pos, chr(i+65+start_label_int), size=13, ha=ha)

# Get a factor of the original rcParam as value
def rcfac(rckwarg: str, factor: float):
    old = plt.rcParams.get(rckwarg)
    if isinstance(old, int):
        return int(np.round(old * factor))
    elif isinstance(old, float):
        return old * factor
    elif old is None:
        raise ValueError(f"not an rcParam: {rckwarg}")
    else:
        raise ValueError(f"unsupported type of rcParam: {type(old)}")

# Set the default parameters for plots

# Generally halve the size and width of fonts, lines, etc.
# plt.rcParams.update(
#     {k:v*element_scaling for k,v in plt.rcParams.items() if (k.endswith("size") or k.endswith("width") or k.endswith("pad")) and isinstance(v, (int, float))}
# )

plt.rcParams["figure.max_open_warning"] = 30
plt.rcParams['figure.dpi'] = 150  # 600 DPI

# Set the dpi for exporting
export_dpi = 300
export_formats = ["png", "pdf", "svg"]

# Set the default line colors

cmaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']

# 4
default_cmap_index=4
default_cmap = plt.get_cmap(cmaps[4])
default_cmap_colors = default_cmap.colors
plt.rcParams['axes.prop_cycle'] = cycler('color', default_cmap_colors)


# Define the figures
thesis_figures = {}

# Create figure 1
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 1))
gs = fig.add_gridspec(2,1, hspace=0.1, height_ratios=(1,0.9))

sfig1 = fig.add_subfigure(gs[0])
label_figure(sfig1, pos=(0.1, 0.87), label="A")
label_figure(sfig1, pos=(0.54, 0.87), label="B")
sfig2 = fig.add_subfigure(gs[1])
label_figure(sfig2, pos=(0.1, 0.85), label="C")
label_figure(sfig2, pos=(0.54, 0.85), label="D")

thesis_figures["CBBcontrol"] = {
    "main": fig,
    "A": sfig1,
    "C": sfig2,
}

# Create figure 2
thesis_figures["mca_wldifference"] = {"main": plt.figure(figsize=figsize(1, 1))}

# Create figure 3
thesis_figures["spectra_monochrom"] = {"main": plt.figure(figsize=figsize(1, 0.3))}

# Create figure 4
thesis_figures["CO2fix_spectra"] = {"main": plt.figure(figsize=figsize(0.5, 1))}

# Create figure 5
thesis_figures["spectra_complex"] = {"main": plt.figure(figsize=figsize(1, 0.25))}

# Create figure 6
thesis_figures["stmodels_analysis"] = {"main": plt.figure(figsize=figsize(1, 0.5))}

# Create figure 7
thesis_figures["stmodels_analysis_PQvsFm"] = {"main": plt.figure(figsize=figsize(1, 1))}

# Create figure 8
thesis_figures["stmodels_analysis_multilight"] = {"main": plt.figure(figsize=figsize(1, 1))}

# Create figure 9
thesis_figures["CO2fix_mono"] = {"main": plt.figure(figsize=figsize(1, 1))}

# Create figure 10
thesis_figures["Biotech_prod"] = {"main": plt.figure(figsize=figsize(1, 1.5))}

# Create figure 11
thesis_figures["PAM_singles"] = {"main": plt.figure(figsize=figsize(1, 1))}

# Create figure 12
thesis_figures["PAM_adapted"] = {"main": plt.figure(figsize=figsize(1, 1))}

# Create figure 13
thesis_figures["PAM_validation"] = {"main": plt.figure(figsize=figsize(1, 1))}

# Create figure 14
thesis_figures["efluxes_main"] = {"main": plt.figure(figsize=figsize(1, 1))}

# Create figure 15
thesis_figures["efluxes_analysis_add"] = {"main": plt.figure(figsize=figsize(1.5, 1))}

# Create figure 16
thesis_figures["efluxes_analysis"] = {"main": plt.figure(figsize=figsize(1, 1))}

# Create figure 17
thesis_figures["O2fluxes"] = {"main": plt.figure(figsize=figsize(1, 0.9))}

# Create figure 18
thesis_figures["Benschop"] = {"main": plt.figure(figsize=figsize(1, 0.9))}

# Create figure 19
thesis_figures["Biotech_Rodrigures"] = {"main": plt.figure(figsize=figsize(1, 1))}

# Create figure 20
thesis_figures["Overexpression"] = {"main": plt.figure(figsize=figsize(1, 0.8))}

# Create figure 21
thesis_figures["Overexpression_mca"] = {"main": plt.figure(figsize=figsize(1, 1))}

# Create figure 22
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 1.5))
gs = fig.add_gridspec(3,1, hspace=0.1)

sfig1 = fig.add_subfigure(gs[0])
label_figure(sfig1, pos=(0.1, 0.87), label="A")
sfig2 = fig.add_subfigure(gs[1])
label_figure(sfig2, pos=(0.1, 0.85), label="B")
sfig3 = fig.add_subfigure(gs[2])
label_figure(sfig3, pos=(0.1, 0.85), label="C")

thesis_figures["Montecarlo_main"] = {
    "main": fig,
    "A": sfig1,
    "B": sfig2,
    "C": sfig3,
}

# Create figure 23
thesis_figures["Montecarlo_supp"] = {"main": plt.figure(figsize=figsize(1, 1.5))}

# Create figure 24
thesis_figures["SPint_experiment"]= {"main": plt.figure(figsize=figsize(1, 0.6))}

# Create figure 25
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 2))
gs = fig.add_gridspec(4,1, hspace=0.1, height_ratios=(1,1,1,0.2))

sfig1 = fig.add_subfigure(gs[0])
label_figure(sfig1, pos=(0.1, 0.95), label="A")
sfig2 = fig.add_subfigure(gs[1])
label_figure(sfig2, pos=(0.13, 1), label="B")
sfig3 = fig.add_subfigure(gs[2])
label_figure(sfig3, pos=(0.13, 1.02), label="C")
sfig4 = fig.add_subfigure(gs[3])
label_figure(sfig4, pos=(0.13, 1.02), label="D")

thesis_figures["Minimize_comparison"] = {
    "main": fig,
    "A": sfig1,
    "B": sfig2,
    "C": sfig3,
    "D": sfig4,
}

# Create figure 26
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 1.5))
gs = fig.add_gridspec(3,1, hspace=0.1)

sfig1 = fig.add_subfigure(gs[0])
label_figure(sfig1, pos=(0.1, 0.87), label="A")
sfig2 = fig.add_subfigure(gs[1])
label_figure(sfig2, pos=(0.1, 0.85), label="B")
sfig3 = fig.add_subfigure(gs[2])
label_figure(sfig3, pos=(0.1, 0.85), label="C")

thesis_figures["Montecarlo_main_allpar"] = {
    "main": fig,
    "A": sfig1,
    "B": sfig2,
    "C": sfig3,
}

# Create figure 27
thesis_figures["Montecarlo_supp_allpar"] = {"main": plt.figure(figsize=figsize(1, 1.5))}

# # Create figure 22
# fig = plt.figure(constrained_layout=True, figsize=figsize(1, 1.5))
# gs = fig.add_gridspec(3,1, hspace=0.1)

# sfig1 = fig.add_subfigure(gs[0])
# label_figure(sfig1, pos=(0.1, 0.95), label="A")
# sfig2 = fig.add_subfigure(gs[1])
# label_figure(sfig2, pos=(0.1, 0.95), label="B")
# sfig3 = fig.add_subfigure(gs[2])
# label_figure(sfig3, pos=(0.1, 0.95), label="C")

# thesis_figures["Montecarlo_supp"] = {
#     "main": fig,
#     "A": sfig1,
#     "B": sfig2,
#     "C": sfig3,
# }

# # Create figure 3
# fig = plt.figure(constrained_layout=True, figsize=figsize(1, 1.2))
# gs = fig.add_gridspec(2,1, height_ratios=[1,0.3])

# sfig1 = fig.add_subfigure(gs[0])
# sfig2 = fig.add_subfigure(gs[1])
# label_figure(sfig2, pos=(0.1, 0.95), label="I")

# thesis_figures["LightDark_WT_AL"] = {
#     "main": fig,
#     "A": sfig1,
#     "I": sfig2,
# }

# # Create figure 4
# thesis_figures["LightDark_WT_AL_parameters"] = {"main": plt.figure(figsize=figsize(1, 0.8))}


# # Create figure 5
# fig = plt.figure(constrained_layout=True, figsize=figsize(1, 1.2))
# gs = fig.add_gridspec(2,1, height_ratios=[1,0.3])

# sfig1 = fig.add_subfigure(gs[0])
# sfig2 = fig.add_subfigure(gs[1])
# label_figure(sfig2, pos=(0.1, 0.95), label="I")

# thesis_figures["LightDark_WT_ALBLFR"] = {
#     "main": fig,
#     "A": sfig1,
#     "I": sfig2,
# }

# # Create figure 6
# thesis_figures["LightDark_WT_ALBLFR_parameters"] = {"main": plt.figure(figsize=figsize(1, 0.5))}

# # Create figure 7
# fig = plt.figure(constrained_layout=True, figsize=figsize(1, 1.2))
# gs = fig.add_gridspec(2,1, height_ratios=[1,0.3])

# sfig1 = fig.add_subfigure(gs[0])
# sfig2 = fig.add_subfigure(gs[1])
# label_figure(sfig2, pos=(0.1, 0.95), label="I")

# thesis_figures["LightDark_allGeno_AL"] = {
#     "main": fig,
#     "A": sfig1,
#     "I": sfig2,
# }

# # Create figure 8
# thesis_figures["LightDark_allGeno_AL_parameters"] = {"main": plt.figure(figsize=figsize(1, 0.8))}

# # Create figure 9
# thesis_figures["lightDarkSimulation"] = {"main": plt.figure(figsize=figsize(1, 1.2))}

# # Create figure 10
# thesis_figures["DIRKSimulation"] = {"main": plt.figure(figsize=figsize(1, 0.7))}

# # Create figure 11
# thesis_figures["DIRKAnalysis"] = {"main": plt.figure(figsize=figsize(0.7, 0.7))}