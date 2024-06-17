import matplotlib.pyplot as plt

# Function for setting the default figure size
def figsize(rel_width, rel_height):
    default_width = 15 # inches

    return (rel_width * default_width, rel_height * default_width)

# Function for labelling figures with letters
def label_figure(fig, pos, label):
    if isinstance(pos, str):
        if pos == "top left":
            pos = (0,1.01)
        elif pos == "top right":
            pos = (0.98,1.01)
    
    if isinstance(pos, tuple):
        fig.text(*pos, label, size=30)


# Create the Figure layouts
paper_figs = {}

plt.rcParams["figure.max_open_warning"] = 30

# MAIN FIGURES
# Create figure 1
paper_figs["fig1"] = {"main": plt.figure(figsize=figsize(1, 0.4))}


# Create figure 2
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 0.5))
gs = fig.add_gridspec(1,2)

sfig1 = fig.add_subfigure(gs[0])
label_figure(sfig1, pos="top left", label="A")
sfig2 = fig.add_subfigure(gs[1])
label_figure(sfig2, pos="top left", label="B")

paper_figs["fig2"] = {
    "main": fig,
    "subs": {
        "A": sfig1,
        "B": sfig2,
    }
}


# Create figure 3
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 1))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 3], width_ratios=[2,1])

sfig2 = fig.add_subfigure(gs[0, :])
label_figure(sfig2, pos=(0.03,1.01), label="A")
label_figure(sfig2, pos=(0.2,1.01), label="B")
label_figure(sfig2, pos=(0.37,1.01), label="C")
label_figure(sfig2, pos=(0.56,1.01), label="D")
label_figure(sfig2, pos=(0.71,1.01), label="E")
sfig3 = fig.add_subfigure(gs[1, :-1])
label_figure(sfig3, pos=(0.03,1.01), label="F")
label_figure(sfig3, pos=(0.03,0.5), label="G")
sfig4 = fig.add_subfigure(gs[1, -1])
label_figure(sfig4, pos=(0.03,1.01), label="H")

paper_figs["fig3"] = {
    "main": fig,
    "subs": {
        "A": sfig2,
        "F": sfig3,
        "H": sfig4,
    }
}


# Create figure 4
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 1.2))
gs = fig.add_gridspec(4,1, height_ratios=[1,1,1.5,1], hspace=0.12)

sfig1 = fig.add_subfigure(gs[0])
label_figure(sfig1, pos=(0.0,0.99), label="A")
sfig2 = fig.add_subfigure(gs[1])
label_figure(sfig2, pos=(0.0,0.99), label="B")
sfig3 = fig.add_subfigure(gs[2])
label_figure(sfig3, pos=(0.0,1.01), label="C")
label_figure(sfig3, pos=(0.25,1.01), label="D")
label_figure(sfig3, pos=(0.5,1.01), label="E")
label_figure(sfig3, pos=(0.75,1.01), label="F")
label_figure(sfig3, pos=(0.0,0.5), label="G")
label_figure(sfig3, pos=(0.25,0.5), label="H")
label_figure(sfig3, pos=(0.5,0.5), label="I")
label_figure(sfig3, pos=(0.75,0.5), label="J")
sfig4 = fig.add_subfigure(gs[3])
label_figure(sfig4, pos=(0.0,0.99), label="K")
label_figure(sfig4, pos=(0.5,0.99), label="L")

paper_figs["fig4"] = {
    "main": fig,
    "subs": {
        "A": sfig1,
        "B": sfig2,
        "C": sfig3,
        "K": sfig4,
    }
}


# Create figure 5
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 0.6))
gs = fig.add_gridspec(3,3, hspace=0.12)

sfig1 = fig.add_subfigure(gs[:1, :])
label_figure(sfig1, pos=(0.0,1.01), label="A")
label_figure(sfig1, pos=(0.25,1.01), label="B")
label_figure(sfig1, pos=(0.5,1.01), label="C")
label_figure(sfig1, pos=(0.75,1.01), label="D")
sfig2 = fig.add_subfigure(gs[1:, :1])
label_figure(sfig2, pos="top left", label="E")
sfig3 = fig.add_subfigure(gs[1:, 1:])
label_figure(sfig3, pos=(0.0,1.01), label="F")
label_figure(sfig3, pos=(0.5,1.01), label="G")

paper_figs["fig5"] = {
    "main": fig,
    "subs": {
        "A": sfig1,
        "E": sfig2,
        "F": sfig3,
    }
}


# Create figure 6
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 0.6))
gs = fig.add_gridspec(2,3, hspace=0.12)

sfig1 = fig.add_subfigure(gs[:, 0])
label_figure(sfig1, pos="top left", label="A")
sfig2 = fig.add_subfigure(gs[0, 1:])
label_figure(sfig2, pos=(0.0,1.01), label="B")
label_figure(sfig2, pos=(0.5,1.01), label="C")
sfig3 = fig.add_subfigure(gs[1, 1])
label_figure(sfig3, pos="top left", label="D")
sfig4 = fig.add_subfigure(gs[1, 2])
label_figure(sfig4, pos="top left", label="E")

paper_figs["fig6"] = {
    "main": fig,
    "subs": {
        "A": sfig1,
        "B": sfig2,
        "D": sfig3,
        "E": sfig4,
    }
}


# Create figure 7
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 1))
gs = fig.add_gridspec(3,3, height_ratios=[0.5,1,1], hspace=0.12)

sfig1 = fig.add_subfigure(gs[0, :])
label_figure(sfig1, pos="top left", label="A")
sfig2 = fig.add_subfigure(gs[1, :])
label_figure(sfig2, pos=(0.0,1.01), label="B")
label_figure(sfig2, pos=(0.33,1.01), label="C")
label_figure(sfig2, pos=(0.66,1.01), label="D")
sfig3 = fig.add_subfigure(gs[2, :])
label_figure(sfig3, pos=(0.0,1.01), label="E")
label_figure(sfig3, pos=(0.5,1.01), label="F")

paper_figs["fig7"] = {
    "main": fig,
    "subs": {
        "A": sfig1,
        "B": sfig2,
        "E": sfig3,
    }
}


# Create figure 8
fig = plt.figure(figsize=figsize(1, 0.4))
label_figure(fig, pos=(0.0,0.95), label="A")
label_figure(fig, pos=(0.33,0.95), label="B")
label_figure(fig, pos=(0.66,0.95), label="C")

paper_figs["fig8"] = {"main": fig}


# SUPPLEMENTAL FIGURES
# Create figure S1
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 0.5))
gs = fig.add_gridspec(1, 2)

sfig1 = fig.add_subfigure(gs[0])
label_figure(sfig1, pos=(0.0,0.9), label="A")
sfig2 = fig.add_subfigure(gs[1])
label_figure(sfig2, pos=(0.0,0.9), label="B")

paper_figs["figS1"] = {
    "main": fig,
    "subs": {
        "A": sfig1,
        "B": sfig2,
    }
}


# Create figure S2
paper_figs["figS2"] = {"main": plt.figure(figsize=figsize(0.6, 0.5))}

# Create figure S3
paper_figs["figS3"] = {"main": plt.figure(figsize=figsize(1, 0.4))}



# Create figure S4
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 0.5))
gs = fig.add_gridspec(1, 3)

sfig1 = fig.add_subfigure(gs[0])
label_figure(sfig1, pos="top left", label="A")
sfig2 = fig.add_subfigure(gs[1])
label_figure(sfig2, pos="top left", label="B")
sfig3 = fig.add_subfigure(gs[2])
label_figure(sfig3, pos="top left", label="C")

paper_figs["figS4"] = {
    "main": fig,
    "subs": {
        "A": sfig1,
        "B": sfig2,
        "C": sfig3,
    }
}


# Create figure S5
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 0.5))
gs = fig.add_gridspec(1,2)

sfig1 = fig.add_subfigure(gs[0])
label_figure(sfig1, pos="top left", label="A")
sfig2 = fig.add_subfigure(gs[1])
label_figure(sfig2, pos="top right", label="B")

paper_figs["figS5"] = {
    "main": fig,
    "subs": {
        "A": sfig1,
        "B": sfig2,
    }
}


# Create figure S6
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 0.5))
gs = fig.add_gridspec(1, 2)

sfig1 = fig.add_subfigure(gs[0])
label_figure(sfig1, pos=(0.0,1.01), label="A")
label_figure(sfig1, pos=(0.0,0.5), label="B")
sfig2 = fig.add_subfigure(gs[1])
label_figure(sfig2, pos=(0.0,1.01), label="C")
label_figure(sfig2, pos=(0.0,0.5), label="D")

paper_figs["figS6"] = {
    "main": fig,
    "subs": {
        "A": sfig1,
        "C": sfig2,
    }
}


# Create figure S7
paper_figs["figS7"] = {"main": plt.figure(figsize=figsize(1, 0.4))}

# Create figure S8
paper_figs["figS8"] = {"main": plt.figure(figsize=figsize(1, 0.8))}


# Create figure S9
fig = plt.figure(constrained_layout=True, figsize=figsize(1, 0.5))
gs = fig.add_gridspec(1, 2)

sfig1 = fig.add_subfigure(gs[0])
label_figure(sfig1, pos="top left", label="A")
sfig2 = fig.add_subfigure(gs[1])
label_figure(sfig2, pos="top left", label="B")


paper_figs["figS9"] = {
    "main": fig,
    "subs": {
        "A": sfig1,
        "B": sfig2,
    }
}



# Create figure S10
fig = plt.figure(figsize=figsize(1, 0.6))
label_figure(fig, pos=(0.0,1.01), label="A")
label_figure(fig, pos=(0.33,1.01), label="B")
label_figure(fig, pos=(0.66,1.01), label="C")

paper_figs["figS10"] = {"main": fig}


# Create figure S11
fig = plt.figure(figsize=figsize(1, 0.6))
label_figure(fig, pos=(0.0,1.01), label="A")
label_figure(fig, pos=(0.5,1.01), label="B")

paper_figs["figS11"] = {"main": fig}


# Create figure S12
fig = plt.figure(figsize=figsize(1, 0.5))
label_figure(fig, pos=(0.0,1.01), label="A")
label_figure(fig, pos=(0.33,1.01), label="B")
label_figure(fig, pos=(0.66,1.01), label="C")

paper_figs["figS12"] = {"main": fig}



# Create figure S13
paper_figs["figS13"] = {"main": plt.figure(figsize=figsize(1, 0.4))}