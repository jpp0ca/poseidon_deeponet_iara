import matplotlib.pyplot as plt
import seaborn as sns

palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def plot_lofargram(Sxx, freq, time, ax=None, figsize=(10, 10), savepath=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()


    # Use extent to set the coordinates of the image boundaries
    # The origin='lower' argument ensures that the y-axis (time) increases from bottom to top.
    ax.imshow(Sxx, aspect='auto', origin='lower', 
              extent=[freq[0], freq[-1], time[0], time[-1]])

    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('LOFARgram')

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')
        
    return ax
    
    
def plot_tsne_embeddings(ax, embeddings_2d, targets, palette, title='t-SNE visualization of embeddings'):
    """
    Function to plot the 2D t-SNE embeddings on the provided axis.
    
    Parameters:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes.Axes): The axis on which to plot the t-SNE embeddings.
        embeddings_2d (numpy.ndarray): The 2D embeddings to plot.
        targets (numpy.ndarray): The target labels for color-coding the points.
        palette (list): A list of color codes corresponding to the classes.
    """
    # Map each target to its corresponding color
    colors = [palette[int(target)] for target in targets]
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.7, s=5)
    ax.set_title(title)
    # ax.set_xlabel('t-SNE Component 1')
    # ax.set_ylabel('t-SNE Component 2')
    class_map = ['Small', 'Medium', 'Large', 'Background']
    # Optionally, add a legend
    handles = [plt.Line2D([], [], marker='o', color=palette[i], linestyle='', label=f'Class {class_map[i]}') for i in range(len(palette))]
    return handles