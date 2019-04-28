def scatter(axes, xs, ys, xlabel, ylabel, *args, **kwargs):
    axes.scatter(xs, ys, *args, **kwargs)
    axes.set_title("{} - {}".format(ylabel, xlabel))
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
