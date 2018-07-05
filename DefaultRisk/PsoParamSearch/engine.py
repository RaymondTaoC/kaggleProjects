from pyswarms.single import GlobalBestPSO


def run_swarm(dimensions, init_position, cost_func, options=None):
    if not options:
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    # Call instance of PSO
    optimizer = GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options, init_pos=init_position)

    # Perform optimization
    cost, pos = optimizer.optimize(cost_func, print_step=100, iters=1000, verbose=3)
    return cost, pos
