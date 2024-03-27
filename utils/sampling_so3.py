import numpy as np
  
  
def generate_queries(self, number_queries=2**12, mode='random'):
    """Generate query rotations from SO(3).

    Args:
        number_queries: The number of queries.
        mode: 'random' or 'grid'; determines whether to generate rotations from
        the uniform distribution over SO(3), or use an equivolumetric grid.

    Returns:
        A tensor of rotation matrices, shape [num_queries, 3, 3].
    """
    if mode == 'random':
        return generate_queries_random(number_queries)
    elif mode == 'healpix':
        return get_closest_available_grid(number_queries)

def generate_random_rotation_matrices(number_matrices):
    rotation_matrices = torch.zeros(number_matrices, 3, 3)

    for i in range(number_matrices):
        # Generate a random rotation matrix
        random_matrix = torch.rand(3, 3)
        # Use orthogonalization to ensure it's a valid rotation matrix
        q, _ = torch.qr(random_matrix)
        rotation_matrices[i] = q

    return rotation_matrices

def get_closest_available_grid(number_queries):
    pass

