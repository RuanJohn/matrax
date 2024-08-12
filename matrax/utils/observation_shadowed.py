# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import chex
import jax
import jax.numpy as jnp


def generate_shadowed_equilibrium_matrix(
    key_int: chex.PRNGKey, n_agents: int, n_actions: int
) -> chex.Array:
    key = jax.random.PRNGKey(key_int)
    matrix_shape = tuple([n_actions] * n_agents)

    # Step 1: Create the optimal hyperplane
    key, subkey = jax.random.split(key)
    optimal_hyperplane = jax.random.uniform(
        subkey, shape=matrix_shape[1:], minval=-30, maxval=0
    )
    optimal_hyperplane = optimal_hyperplane.at[(0,) * (n_agents - 1)].set(
        11
    )  # Set global optimum

    # Step 2: Create shadowing hyperplanes
    shadowing_hyperplanes = []
    for _ in range(1, n_actions):
        key, subkey = jax.random.split(key)
        hyperplane = jax.random.uniform(
            subkey, shape=matrix_shape[1:], minval=1, maxval=10
        )
        shadowing_hyperplanes.append(hyperplane)

    # Step 3: Combine hyperplanes
    payoff_matrix = jnp.stack([optimal_hyperplane] + shadowing_hyperplanes, axis=0)

    # Step 4: Ensure global optimum is unique maximum
    payoff_matrix = jnp.minimum(payoff_matrix, 10.99)
    payoff_matrix = payoff_matrix.at[(0,) * n_agents].set(11)

    return payoff_matrix


def verify_shadowed_equilibrium(matrix: chex.Array) -> chex.Array:
    global_optimum = matrix[tuple([0] * matrix.ndim)]

    condition1 = jnp.logical_and(
        global_optimum == 11, jnp.max(matrix) == global_optimum
    )

    optimal_hyperplane_mean = jnp.mean(
        matrix[tuple([0] + [slice(None)] * (matrix.ndim - 1))]
    )

    def check_hyperplane(i):  # type: ignore
        hyperplane_mean = jnp.mean(
            matrix[tuple([i] + [slice(None)] * (matrix.ndim - 1))]
        )
        return hyperplane_mean > optimal_hyperplane_mean

    other_hyperplanes_check = jax.lax.map(
        check_hyperplane, jnp.arange(1, matrix.shape[0])
    )
    condition2 = jnp.any(other_hyperplanes_check)

    return jnp.logical_and(condition1, condition2)


if __name__ == "__main__":
    # Example usage
    n_agents = 3
    n_actions = 3
    key_int = 42

    jitted_generate = jax.jit(
        generate_shadowed_equilibrium_matrix, static_argnums=(1, 2)
    )
    jitted_verify = jax.jit(verify_shadowed_equilibrium)

    payoff_matrix = jitted_generate(key_int, n_agents, n_actions)
    is_shadowed = jitted_verify(payoff_matrix)

    print("Payoff Matrix:")
    print(jnp.round(payoff_matrix, 2))
    print("\nIs Shadowed Equilibrium:", bool(is_shadowed))

    # Generate and verify multiple matrices
    def generate_and_verify_multiple(
        num_matrices: int, n_agents: int, n_actions: int, start_key: int
    ) -> chex.Array:
        results = []
        for i in range(num_matrices):
            matrix = jitted_generate(start_key + i, n_agents, n_actions)
            is_shadowed = jitted_verify(matrix)
            results.append(is_shadowed)
        return results

    for n in range(2, 11):
        print(f"\nTest with {n} agents:")
        multi_results = generate_and_verify_multiple(100, n, n_actions, key_int)
        print(f"All shadowed: {all(multi_results)}")
        print(f"Percentage shadowed: {sum(multi_results) / len(multi_results) * 100}%")
