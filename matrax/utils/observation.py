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


from typing import Tuple

import chex
import jax.numpy as jnp
from jax import lax, random

# def create_random_matrix(key_integer: int, num_agents: int, action_space: int) -> chex.Array:
#     # Define the shape of the matrix
#     shape = (action_space,) * num_agents

#     key = random.PRNGKey(key_integer)
#     random_matrix = random.uniform(key, shape=shape, minval=-1, maxval=1)

#     max_reward = 11
#     min_reward = -30

#     # Generate a random postion on the full matrix for where to place the reward

#     return random_matrix


def create_random_matrix(
    key_integrer: int, num_agents: int, action_space: int
) -> chex.Array:
    shape = (action_space,) * num_agents

    subkey1, subkey2, subkey3 = random.split(random.PRNGKey(key_integrer), 3)

    random_matrix = random.uniform(subkey1, shape=shape, minval=-1, maxval=1)

    max_reward = 11
    min_reward = -30

    # Generate random indices for each axis for max reward
    max_pos = random.randint(subkey2, (num_agents,), 0, action_space)

    # Function to generate a new min_pos
    def generate_min_pos(key: chex.PRNGKey) -> chex.Array:
        return random.randint(key, (num_agents,), 0, action_space)

    # Use lax.while_loop to ensure min_pos is different from max_pos
    def cond_fun(carry: Tuple[chex.PRNGKey, chex.Array]) -> chex.Array:
        key, min_pos = carry
        return jnp.all(max_pos == min_pos)

    def body_fun(
        carry: Tuple[chex.PRNGKey, chex.Array],
    ) -> Tuple[chex.PRNGKey, chex.Array]:
        key, _ = carry
        new_key, subkey = random.split(key)
        return (new_key, generate_min_pos(subkey))

    _, min_pos = lax.while_loop(
        cond_fun, body_fun, (subkey3, generate_min_pos(subkey3))
    )

    # Place max and min rewards in the matrix
    random_matrix = random_matrix.at[tuple(max_pos)].set(max_reward)
    random_matrix = random_matrix.at[tuple(min_pos)].set(min_reward)

    return random_matrix


if __name__ == "__main__":
    # Example usage
    key_int = 2
    num_agents = 4
    action_space = 3
    matrix = create_random_matrix(key_int, num_agents, action_space)
    print(matrix.shape)
