"""
Tests for Gymnasium environments with strategy testing.

This module contains strategy tests for various Gymnasium environments.
Each test function focuses on testing different strategies against a specific environment.
"""

import pytest
import time
import random
from envs.gym_env.client import GymAction, GymEnvironment


try:
    # Ensure gymnasium is available; skip the whole module if it's missing.
    import importlib

    if importlib.util.find_spec("gymnasium") is None:
        raise ModuleNotFoundError
except ModuleNotFoundError:
    pytest.skip("gymnasium not installed", allow_module_level=True)

CARTPOLE_ENV_ID = "CartPole-v1"
MOUNTAINCAR_ENV_ID = "MountainCarContinuous-v0"

@pytest.fixture(name="cartpole_env")
def fixture_cartpole_env():
    from envs.gym_env.server.gymnasium_environment import GymnasiumEnvironment
    env = GymnasiumEnvironment(env_id=CARTPOLE_ENV_ID, seed=42, render_mode="rgb_array")
    yield env
    env.close()


@pytest.fixture(name="mountaincar_env")
def fixture_mountaincar_env():
    from envs.gym_env.server.gymnasium_environment import GymnasiumEnvironment
    env = GymnasiumEnvironment(env_id=MOUNTAINCAR_ENV_ID, seed=42, render_mode="rgb_array")
    yield env
    env.close()


def decent_cartpole_strategy(state):
    """
    A decent (but not optimal) CartPole control strategy.

    This strategy uses a simple heuristic based on the pole angle and angular velocity:
    - Pole angle (θ): current tilt of the pole
    - Angular velocity (ω): rate of change of pole angle

    The strategy computes a "score" = θ + 0.1 * ω
    - If score > 0, push cart to the right (action=1) to counter clockwise rotation
    - If score ≤ 0, push cart to the left (action=0) to counter clockwise rotation

    This is a reactive strategy that responds to the current state but doesn't
    account for cart position or velocity, so it's not optimal but works reasonably well.

    Args:
        state: List of 4 floats [x, dx, angle, dangle]
            - x: cart position (ignored in this simple strategy)
            - dx: cart velocity (ignored in this simple strategy)
            - angle: pole angle (radians)
            - dangle: pole angular velocity

    Returns:
        action: 0 (push left) or 1 (push right)
    """
    # state: [x, dx, angle, dangle]
    x, dx, ang, dang = state
    score = ang + 0.1 * dang

    # Push right if pole is tilting/increasing clockwise (negative anguar accel)
    # Push left if pole is tilting/increasing counter-clockwise (positive angular accel)
    return 1 if score > 0.0 else 0


def baseline_bad_cartpole_strategy(state):
    """
    A deliberately bad strategy that always pushes left.

    Args:
        state: Current state (ignored)

    Returns:
        action: Always 0 (push left)
    """
    return 0


def test_cartpole_reset_and_step(cartpole_env):
    """Test that CartPole environment can reset and step with basic actions."""
    obs = cartpole_env.reset()
    state = cartpole_env.state

    assert state.env_id == CARTPOLE_ENV_ID
    assert state.step_count == 0
    assert isinstance(obs.state, list)
    assert len(obs.state) == 4  # CartPole has 4 state variables

    # Test a step with discrete action
    next_obs = cartpole_env.step(GymAction(action=0))  # Left action
    assert cartpole_env.state.step_count == 1
    assert isinstance(next_obs.state, list)
    assert next_obs.reward is not None
    assert isinstance(next_obs.done, bool)


def test_cartpole_strategy_comparison(cartpole_env):
    """
    Test that the decent cartpole strategy outperforms the baseline bad strategy.

    The decent strategy should maintain balance significantly longer than
    always pushing left (baseline bad strategy).
    """
    num_episodes = 3  # Reduced for testing

    # Run decent strategy
    smart_results = _test_strategy_against_cartpole(
        cartpole_env, decent_cartpole_strategy, num_episodes=num_episodes, use_random_seeds=False
    )

    # Run baseline bad strategy
    bad_results = _test_strategy_against_cartpole(
        cartpole_env, baseline_bad_cartpole_strategy, num_episodes=num_episodes, use_random_seeds=False
    )

    # Assertions
    assert smart_results['avg_length'] > bad_results['avg_length'], \
        f"Smart strategy ({smart_results['avg_length']:.1f}) should beat bad strategy ({bad_results['avg_length']:.1f})"

    # Smart strategy should be significantly better (at least 2x better)
    improvement_ratio = smart_results['avg_length'] / max(bad_results['avg_length'], 1)
    assert improvement_ratio > 2.0, f"Improvement ratio {improvement_ratio:.1f} should be > 2.0"

    # Smart strategy should maintain balance for at least 100 steps on average
    assert smart_results['avg_length'] > 100, \
        f"Smart strategy should maintain balance > 100 steps ({smart_results['avg_length']:.1f})"

    # Action distribution should be balanced for smart strategy
    actions = smart_results['action_distribution']
    total_actions = actions['left'] + actions['right']
    left_ratio = actions['left'] / total_actions
    assert 0.4 <= left_ratio <= 0.6, f"Smart strategy should have balanced actions (~50%), got {left_ratio:.1%}"





def decent_mountaincar_strategy(state):
    """
    A decent but not perfect strategy for MountainCarContinuous.

    Uses a simple heuristic: if velocity is positive, apply positive force,
    if velocity is negative, apply negative force. This helps maintain momentum
    but doesn't optimally build potential energy.

    Args:
        state: List of 2 floats [position, velocity]

    Returns:
        engine_force: float between -1.0 and 1.0
    """
    position, velocity = state

    # Simple momentum-based strategy
    if velocity > 0:
        return 1.0   # Keep going right
    else:
        return -1.0  # Go left to build momentum


def baseline_bad_mountaincar_strategy(state):
    """
    A deliberately bad strategy that always applies maximum right force.

    This will likely get stuck in the right valley and never reach the goal.

    Args:
        state: Current state (ignored)

    Returns:
        engine_force: Always 1.0 (full right force)
    """
    return 1.0


def test_mountaincar_reset_and_step(mountaincar_env):
    """Test that MountainCarContinuous environment can reset and step with basic actions."""
    obs = mountaincar_env.reset()
    state = mountaincar_env.state

    assert state.env_id == MOUNTAINCAR_ENV_ID
    assert state.step_count == 0
    assert isinstance(obs.state, list)
    assert len(obs.state) == 2  # MountainCar has 2 state variables

    # Test a step with continuous action
    next_obs = mountaincar_env.step(GymAction(action=[0.5]))  # Half throttle right
    assert mountaincar_env.state.step_count == 1
    assert isinstance(next_obs.state, list)
    assert next_obs.reward is not None
    assert isinstance(next_obs.done, bool)


def test_mountaincar_strategy_comparison(mountaincar_env):
    """
    Test that different MountainCarContinuous strategies perform differently.
    """
    num_episodes = 2  # Reduced for testing

    strategies = [
        (decent_mountaincar_strategy, "Decent"),
        (baseline_bad_mountaincar_strategy, "Baseline Bad")
    ]

    all_results = []

    for strategy_func, strategy_name in strategies:
        results = _test_mountaincar_strategy(
            mountaincar_env, strategy_func, strategy_name, num_episodes=num_episodes, use_random_seeds=False
        )
        all_results.append(results)

    # Basic assertion: decent strategy should do better than baseline bad strategy
    decent_result = next(r for r in all_results if r['strategy_name'] == 'Decent')
    bad_result = next(r for r in all_results if r['strategy_name'] == 'Baseline Bad')

    assert decent_result['success_rate'] >= bad_result['success_rate'], \
        f"Decent strategy success rate ({decent_result['success_rate']:.1%}) should beat baseline bad strategy ({bad_result['success_rate']:.1%})"

    # Decent strategy should reach higher positions on average
    assert decent_result['avg_final_position'] >= bad_result['avg_final_position'], \
        f"Decent strategy final position ({decent_result['avg_final_position']:.3f}) should beat baseline bad strategy ({bad_result['avg_final_position']:.3f})"


def _test_mountaincar_strategy(client, strategy_func, strategy_name, num_episodes=5, max_steps_per_episode=200, use_random_seeds=True):
    """
    Test a single MountainCar strategy over multiple episodes (silent version for pytest).

    Args:
        client: GymEnvironment client
        strategy_func: Function that takes state and returns engine_force (-1.0 to 1.0)
        strategy_name: Name of the strategy for display
        num_episodes: Number of episodes to test
        max_steps_per_episode: Maximum steps per episode
        use_random_seeds: Whether to use random seeds for each episode

    Returns:
        dict: Performance metrics
    """
    import numpy as np

    episode_lengths = []
    episode_rewards = []
    final_positions = []
    success_count = 0

    for _ in range(num_episodes):
        # Reset environment
        obs = client.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps_per_episode:
            # Get current state
            current_state = obs.state

            # Apply strategy to get engine force
            engine_force = strategy_func(current_state)
            engine_force = np.clip(engine_force, -1.0, 1.0)  # Ensure valid range

            # Take action (need to use list for continuous action)
            obs = client.step(GymAction(action=[engine_force]))

            episode_reward += obs.reward or 0
            episode_length += 1
            done = obs.done

        # Record episode results
        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)
        final_position = obs.state[0]
        final_positions.append(final_position)

        # Check if episode was successful (reached goal position >= 0.45)
        if final_position >= 0.45:
            success_count += 1

    # Calculate performance metrics
    avg_length = sum(episode_lengths) / len(episode_lengths)
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_final_position = sum(final_positions) / len(final_positions)
    success_rate = success_count / num_episodes

    return {
        'strategy_name': strategy_name,
        'avg_length': avg_length,
        'avg_reward': avg_reward,
        'avg_final_position': avg_final_position,
        'success_rate': success_rate,
        'success_count': success_count
    }


def _test_strategy_against_cartpole(client: GymEnvironment, strategy_func, num_episodes: int = 10, use_random_seeds: bool = True):
    """
    Test a single CartPole strategy over multiple episodes (silent version for pytest).

    Args:
        client: GymEnvironment client
        strategy_func: Function that takes state and returns action (0 or 1)
        num_episodes: Number of episodes to test
        use_random_seeds: Whether to use random seeds for each episode

    Returns:
        dict: Performance metrics
    """
    episode_lengths = []
    episode_rewards = []
    actions_taken = []

    for _ in range(num_episodes):
        # Reset environment
        obs = client.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < 1000:  # Prevent infinite loops in testing
            # Get current state
            current_state = obs.state

            # Apply strategy to get action
            action = strategy_func(current_state)
            actions_taken.append(action)

            # Take action
            obs = client.step(GymAction(action=action))

            episode_reward += obs.reward or 0
            episode_length += 1
            done = obs.done

        # Record episode results
        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)

    # Calculate performance metrics
    avg_length = sum(episode_lengths) / len(episode_lengths)
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    max_length = max(episode_lengths)
    min_length = min(episode_lengths)

    # Action distribution
    action_0_count = actions_taken.count(0)
    action_1_count = actions_taken.count(1)

    return {
        'avg_length': avg_length,
        'avg_reward': avg_reward,
        'max_length': max_length,
        'min_length': min_length,
        'action_distribution': {'left': action_0_count, 'right': action_1_count}
    }
