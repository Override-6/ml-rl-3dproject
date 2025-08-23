use crate::ai::input::InputSet;
use crate::map::{ComponentType, MAP_SQUARE_SIZE};
use crate::player::{Player, LASER_LENGTH};
use crate::simulation::{PlayerState, SimulationState};
use bevy::prelude::Vec3;

#[repr(C)]
#[derive(Clone, Default)]
pub struct PlayerEvaluation {
    /// Reward of the step. Did the player performed well on that step ?
    pub(crate) reward: f32,
    /// True if the player performed a terminal operation.
    pub(crate) done: bool,
}

/// Encourages the player to reach objective
fn go_to_objective_evaluation(
    player_previous_state: &PlayerState,
    player_current_state: &PlayerState,
    in_objective: bool,
    objective_pos: Vec3,
) -> PlayerEvaluation {
    let previous_objective_distance = objective_pos.distance(player_previous_state.position);
    let current_objective_distance = objective_pos.distance(player_current_state.position);

    let mut reward = 0.0;
    let mut done = false;

    if previous_objective_distance > current_objective_distance {
        reward += 0.1;
    } else {
        reward -= 0.5
    }

    if in_objective {
        reward = 5.0;
        done = true;
    }

    PlayerEvaluation { reward, done }
}

/// Encourages the player to keep object and objective components in any sight
fn keep_objects_in_sight_evaluation(
    player_current_state: &PlayerState,
    in_objective: bool,
) -> PlayerEvaluation {
    let mut reward = 0.0;
    let mut done = false;

    for laser in player_current_state.laser_hit.iter() {
        reward += match laser.component_type {
            ComponentType::Objective => 2.0,
            ComponentType::Obstacle => 0.5,
            ComponentType::Ground => 0.5,
            ComponentType::Unknown => 0.0,
            ComponentType::None => -0.5,
        };
    }

    if in_objective {
        reward += 10.0;
        done = true;
    }

    PlayerEvaluation { reward, done }
}

/// Encourages the player to always aim at the objective.
/// Also encourages the player to be as close as possible until being inside (based on front sight hit distance)
fn keep_objective_in_front_sight_and_go_to_evaluation(
    player_current_state: &PlayerState,
    in_objective: bool,
) -> PlayerEvaluation {
    let mut reward = 0.0;
    let mut done = false;

    const LASER_MINIMAL_DISTANCE: f32 = LASER_LENGTH;

    for (idx, hit) in player_current_state.laser_hit.iter().enumerate() {
        let is_front_laser = idx == 1; //laser at index one is the front sight
        let distance_reward = 1.0 - (hit.distance / LASER_MINIMAL_DISTANCE);
        reward += match hit.component_type {
            // ComponentType::Objective if is_front_laser => 1.0 + 2.5 * distance_reward,
            ComponentType::Objective => 1.0 + 4.0 * distance_reward,
            ComponentType::Obstacle => 0.0, // 1 max per lasers, 4 max
            ComponentType::Ground => 0.0,
            ComponentType::Unknown => 0.0,
            ComponentType::None => -0.5, // -2 max
        };
    }

    if in_objective {
        reward += 10.0;
        done = true;
    }

    PlayerEvaluation { reward, done }
}

fn avoid_obstacles_and_explore_evaluation(
    player_current_state: &PlayerState,
    in_objective: bool,
    player: &Player,
    sim: &SimulationState,
) -> PlayerEvaluation {
    let mut reward = 0.0;
    let mut done = false;

    // force the model to act
    reward -= 0.01;

    const LASER_MINIMAL_DISTANCE: f32 = LASER_LENGTH;

    for hit in player_current_state.laser_hit.iter() {
        match hit.component_type {
            ComponentType::Objective => {
                if hit.distance > 0.0 {
                    // Small shaping reward for being closer to objective
                    let closeness = 1.0 - (hit.distance / LASER_MINIMAL_DISTANCE);
                    reward += 0.5 * closeness.max(0.0);
                }
            }
            ComponentType::Obstacle => {
                if hit.distance > 0.0 {
                    // Smooth penalty for being near obstacles
                    let proximity = 1.0 - (hit.distance / 20.0);
                    reward -= 0.5 * proximity.clamp(0.0, 1.0);
                }
            }
            // Ground, Unknown, None => no shaping reward
            _ => {}
        }
    }

    let current_step_rt_info = sim.current_step_rt_info();
    if let Some(previous_step_rt_info) = sim.previous_step_rt_info() {
        let current_discovered_obstacles = &current_step_rt_info.discovered_obstacles[player.id];
        let previous_discovered_obstacles = &previous_step_rt_info.discovered_obstacles[player.id];

        let face_comparison = current_discovered_obstacles.faces.iter().zip(previous_discovered_obstacles.faces.iter());

        // reward when a new face is discovered
        for (current, previous) in face_comparison {
            if current != previous {
                assert!(current > previous);
                reward += 1.0;
            }
        }
    }

    if player.touching_obstacle {
        reward = -10.0;
        done = true;
    } else if in_objective {
        reward = 10.0;
        done = true;
    }

    PlayerEvaluation { reward, done }
}

/// As evaluation is not executed on the very first step (as it's a "zeroize" step which outputs initial positions)
/// This allows safely unwrapping sim.previous_step_state() and sim.previous_step_rt_info()
pub fn evaluate_player(
    current_state: &PlayerState,
    player: &Player,
    in_objective: bool,
    sim: &SimulationState,
    inputs: InputSet,
) -> PlayerEvaluation {
    let previous_state = &sim.current_step_state().player_states[player.id].state;
    // go_to_objective_evaluation(previous_state, current_state, in_objective, Vec3::new(-MAP_SQUARE_SIZE, 0.0, -MAP_SQUARE_SIZE))
    // keep_objects_in_sight_evaluation(player_current_state, in_objective)
    // keep_objective_in_front_sight_and_go_to_evaluation(player_current_state, in_objective)
    avoid_obstacles_and_explore_evaluation(current_state, in_objective, player, sim)
}
