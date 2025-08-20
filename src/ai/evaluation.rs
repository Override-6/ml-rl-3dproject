use crate::ai::input::InputSet;
use crate::map::ComponentType;
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
            ComponentType::Object => 0.5,
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
/// Also encourages the player to be as close as possible until being inside (based on front sight
fn keep_objective_in_front_sight_and_go_to_evaluation(
    player_current_state: &PlayerState,
    in_objective: bool,
) -> PlayerEvaluation {
    let mut reward = 0.0;
    let mut done = false;

    const LASER_MINIMAL_DISTANCE: f32 = 400.0;

    for (idx, hit) in player_current_state.laser_hit.iter().enumerate() {
        let is_front_laser = idx == 1; //laser at index one is the front sight
        let distance_reward = 1.0 - (hit.distance / LASER_MINIMAL_DISTANCE);
        reward += match hit.component_type {
            ComponentType::Objective if is_front_laser => 5.0 + 5.0 * distance_reward,
            ComponentType::Objective => 2.5 + 2.0 * distance_reward,
            ComponentType::Object => 0.5 + 0.75 * distance_reward,
            ComponentType::Ground => 0.0,
            ComponentType::Unknown => 0.0,
            ComponentType::None => -0.5,
        };
    }

    if in_objective {
        reward += 30.0;
        done = true;
    }


    PlayerEvaluation { reward, done }
}

pub fn evaluate_player(
    player_previous_state: &PlayerState,
    player_current_state: &PlayerState,
    in_objective: bool,
    sim: &SimulationState,
    inputs: InputSet,
) -> PlayerEvaluation {
    // go_to_objective_evaluation(player_previous_state, player_current_state, in_objective)
    // keep_objects_in_sight_evaluation(player_current_state, in_objective)
    keep_objective_in_front_sight_and_go_to_evaluation(player_current_state, in_objective)
}