use crate::ai::player::{AIPlayer, AIPlayerId};
use crate::ai::script::Script;
use crate::human::player::HumanPlayer;
use crate::map::{ComponentType, OBJECTIVE_POS};
use crate::player::{PLAYER_LASER_COUNT, Player};
use crate::sensor::objective::IsInObjective;
use bevy::prelude::{Commands, Res, Resource};
use bevy_math::Vec3;
use std::ops::Deref;

pub const TICK_RATE: f32 = 5.0;
pub const DELTA_TIME: f32 = 1.0 / TICK_RATE;

#[derive(Resource)]
pub enum SimulationConfig {
    Simulation {
        num_ai_players: usize,
        script: Script,
    },
    Game,
}

/// State of the simulation after one update step
#[derive(Resource, Clone, Default)]
pub struct SimulationStepState {
    pub player_states: Vec<PlayerStep>,
}

/// Represents the state and reward of a player after a simulated step
#[repr(C)]
#[derive(Clone)]
pub struct PlayerStep {
    /// Step evaluation
    pub evaluation: PlayerEvaluation,
    /// Player's state after this step
    pub state: PlayerState,
}

#[repr(C)]
#[derive(Clone)]
pub struct PlayerState {
    pub position: Vec3,
    pub ang_velocity: Vec3,
    pub lin_velocity: Vec3,
    pub rotation: Vec3,
    pub lasers: [LaserHit; PLAYER_LASER_COUNT],
}

#[repr(C)]
#[derive(Clone)]
pub struct LaserHit {
    pub distance: f32,                 // is -1 if no hit
    pub component_type: ComponentType, // is ComponentType::None if no hit
}

impl Default for LaserHit {
    fn default() -> Self {
        Self {
            distance: -1.0,
            component_type: ComponentType::None,
        }
    }
}

pub fn spawn_players(mut commands: Commands, simulation: Res<SimulationConfig>) {
    match simulation.deref() {
        SimulationConfig::Simulation { num_ai_players, .. } => {
            commands.spawn_batch(
                (0..*num_ai_players)
                    .into_iter()
                    .map(|id| (Player, AIPlayer::new(AIPlayerId(id)))),
            );
        }
        SimulationConfig::Game => {
            commands.spawn((Player, HumanPlayer));
        }
    }
}

#[repr(C)]
#[derive(Clone, Default)]
pub struct PlayerEvaluation {
    /// Reward of the step. Did the player performed well on that step ?
    reward: f32,
    /// True if the player performed a terminal operation.
    done: bool,
}

pub fn evaluate_player(
    player_previous_state: &PlayerState,
    player_current_state: &PlayerState,
    in_objective: bool,
) -> PlayerEvaluation {
    let previous_objective_distance = OBJECTIVE_POS.distance(player_previous_state.position);
    let current_objective_distance = OBJECTIVE_POS.distance(player_current_state.position);

    let mut reward = 0.0;
    let mut done = false;

    if previous_objective_distance < current_objective_distance {
        reward += 0.1;
    } else {
        reward -= 0.1
    }

    if in_objective {
        reward += 1.0;
        done = true;
    }

    PlayerEvaluation { reward, done }
}
