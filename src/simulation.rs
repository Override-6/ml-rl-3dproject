use crate::ai::player::{AIPlayer, AIPlayerId};
use crate::ai::script::Script;
use crate::human::player::HumanPlayer;
use crate::map::ComponentType;
use crate::player::{Player, PLAYER_LASER_COUNT};
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
pub struct SimulationStepState {
    pub player_states: Vec<PlayerState>
}

#[repr(C)]
pub struct PlayerState {
    pub position: Vec3,
    pub ang_velocity: Vec3,
    pub lin_velocity: Vec3,
    pub rotation: Vec3,
    pub lasers: [LaserHit; PLAYER_LASER_COUNT],
}

#[repr(C)]
pub struct LaserHit {
    pub distance: f32, // is -1 if no hit
    pub component_type: ComponentType, // is ComponentType::None if no hit
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
