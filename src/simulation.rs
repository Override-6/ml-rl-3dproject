use crate::ai::model_control_pipeline::ModelCommands;
use crate::ai::player::AIPlayer;
use crate::ai::script::Script;
use crate::component::player_character::PlayerInfoUI;
use crate::human::player::HumanPlayer;
use crate::map::{ComponentType, OBJECTIVE_POS};
use crate::player::{PLAYER_LASER_COUNT, Player};
use bevy::prelude::{Commands, Entity, Query, Res, ResMut, Resource, With};
use bevy_math::Vec3;
use std::cmp::Ordering;
use std::ops::Deref;

pub const TICK_RATE: f32 = 10.0;
pub const DELTA_TIME: f32 = 1.0 / TICK_RATE;

#[derive(Resource)]
pub enum SimulationConfig {
    Simulation {
        num_ai_players: usize,
        script: Script,
    },
    Game,
}

#[derive(Resource, Clone, Default)]
pub struct SimulationState {
    pub timestep: u32,
    pub resetting: bool,
    pub debug: DebugVariables,
    step_states: Vec<SimulationStepState>,
}

#[derive(Clone, Default)]
pub struct DebugVariables {
    pub print_all_lasers: bool
}

impl SimulationState {
    pub fn current_step_state(&self) -> &SimulationStepState {
        self.step_states.last().unwrap()
    }

    pub fn previous_step_state(&self) -> Option<&SimulationStepState> {
        self.step_states.get(self.step_states.len() - 2)
    }

    pub fn iter_states(&self) -> impl Iterator<Item = &SimulationStepState> {
        self.step_states.iter()
    }

    pub fn push_step_state(&mut self, state: SimulationStepState) {
        self.step_states.push(state);
    }
}

/// State of the simulation after one update step
#[derive(Clone, Default)]
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
                    .map(|id| (Player::new(id), AIPlayer)),
            );
        }
        SimulationConfig::Game => {
            commands.spawn((Player::new(0), HumanPlayer));
        }
    }
}

#[repr(C)]
#[derive(Clone, Default)]
pub struct PlayerEvaluation {
    /// Reward of the step. Did the player performed well on that step ?
    pub(crate) reward: f32,
    /// True if the player performed a terminal operation.
    pub(crate) done: bool,
}

pub fn evaluate_player(
    player_previous_state: &PlayerState,
    player_current_state: &PlayerState,
    in_objective: bool,
    sim: &SimulationState,
) -> PlayerEvaluation {
    // THIS ONE WORKS WELL
    let previous_objective_distance = OBJECTIVE_POS.distance(player_previous_state.position);
    let current_objective_distance = OBJECTIVE_POS.distance(player_current_state.position);

    let mut reward = 0.0;
    let mut done = false;

    if previous_objective_distance > current_objective_distance {
        reward += 0.1;
    } else {
        reward -= 0.5
    }

    // encourage keeping stuff in sight
    // for laser in player_previous_state.lasers.iter() {
    //     reward += match laser.component_type {
    //         ComponentType::Objective => 20.0,
    //         ComponentType::Object => 1.0,
    //         ComponentType::Ground => 0.0,
    //         ComponentType::Unknown => 0.0,
    //         ComponentType::None => -5.0
    //     };
    // }

    if in_objective {
        reward = 5.0;
        done = true;
    }

    PlayerEvaluation { reward, done }
}

pub fn print_simulation_players(players: Query<&Player>) {
    let timesteps = players
        .iter()
        .map(|p| p.objective_reached_at_timestep)
        .collect::<Vec<_>>();

    let winners = timesteps
        .iter()
        .filter(|&&t| t >= 0)
        .copied()
        .collect::<Vec<_>>();
    let fastest = winners.iter().min().unwrap_or(&-1);
    let slowest = winners.iter().max().unwrap_or(&-1);
    let total_winners = winners.len();
    let average_winners =
        winners.iter().filter(|&&t| t >= 0).sum::<i32>() as f32 / total_winners as f32;

    println!(
        "fastest: {fastest} slowest: {slowest} total winners: {total_winners} winners to complete avg: {average_winners} ticks"
    );
}

pub fn reset_simulation(mut sim: ResMut<SimulationState>) {

}

pub fn clean_simulation(
    mut sim: ResMut<SimulationState>,
    mut commands: Commands,
    players: Query<Entity, With<Player>>,
    players_info: Query<Entity, With<PlayerInfoUI>>,
) {
    sim.resetting = false;
    players.iter().chain(players_info.iter()).for_each(|e| {
        commands.entity(e).despawn();
    });
    commands.insert_resource(SimulationState {
        debug: sim.debug.clone(),
        ..Default::default()
    });
}

pub fn simulation_tick(mut simulation_state: ResMut<SimulationState>) {
    simulation_state.timestep += 1;
}
