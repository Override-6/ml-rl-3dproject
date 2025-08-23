use crate::ai::player::AIPlayer;
use crate::ai::script::Script;
use crate::human::player::HumanPlayer;
use crate::map::{ComponentType, MapObstacleSet, ObstacleFaceSet, COMPONENTS_COUNT};
use crate::player::{Player, PLAYER_LASER_COUNT};
use bevy::prelude::{Commands, Query, Res, ResMut, Resource};
use bevy_math::Vec3;
use std::ops::Deref;
use crate::ai::evaluation::PlayerEvaluation;

pub const TICK_RATE: f32 = 10.0;
pub const DELTA_TIME: f32 = 1.0 / TICK_RATE;

pub const TOTAL_STEP_AMOUNT: u32 = (TICK_RATE * 25.0) as u32;

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
    pub epoch: u32,
    pub resetting: bool,
    pub debug: DebugVariables,
    pub step_states: Vec<SimulationStepResult>,
    pub rt_info: Vec<SimulationRuntimeInfo>
}

#[derive(Clone)]
pub struct DebugVariables {
    pub print_all_lasers: bool,
    pub print_players_rewards: bool,
}

impl Default for DebugVariables {
    fn default() -> Self {
        Self {
            print_all_lasers: false,
            print_players_rewards: true,
        }
    }
}

impl SimulationState {

    pub fn previous_step_rt_info(&self) -> Option<&SimulationRuntimeInfo> {
        if self.rt_info.len() < 2 {
            return None;
        }
        let idx = self.rt_info.len() - 2;
        self.rt_info.get(idx)
    }

    pub fn current_step_rt_info_mut(&mut self) -> &mut SimulationRuntimeInfo {
        self.rt_info.last_mut().unwrap()
    }

    pub fn current_step_rt_info(&self) -> &SimulationRuntimeInfo {
        self.rt_info.last().unwrap()
    }

    pub fn current_step_state(&self) -> &SimulationStepResult {
        self.step_states.last().unwrap()
    }

    pub fn iter_states(&self) -> impl Iterator<Item = &SimulationStepResult> {
        self.step_states.iter()
    }

    pub fn push_step_state(&mut self, state: SimulationStepResult) {
        self.step_states.push(state);
    }
}

#[derive(Clone, Default)]
pub struct SimulationRuntimeInfo {
    /// All discovered faces of players
    pub discovered_obstacles: Vec<MapObstacleSet>,
}

/// State of the simulation after one update step
#[derive(Clone, Default)]
pub struct SimulationStepResult {
    pub player_states: Vec<PlayerStepResult>,
}

/// Represents the state and reward of a player after a simulated step
#[repr(C)]
#[derive(Clone)]
pub struct PlayerStepResult {
    /// Step evaluation
    pub evaluation: PlayerEvaluation,
    /// Player's state after this step
    pub state: PlayerState,
}

#[repr(C)]
#[derive(Clone, Default)]
/// Model input for one player and for one step
pub struct PlayerState {
    pub position: Vec3,
    pub rotation: Vec3,
    pub ang_velocity: Vec3,
    pub lin_velocity: Vec3,
    pub laser_hit: [LaserHit; PLAYER_LASER_COUNT],
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

pub fn reset_simulation(mut sim: ResMut<SimulationState>) -> bevy::prelude::Result<()> {
    // clear sim state and leave resetting flags
    sim.resetting = false;
    sim.timestep = 0;
    sim.step_states.clear();
    sim.rt_info.clear();
    sim.epoch += 1;

    Ok(())
}

pub fn simulation_tick(mut simulation_state: ResMut<SimulationState>) {
    simulation_state.timestep += 1;
}

pub fn prepare_sim_state(
    mut simulation_state: ResMut<SimulationState>,
    config: Res<SimulationConfig>,
) {
    let SimulationConfig::Simulation { num_ai_players, .. } = *config else {
        panic!("SimulationConfig::Simulation expected");
    };
    let last = simulation_state.rt_info.last().cloned().unwrap_or_else(|| SimulationRuntimeInfo {
        discovered_obstacles: vec![MapObstacleSet {
            faces: vec![ObstacleFaceSet::default(); COMPONENTS_COUNT + 1],
        }; num_ai_players],
    });
    simulation_state.rt_info.push(last);
    // println!("Simulation state prepared");
}