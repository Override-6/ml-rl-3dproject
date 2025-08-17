use crate::ai::model_control_pipeline::SimulationPlayersInputs;
use crate::ai::player::AIPlayer;
use crate::ai::script::Script;
use crate::human::player::HumanPlayer;
use crate::map::{ComponentType, OBJECTIVE_POS};
use crate::player::{PLAYER_LASER_COUNT, Player};
use bevy::prelude::{Commands, Query, Res, ResMut, Resource, With};
use bevy_math::Vec3;
use bevy_rapier3d::dynamics::Velocity;
use bevy_rapier3d::na::{Quaternion, UnitQuaternion};
use bevy_rapier3d::plugin::WriteRapierContext;
use bevy_rapier3d::prelude::RapierRigidBodyHandle;
use bevy_rapier3d::rapier::prelude as rapier;
use rand::Rng;
use std::ops::Deref;
use crate::ai::input::{Input, InputSet};

pub const TICK_RATE: f32 = 10.0;
pub const DELTA_TIME: f32 = 1.0 / TICK_RATE;

pub const TOTAL_STEP_AMOUNT: u32 = (TICK_RATE * 10.0) as u32;

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

#[derive(Clone)]
pub struct DebugVariables {
    pub print_all_lasers: bool,
    pub print_players_rewards: bool,
    pub did_reset: bool,
}

impl Default for DebugVariables {
    fn default() -> Self {
        Self {
            print_all_lasers: false,
            print_players_rewards: true,
            did_reset: false,
        }
    }
}

impl SimulationState {
    pub fn current_step_state(&self) -> &SimulationStepState {
        self.step_states.last().unwrap()
    }

    pub fn previous_step_state(&self) -> Option<&SimulationStepState> {
        if self.step_states.len() < 2 {
            return None;
        }
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

fn go_to_objective_evaluation(
    player_previous_state: &PlayerState,
    player_current_state: &PlayerState,
    in_objective: bool,
) -> PlayerEvaluation {
    let previous_objective_distance = OBJECTIVE_POS.distance(player_previous_state.position);
    let current_objective_distance = OBJECTIVE_POS.distance(player_current_state.position);

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

fn keep_objects_in_sight_evaluation(
    player_current_state: &PlayerState,
    sim: &SimulationState,
    inputs: InputSet,
) -> PlayerEvaluation {
    let mut reward = 0.0;
    let mut done = false;

    for laser in player_current_state.lasers.iter() {
        reward += match laser.component_type {
            ComponentType::Objective => 2.0,
            ComponentType::Object => 0.5,
            ComponentType::Ground => 0.5,
            ComponentType::Unknown => 0.0,
            ComponentType::None => -0.5,
        };
    }

    // // avoid jumping at the beginning
    // if inputs & Input::Jump != 0 && sim.timestep == 3 {
    //     done = true;
    //     reward = -10.0;
    // }

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
    keep_objects_in_sight_evaluation(player_current_state, sim, inputs)
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

pub fn reset_simulation(
    mut players: Query<(&RapierRigidBodyHandle, &mut Player, Option<&mut Velocity>), With<Player>>,
    mut sim: ResMut<SimulationState>,
    mut rapier_ctx: WriteRapierContext,
) -> bevy::prelude::Result<()> {
    assert_eq!(sim.timestep, TOTAL_STEP_AMOUNT + 1);

    let mut context = rapier_ctx.single_mut()?;
    let mut rng = rand::rng();

    for (handle, mut player, vel_opt) in players.iter_mut() {
        let handle = handle.0;

        // use the same path you used originally to get the rb
        if let Some(rb) = context.rigidbody_set.bodies.get_mut(handle) {
            // 1) Temporarily make the body Fixed so the solver can't produce impulses while we move it
            rb.set_body_type(rapier::RigidBodyType::Fixed, true);

            // 2) Teleport translation & rotation
            rb.set_translation(
                rapier::Vector::new(
                    rng.random_range(-300.0f32..300.0f32),
                    10.0f32,
                    rng.random_range(-300.0f32..300.0f32),
                ),
                /*wake*/ false,
            );

            let yaw_deg = rng.random_range(0.0f32..360.0f32);
            let yaw = yaw_deg.to_radians();
            let unit_q = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0);
            rb.set_rotation(unit_q, /*wake*/ false);

            // 3) Zero rapier velocities (defensive)
            rb.set_linvel(rapier::Vector::zeros(), /*wake*/ false);
            rb.set_angvel(rapier::Vector::zeros(), /*wake*/ false);

            // 4) Restore to Dynamic, but DON'T wake the body (false)
            //    This keeps the body asleep until you explicitly wake a subset per-frame.
            rb.set_body_type(rapier::RigidBodyType::Dynamic, /*wake*/ false);
        }

        // 5) Reset gameplay state
        player.freeze = false;
        player.objective_reached_at_timestep = -1;

        // 6) Also reset the Bevy Velocity component (if present) so ECS view stays consistent
        if let Some(mut vel) = vel_opt {
            *vel = Velocity::default();
        }
    }

    // clear sim state and leave resetting flags
    sim.resetting = false;
    sim.timestep = 0;
    sim.step_states.clear();

    sim.debug.did_reset = true;

    Ok(())
}

pub fn simulation_tick(mut simulation_state: ResMut<SimulationState>) {
    simulation_state.timestep += 1;
}
