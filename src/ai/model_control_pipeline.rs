use crate::ai::evaluation::{PlayerEvaluation, evaluate_player};
use crate::ai::input::{Input, PulseInputSet};
use crate::map::ComponentType;
use crate::player::Player;
use crate::sensor::objective::IsInObjective;
use crate::sensor::player_vibrissae::PlayerVibrissae;
use crate::simulation::{
    LaserHit, PlayerState, PlayerStepResult, SimulationState, SimulationStepResult,
};
use bevy::prelude::{Commands, Query, Res, ResMut, Resource, Transform, Vec3, With};
use bevy::reflect::List;
use bevy_math::u32;
use bevy_rapier3d::prelude::{Sleeping, Velocity};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::ops::{Deref, DerefMut};
use std::thread::sleep;
use std::time::Duration;

#[derive(Resource)]
pub struct ModelCommands {
    stream: TcpStream,
}

#[derive(Resource)]
pub struct SimulationPlayersInputs {
    pub inputs: Vec<Input>,
}

pub enum ModelDirective {
    ResetSimulation,
    NexStep(Vec<PulseInputSet>, Vec<f32>),
}

impl ModelCommands {
    pub fn send_step_outputs(
        &mut self,
        state: SimulationStepResult,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let players_states = state.player_states.as_slice();
        let mut buf = Vec::with_capacity(4 + players_states.len() * size_of::<PlayerStepResult>());
        buf.extend_from_slice(&(players_states.len() as u32).to_le_bytes());
        let bytes = unsafe {
            std::slice::from_raw_parts(
                players_states.as_ptr() as *const u8,
                size_of_val(players_states),
            )
        };
        buf.extend_from_slice(bytes);

        self.stream.write_all(&buf)?;

        Ok(())
    }

    fn poll_model_outputs(&mut self) -> std::io::Result<(Vec<PulseInputSet>, Vec<f32>)> {
        let mut head_buff = [0u8; size_of::<u32>()];
        self.stream.read_exact(&mut head_buff)?;
        let item_count = u32::from_le_bytes(head_buff) as usize;

        let mut pulse_inputs = vec![0u8 as PulseInputSet; item_count];
        self.stream.read_exact(&mut pulse_inputs)?;

        let mut laser_directions = vec![0u8; item_count * size_of::<f32>()];
        self.stream.read_exact(&mut laser_directions)?;

        let laser_directions = unsafe {
            std::slice::from_raw_parts(laser_directions.as_ptr() as *const f32, item_count)
                .to_vec()
        };

        Ok((pulse_inputs, laser_directions))
    }

    pub fn poll_model_directive(&mut self) -> std::io::Result<ModelDirective> {
        let mut packet_type_buff = [0u8; size_of::<u32>()];
        self.stream.read_exact(&mut packet_type_buff)?;
        let packet_type = u32::from_le_bytes(packet_type_buff);

        let result = match packet_type {
            0 => Ok(ModelDirective::ResetSimulation),
            1 => self
                .poll_model_outputs()
                .map(|(p, d)| ModelDirective::NexStep(p, d)),
            _ => panic!("Received unexpected packet type {packet_type}"),
        };
        result
    }
}

pub fn setup_model_connection(mut commands: Commands) {
    let mut stream = TcpStream::connect("localhost:9999").unwrap();
    stream.set_nodelay(true).unwrap();
    commands.insert_resource(ModelCommands {
        stream: stream.try_clone().unwrap(),
    });
    sleep(Duration::from_secs(1));
}

pub fn sync_state_outputs(
    mut state: ResMut<SimulationState>,
    mut model_commands: ResMut<ModelCommands>,
    inputs: Option<Res<SimulationPlayersInputs>>,
    players: Query<
        (
            &PlayerVibrissae,
            &mut Velocity,
            &Transform,
            &IsInObjective,
            &mut Player,
            &mut Sleeping,
        ),
        With<Player>,
    >,
) {
    let step_state = evaluate_players(players, state.deref(), inputs.as_ref());
    state.push_step_state(step_state.clone());
    model_commands.send_step_outputs(step_state).unwrap();
}

fn evaluate_players(
    mut players: Query<
        (
            &PlayerVibrissae,
            &mut Velocity,
            &Transform,
            &IsInObjective,
            &mut Player,
            &mut Sleeping,
        ),
        With<Player>,
    >,
    sim: &SimulationState,
    inputs: Option<&Res<SimulationPlayersInputs>>,
) -> SimulationStepResult {
    // let is_last_step = sim.timestep == TOTAL_STEP_AMOUNT;

    let last_state = sim.step_states.last();

    let mut player_states: Vec<_> = players.iter_mut().collect::<Vec<_>>();

    player_states.sort_by_key(|(_, _, _, _, p, _)| p.id);

    let player_states: Vec<_> = player_states
        .iter_mut()
        .map(|(vibrissae, velocity, transform, itz, player, sleeping)| {
            (
                PlayerState {
                    position: transform.translation,
                    rotation: transform.rotation.xyz(),
                    ang_velocity: velocity.angvel,
                    lin_velocity: velocity.linvel,
                    laser_hit: vibrissae.lasers.clone().map(|s| {
                        s.hit.map_or(
                            LaserHit {
                                component_type: ComponentType::None,
                                distance: -1.0,
                            },
                            |h| LaserHit {
                                component_type: h.comp_type,
                                distance: h.distance,
                            },
                        )
                    }),
                },
                itz,
                player,
                velocity,
                sleeping,
            )
        })
        .map(|(state, itz, player, velocity, sleeping)| {
            let evaluation = last_state
                .as_ref()
                .map_or(PlayerEvaluation::default(), |ls| {
                    let last_player_step = &ls.player_states[player.id];
                    // keep null reward when the player already won
                    if last_player_step.evaluation.done {
                        last_player_step.evaluation.clone()
                    } else {
                        let inputs = inputs.map_or(0, |inputs| inputs.inputs[player.id].pulse);
                        evaluate_player(&state, player, itz.0, sim, inputs)
                    }
                });
            if evaluation.done {
                player.freeze = true;
                if evaluation.reward >= 0.0 {
                    player.objective_reached_at_timestep = sim.timestep as i32;
                }
                sleeping.sleeping = true;
                *velocity.deref_mut() = Velocity::default();
            }
            PlayerStepResult { evaluation, state }
        })
        .collect();

    // let living_players = players
    //     .iter()
    //     .filter(|(_, _, _, _, p, _)| !p.freeze)
    //     .count();
    //
    // let living_players_dones = player_states
    //     .iter()
    //     .filter(|s| !s.evaluation.done)
    //     .count();
    //
    // println!("Living players (player.freeze): {}", living_players);
    // println!("Living players (dones): {living_players_dones}");

    SimulationStepResult { player_states }
}

pub fn poll_model_directive(
    mut model_commands: ResMut<ModelCommands>,
    mut sim: ResMut<SimulationState>,
    mut commands: Commands,
) {
    // println!("Waiting for next model directive...");
    let directive = model_commands.poll_model_directive().unwrap();

    match directive {
        ModelDirective::ResetSimulation => {
            // println!("\rReceived reset packet, resetting simulation...");
            sim.resetting = true;
        }
        ModelDirective::NexStep(pulses, laser_dir) => {
            // println!("\rReceived model outputs, advancing simulation...");

            let inputs = pulses
                .into_iter()
                .zip(laser_dir.into_iter())
                .map(|(p, d)| Input {
                    pulse: p,
                    laser_dir: d,
                })
                .collect::<Vec<_>>();

            commands.insert_resource(SimulationPlayersInputs { inputs });
        }
    }
}
