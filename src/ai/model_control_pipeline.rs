use crate::ai::input::InputSet;
use crate::map::ComponentType;
use crate::player::Player;
use crate::sensor::objective::IsInObjective;
use crate::sensor::player_vibrissae::PlayerVibrissae;
use crate::simulation::{
    LaserHit, PlayerEvaluation, PlayerState, PlayerStep, SimulationStepState, evaluate_player,
};
use bevy::prelude::{Commands, Query, Res, ResMut, Resource, Transform, With};
use bevy_math::u32;
use bevy_rapier3d::prelude::{Sleeping, Velocity};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::thread::sleep;
use std::time::Duration;

#[derive(Resource)]
pub struct ModelCommands {
    stream: TcpStream,
    pub current_step_is_reset: bool,
}

#[derive(Resource)]
pub struct SimulationPlayersInputs {
    pub inputs: Vec<InputSet>,
}

pub enum ModelDirective {
    ResetSimulation,
    NexStep(Vec<InputSet>),
}

impl ModelCommands {
    pub fn send_step_outputs(
        &mut self,
        state: SimulationStepState,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let players_states = state.player_states.as_slice();
        let count_bytes = &u32::to_le_bytes(players_states.len() as u32);
        self.stream.write_all(count_bytes)?;
        let byte_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                players_states.as_ptr() as *const u8,
                players_states.len() * size_of::<PlayerStep>(),
            )
        };
        self.stream.write_all(byte_slice)?;
        Ok(())
    }

    fn poll_model_outputs(&mut self) -> std::io::Result<Vec<InputSet>> {
        let mut head_buff = [0u8; size_of::<u32>()];
        self.stream.read_exact(&mut head_buff)?;
        let item_count = u32::from_le_bytes(head_buff) as usize;

        let mut inputs = vec![0u8 as InputSet; item_count];
        self.stream.read_exact(&mut inputs)?;

        Ok(inputs)
    }

    pub fn pull_model_directive(&mut self) -> std::io::Result<ModelDirective> {
        let mut packet_type_buff = [0u8; size_of::<u32>()];
        self.stream.read_exact(&mut packet_type_buff)?;
        let packet_type = u32::from_le_bytes(packet_type_buff);

        match packet_type {
            0 => Ok(ModelDirective::ResetSimulation),
            1 => self.poll_model_outputs().map(ModelDirective::NexStep),
            _ => panic!("Received unexpected packet type {packet_type}"),
        }
    }
}

pub fn setup_model_connection(mut commands: Commands) {
    let stream = TcpStream::connect("localhost:9999").unwrap();
    commands.insert_resource(ModelCommands {
        stream,
        current_step_is_reset: false,
    });
    sleep(Duration::from_secs(1));
}

pub fn sync_state_outputs(
    last_state: Option<Res<SimulationStepState>>,
    mut commands: Commands,
    mut model_commands: ResMut<ModelCommands>,
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
    print!("Sending state...");
    let state = evaluate_players(last_state, players, model_commands.current_step_is_reset);
    commands.insert_resource(state.clone());
    model_commands.send_step_outputs(state).unwrap();
    println!("\rSent state.")
}

fn evaluate_players(
    last_state: Option<Res<SimulationStepState>>,
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
    just_reset: bool,
) -> SimulationStepState {
    let player_states: Vec<_> = players
        .iter_mut()
        .map(|(vibrissae, velocity, transform, itz, player, sleeping)| {
            (
                PlayerState {
                    position: transform.translation,
                    rotation: transform.rotation.xyz(),
                    ang_velocity: velocity.angvel,
                    lin_velocity: velocity.linvel,
                    lasers: vibrissae.lasers.clone().map(|s| {
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
        .map(|(state, itz, mut player, mut velocity, mut sleeping)| {
            let evaluation = last_state
                .as_ref()
                .filter(|_| !just_reset)
                .map_or(PlayerEvaluation::default(), |ls| {
                    evaluate_player(&ls.player_states[player.id].state, &state, itz.0)
                });
            if evaluation.done && !player.freeze {
                println!("Froze player {} as it reached a terminal state.", player.id);
                player.freeze = true;
                sleeping.sleeping = true;
                *velocity = Velocity::default();
            }
            PlayerStep { evaluation, state }
        })
        .collect();

    SimulationStepState { player_states }
}

pub fn poll_model_directive(
    mut model_commands: ResMut<ModelCommands>,
    mut commands: Commands,
) {
    print!("Waiting for next model directive...");
    let directive = model_commands.pull_model_directive().unwrap();

    match directive {
        ModelDirective::ResetSimulation => {
            println!("\rReceived reset packet, resetting simulation...");
            model_commands.current_step_is_reset = true;
        }
        ModelDirective::NexStep(inputs) => {
            println!("\rReceived model outputs, advancing simulation...");
            commands.insert_resource(SimulationPlayersInputs { inputs });
        }
    }
}
