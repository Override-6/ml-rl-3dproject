use crate::ai::input::InputSet;
use crate::map::ComponentType;
use crate::player::Player;
use crate::sensor::objective::IsInObjective;
use crate::sensor::player_vibrissae::PlayerVibrissae;
use crate::simulation::{evaluate_player, reset_simulation, LaserHit, PlayerEvaluation, PlayerState, PlayerStep, SimulationStepState};
use bevy::prelude::{Commands, Query, Res, ResMut, Resource, Transform, With};
use bevy::render::render_resource::encase::private::RuntimeSizedArray;
use bevy_math::u32;
use bevy_rapier3d::prelude::Velocity;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::thread::sleep;
use std::time::Duration;

#[derive(Resource)]
pub struct ModelCommands {
    stream: TcpStream,
}

#[derive(Resource)]
pub struct SimulationPlayersInputs {
    pub inputs: Vec<InputSet>,
}

enum ModelDirective {
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

    pub fn poll_model_outputs(&mut self) -> std::io::Result<Vec<InputSet>> {
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
    commands.insert_resource(ModelCommands { stream });
    sleep(Duration::from_secs(1));
}

pub fn sync_state_outputs(
    last_state: Option<Res<SimulationStepState>>,
    mut commands: Commands,
    mut model_commands: ResMut<ModelCommands>,
    mut players: Query<
        (
            &PlayerVibrissae,
            &mut Velocity,
            &Transform,
            &IsInObjective,
            &mut Player
        ),
        With<Player>,
    >,
) {
    print!("Sending state...");
    let player_states = players
        .iter_mut()
        .map(|(vibrissae, velocity, transform, itz, player)| {
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
                velocity
            )
        })
        .enumerate()
        .map(|(player_idx, (state, itz, mut player, mut velocity))| {
            let evaluation = last_state
                .as_ref()
                .map_or(PlayerEvaluation::default(), |ls| {
                    evaluate_player(&ls.player_states[player_idx].state, &state, itz.0)
                });
            if evaluation.done {
                println!("Froze player {player_idx} as it reached a terminal state.");
                player.freeze = true;
                *velocity = Velocity::default();
            }
            print!("{} ", evaluation.reward);
            PlayerStep { evaluation, state }
        })
        .collect();
    println!();

    let state = SimulationStepState { player_states };
    commands.insert_resource(state.clone());
    model_commands.send_step_outputs(state).unwrap();
    println!("\rSent state.")
}

pub fn sync_model_outputs(
    mut model_commands: ResMut<ModelCommands>,
    mut commands: Commands,
    players_query: Query<(&mut Transform, &mut Velocity, &mut Player), With<Player>>,
) {
    print!("Waiting for next model directive...");
    let directive = model_commands.pull_model_directive().unwrap();

    match directive {
        ModelDirective::ResetSimulation => {
            println!("\rReceived reset packet, resetting simulation...");
            reset_simulation(players_query);
        }
        ModelDirective::NexStep(inputs) => {
            println!("\rReceived model outputs, advancing simulation...");
            commands.insert_resource(SimulationPlayersInputs { inputs });
        }
    }
}
