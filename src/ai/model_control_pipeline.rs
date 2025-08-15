use crate::ai::input::InputSet;
use crate::map::ComponentType;
use crate::player::Player;
use crate::sensor::objective::IsInObjective;
use crate::sensor::player_vibrissae::PlayerVibrissae;
use crate::simulation::{
    LaserHit, PlayerEvaluation, PlayerState, PlayerStep, SimulationStepState, evaluate_player,
};
use bevy::prelude::{Commands, GlobalTransform, Query, Res, ResMut, Resource, With};
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
        println!("Pulling model outputs");
        let mut head_buff = [0u8; size_of::<u32>()];
        self.stream.read_exact(&mut head_buff)?;
        let item_count = u32::from_le_bytes(head_buff) as usize;

        let mut inputs = vec![0u8 as InputSet; item_count];
        self.stream.read_exact(&mut inputs)?;

        Ok(inputs)
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
    players: Query<
        (
            &PlayerVibrissae,
            &Velocity,
            &GlobalTransform,
            &IsInObjective,
        ),
        With<Player>,
    >,
) {
    let player_states = players
        .iter()
        .map(|(vibrissae, velocity, transform, itz)| {
            (
                PlayerState {
                    position: transform.translation(),
                    rotation: transform.rotation().xyz(),
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
            )
        })
        .enumerate()
        .map(|(player_idx, (state, itz))| PlayerStep {
            evaluation: last_state
                .as_ref()
                .map_or(PlayerEvaluation::default(), |ls| {
                    evaluate_player(&ls.player_states[player_idx].state, &state, itz.0)
                }),
            state,
        })
        .collect();

    let state = SimulationStepState { player_states };
    commands.insert_resource(state.clone());
    model_commands.send_step_outputs(state).unwrap();
    println!("Sent state, awaiting models outputs...")
}

pub fn sync_model_outputs(mut model_commands: ResMut<ModelCommands>, mut commands: Commands) {
    let inputs = model_commands.poll_model_outputs().unwrap();
    commands.insert_resource(SimulationPlayersInputs { inputs });
    println!("Received model outputs, advancing simulation...")
}
