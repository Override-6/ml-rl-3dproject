mod map;
mod player;
mod ui;
// unused
mod ai;
mod component;
mod human;
mod sensor;
mod simulation;

use crate::ai::input_recorder::GameInputRecorder;
use crate::ai::model_control_pipeline::{
    poll_model_directive, setup_model_connection, sync_state_outputs,
};
use crate::ai::player::follow_all_script;
use crate::ai::script::Script;
use crate::component::arrow::{spawn_arrow_resource, spawn_arrows_to_players};
use crate::component::player_character::spawn_player_characters;
use crate::human::camera_controller::{
    camera_follow, mouse_look, mouse_zoom, spawn_camera_controller,
};
use crate::human::debug_controls::debug_controls;
use crate::human::player::move_player;
use crate::map::{reset_map, setup_map};
use crate::player::{player_collision_detection, reset_players};
use crate::sensor::ground_sensor::ground_sensor_events;
use crate::sensor::player_vibrissae::{debug_render_lasers, update_all_vibrissae_lasers};
use crate::simulation::{
    DELTA_TIME, PlayerStep, SimulationConfig, SimulationState, TICK_RATE, print_simulation_players,
    reset_simulation, simulation_tick, spawn_players,
};
use crate::ui::{setup_ui, update_player_info, update_stats_text};
use bevy::app::{RunMode, ScheduleRunnerPlugin};
use bevy::audio::AudioPlugin;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::pbr::DirectionalLightShadowMap;
use bevy::prelude::*;
use bevy::window::PresentMode;
use bevy::winit::{UpdateMode, WinitSettings};
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::rapier::dynamics::IntegrationParameters;
use bincode::encode_into_slice;
use clap::{Parser, ValueEnum};
use log::info;
use sensor::objective::check_trigger_zone;
use std::cmp::PartialEq;
use std::fs::File;
use std::io::{Read, Write};
use std::time::{Duration, Instant};

const NB_AI_PLAYERS: usize = 200;
const NB_AI_PLAYERS_NO_HEAD: usize = 1000;

#[derive(ScheduleLabel, Clone, Hash, PartialEq, Eq, Debug)]
pub struct PostPhysics;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum HeadMode {
    #[clap(alias = "rt")]
    HeadRealTime,
    #[clap(alias = "rush")]
    HeadRush,
    #[clap(alias = "none")]
    None,
}

#[derive(Parser, Debug)]
struct Args {
    #[clap(long, default_value = "rt", value_name = "head")]
    head: HeadMode,
    #[clap(short = 'p', long = "players", default_value = "200")]
    player_count: usize,
}

fn main() {
    let args = Args::parse();
    let script = read_script_from_file("script.bin");
    let app = create_app(args.head, args.player_count, script);
    run_simulation(app)
}

fn read_script_from_file(file: &str) -> Option<Script> {
    let file = File::open(file);
    let mut script = None;
    if let Ok(mut file) = file {
        let mut script_data = Vec::new();
        file.read_to_end(&mut script_data).unwrap();
        let decoded_script: Script =
            bincode::decode_from_slice(&script_data, bincode::config::standard())
                .unwrap()
                .0;
        script = Some(decoded_script);
    }

    script
}

fn create_app(head: HeadMode, player_count: usize, script: Option<Script>) -> App {
    let mut app = App::new();

    // if true, rush as fast as possible
    let rush = head != HeadMode::HeadRealTime;

    if head == HeadMode::None {
        app.add_plugins(
            MinimalPlugins
                .set(ScheduleRunnerPlugin {
                    run_mode: RunMode::Loop { wait: None },
                })
                .add(TransformPlugin),
        );
    } else {
        app.add_plugins(
            DefaultPlugins
                .build()
                .disable::<AudioPlugin>()
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: PresentMode::Immediate,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
        )
        .insert_resource(WinitSettings {
            focused_mode: UpdateMode::Continuous, // keep running full speed when focused
            unfocused_mode: UpdateMode::Continuous, // keep running full speed when not focused

            ..default()
        })
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_systems(
            Startup,
            (setup_ui, spawn_arrow_resource, spawn_camera_controller),
        )
        .add_systems(
            PostStartup,
            (spawn_arrows_to_players.after(spawn_players), reset_players),
        )
        .add_systems(
            Update,
            (
                mouse_look,
                mouse_zoom,
                update_stats_text,
                update_player_info.after(sync_state_outputs),
                debug_controls,
            ),
        )
        .add_systems(
            PostUpdate,
            (
                camera_follow.after(PhysicsSet::Writeback),
                update_stats_text,
                update_player_info.after(sync_state_outputs),
                debug_render_lasers
                    .after(update_all_vibrissae_lasers)
                    .after(PhysicsSet::Writeback),
            ),
        );
    }

    app.add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_systems(
            Startup,
            (
                (
                    setup_model_connection,
                    spawn_players,
                    spawn_player_characters,
                    sync_state_outputs,
                )
                    .chain(),
                (setup_map, reset_players, reset_map).chain(),
            ),
        )
        .add_systems(Update, cleanup_on_exit)
        .insert_resource(SimulationState::default())
        .insert_resource(
            RapierContextInitialization::InitializeDefaultRapierContext {
                integration_parameters: IntegrationParameters {
                    dt: DELTA_TIME,
                    max_ccd_substeps: 10,
                    ..default()
                },
                rapier_configuration: RapierConfiguration {
                    ..RapierConfiguration::new(100f32)
                },
            },
        )
        .insert_resource(DirectionalLightShadowMap { size: 100 });
    if rush {
        app.insert_resource(TimestepMode::Fixed {
            dt: DELTA_TIME,
            substeps: 1,
        });
    } else {
        // run relative to a fixed tick rate
        app.insert_resource(Time::<Fixed>::from_hz(TICK_RATE as f64));
        app.insert_resource(TimestepMode::Variable {
            max_dt: DELTA_TIME,
            time_scale: 1.0,
            substeps: 1,
        });
    };

    macro_rules! add_game_logic_systems {
        ($rs: expr, $nrs: expr, $systems: expr) => {
            if rush {
                app.add_systems($rs, $systems)
            } else {
                app.add_systems($nrs, $systems)
            }
        };
    }

    if let Some(script) = script {
        app.insert_resource(SimulationConfig::Simulation {
            script,
            num_ai_players: player_count,
        });
        add_game_logic_systems!(Update, FixedUpdate, follow_all_script);
    } else {
        assert_eq!(
            head,
            HeadMode::HeadRealTime,
            "Must be in HeadMode::RealTime if you want to record inputs"
        );
        app.insert_resource(GameInputRecorder::new());
        app.insert_resource(SimulationConfig::Game);
        add_game_logic_systems!(Update, FixedUpdate, move_player.after(poll_model_directive));
    }

    add_game_logic_systems!(
        PreUpdate,
        FixedPreUpdate,
        (
            ground_sensor_events,
            check_trigger_zone,
            player_collision_detection,
            poll_model_directive,
            simulation_tick
        )
    );

    add_game_logic_systems!(
        PostUpdate,
        FixedPostUpdate,
        (
            update_all_vibrissae_lasers.after(PhysicsSet::Writeback),
            sync_state_outputs.after(PhysicsSet::Writeback),
            (
                print_simulation_players,
                reset_simulation,
                reset_players,
                reset_map
            )
                .chain()
                .run_if(|cmd: Res<SimulationState>| cmd.resetting)
                .before(PhysicsSet::SyncBackend),
        )
    );

    app
}

fn run_simulation(mut app: App) {
    println!("sizeof PlayerStep {}", size_of::<PlayerStep>());
    app.run();
}

fn cleanup_on_exit(
    mut events: EventReader<AppExit>,
    recorded_input: Option<Res<GameInputRecorder>>,
) -> Result<(), BevyError> {
    if let Some(event) = events.read().next() {
        info!("App is exiting with status: {:?}", event);

        let Some(recorded_input) = recorded_input else {
            return Ok(());
        };

        let recorded_input: Script = recorded_input.as_ref().into();

        let mut file = File::create("script.bin")?;
        let mut buff = [0; 10000];
        encode_into_slice(&recorded_input, &mut buff, bincode::config::standard())?;
        file.write_all(&buff)?;
    }
    Ok(())
}
